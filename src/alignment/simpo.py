"""SimPO — Simple Preference Optimization (NeurIPS 2024).

Reference-free DPO variant using average log-prob reward.
No reference model or KL regularization needed.

Key differences from DPO:
- No reference model or KL term
- Reward = average log-probability (length-normalized), not sum
- Margin γ (gamma) enforces a target reward gap between chosen/rejected

Loss: -log sigmoid((beta * avg_logp_chosen - beta * avg_logp_rejected - gamma) / 1)
      = -log sigmoid(beta * (avg_logp_chosen - avg_logp_rejected) - gamma)

References:
    - SimPO: Simple Preference Optimization with a Reference-Free Reward (NeurIPS 2024)
      https://arxiv.org/abs/2405.14734
    - AdaDPO: Self-Adaptive DPO with Balancing Coefficients (arxiv 2605.28440)
      https://arxiv.org/abs/2605.28440
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    """SimPO loss module.

    Args:
        beta: Scaling factor for the reward signal. Default: 2.0.
        gamma: Target reward margin between chosen and rejected. Default: 0.5.
        label_smoothing: Label smoothing coefficient in [0, 1). Default: 0.0 (no smoothing).
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,  # shape: (batch,)
        policy_rejected_logps: torch.Tensor,  # shape: (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SimPO loss.

        Args:
            policy_chosen_logps: Average log-probabilities for chosen sequences. Shape: (batch,).
            policy_rejected_logps: Average log-probabilities for rejected sequences. Shape: (batch,).

        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards), where loss is a scalar
            and rewards have shape (batch,).
        """  # noqa: E501
        chosen_rewards = self.beta * policy_chosen_logps
        rejected_rewards = self.beta * policy_rejected_logps

        reward_margin = chosen_rewards - rejected_rewards - self.gamma

        # Primary NLL loss: -log sigmoid(reward_margin)
        nll_loss = -F.logsigmoid(reward_margin)

        if self.label_smoothing > 0.0:
            # Smoothed loss blends in the "wrong" direction
            smooth_loss = -F.logsigmoid(-reward_margin)
            loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss

        return loss.mean(), chosen_rewards, rejected_rewards


class AdaSimPOLoss(SimPOLoss):
    """AdaDPO-style adaptive SimPO loss (arxiv 2605.28440).

    Extends SimPO with per-pair adaptive balancing coefficients that correct
    the asymmetric gradient pathology in pairwise preference losses.

    The key insight: DPO/SimPO gradients are inherently unbalanced — pairs where
    the model already strongly agrees with the preference produce small gradients,
    while pairs where the model disagrees produce disproportionately large gradients
    with high variance. AdaDPO fixes this with a per-pair coefficient:

        w_i = clamp(exp(alpha * (logp_chosen_i - logp_rejected_i)), max=C)

    where pairs with large log-prob gaps are down-weighted, and pairs with small
    or negative gaps are up-weighted. This acts as an implicit curriculum: the
    model focuses on pairs it's getting wrong rather than pairs it's already right about.

    Args:
        beta: SimPO scaling factor. Default: 2.0.
        gamma: SimPO target margin. Default: 0.5.
        alpha: Adaptive coefficient temperature. Default: 1.0.
               Higher values produce stronger adaptive weighting.
        C: Max clipping value for adaptive weights. Default: 5.0.
           Prevents excessively large weights on extremely hard pairs.
        label_smoothing: SimPO label smoothing. Default: 0.0.

    Reference:
        AdaDPO: Self-Adaptive DPO with Balancing Coefficients
        https://arxiv.org/abs/2605.28440
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        alpha: float = 1.0,
        C: float = 5.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(beta=beta, gamma=gamma, label_smoothing=label_smoothing)
        self.alpha = alpha
        self.C = C

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute standard SimPO rewards
        chosen_rewards = self.beta * policy_chosen_logps
        rejected_rewards = self.beta * policy_rejected_logps
        reward_margin = chosen_rewards - rejected_rewards - self.gamma

        # AdaDPO balancing coefficient per pair
        # Down-weight pairs where model is already correct (large logp gap)
        logp_gap = policy_chosen_logps - policy_rejected_logps  # (batch,)
        # w_i = clamp(exp(alpha * gap), max=C) with stop_gradient
        with torch.no_grad():
            w = torch.clamp(torch.exp(self.alpha * logp_gap), max=self.C)  # (batch,)

        # Weighted loss: w_i * -log sigmoid(margin_i)
        nll_loss = -F.logsigmoid(reward_margin)  # (batch,)

        if self.label_smoothing > 0.0:
            smooth_loss = -F.logsigmoid(-reward_margin)
            loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss

        # Apply adaptive weights; normalize by sum of weights for stable batch size
        weighted_loss = (w * loss).sum() / w.sum()

        return weighted_loss, chosen_rewards, rejected_rewards
