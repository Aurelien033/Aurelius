"""Differentiable Tier-1 → Tier-2 promotion gate.

Uses Gumbel-softmax to learn a discrete store/skip decision while maintaining
gradient flow for end-to-end training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.memory.amc_tier2 import AMCTier2Hook
from src.model.amc_ssm_layer import AMCForwardOutput


class PromotionGate(nn.Module):
    """Per-layer promotion gate: should this step's memory be stored?

    Input: hidden state ``(B, d_model)`` + surprise score ``(B,)``.
    Output: ``(store_hard, store_soft)`` — hard discrete decision and soft
    probability for training (BCE / straight-through).
    """

    def __init__(self, d_model: int, temperature: float = 0.5) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        hidden = max(1, d_model // 4)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        surprise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(store_hard, store_soft)`` both shaped ``(B,)``."""
        if hidden.dim() != 2:
            raise ValueError(f"hidden must be (B, d_model), got {tuple(hidden.shape)}")
        if surprise.dim() != 1 or surprise.shape[0] != hidden.shape[0]:
            raise ValueError("surprise must be (B,) matching hidden batch size")

        x = torch.cat([hidden, surprise.unsqueeze(-1)], dim=-1)
        logits = self.gate_net(x)

        if self.training:
            soft = F.gumbel_softmax(logits, tau=self.temperature, hard=False)[:, 1]
            hard = F.gumbel_softmax(logits, tau=self.temperature, hard=True)[:, 1]
            return hard, soft

        store = (logits.argmax(dim=-1) == 1).to(dtype=hidden.dtype)
        return store, store

    def promote_loss(
        self,
        store_soft: torch.Tensor,
        target_store: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss: train gate to predict which turns should be stored."""
        return F.binary_cross_entropy(
            store_soft.clamp(1e-7, 1.0 - 1e-7),
            target_store,
        )

    def straight_through_store(
        self,
        store_hard: torch.Tensor,
        store_soft: torch.Tensor,
    ) -> torch.Tensor:
        """Gumbel straight-through: hard forward, soft backward."""
        return store_hard - store_soft.detach() + store_soft


def tier1_to_tier2_promotion(
    layer_output: AMCForwardOutput,
    promotion_gate: PromotionGate,
    tier2_hook: AMCTier2Hook,
    *,
    promote_threshold: float = 0.5,
) -> tuple[int, torch.Tensor, list]:
    """Run the promotion step: decide and execute Tier-2 storage."""
    hidden = layer_output.hidden[:, -1, :]
    surprise = layer_output.surprise_scores[:, -1]

    store_hard, store_soft = promotion_gate(hidden, surprise)

    promoted: list = []
    for batch_idx in range(store_hard.shape[0]):
        if store_hard[batch_idx].item() > promote_threshold:
            entry = tier2_hook.observe(
                role="working_memory",
                content=(
                    f"tier1_l{layer_output.tensor_state.layer_index}_"
                    f"s{layer_output.tensor_state.token_count}"
                ),
                surprise=float(surprise[batch_idx].item()),
                importance=float(store_soft[batch_idx].item()),
            )
            if entry is not None:
                promoted.append(entry)

    return len(promoted), store_soft.mean(), promoted


__all__ = ["PromotionGate", "tier1_to_tier2_promotion"]
