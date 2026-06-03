"""
Solus-MoE -- Token Router

Routes each token position to the top-k experts and returns
gating weights, expert indices, and an auxiliary load-balance loss.

Design
------
  Hidden -> expert_logits : Linear(Hidden -> NumExperts)
  Gate softmax            : stable softmax (subtract max)
  Top-K selection         : torch.topk(k=moe_top_k)
  Auxiliary loss          : sigmoid-gate loss (Shazeer 2017 / ST-MoE)
    L_aux = alpha * num_experts * E_i[p_i * f_i]
    where p_i = mean routing probability, f_i = mean expert load fraction

References
----------
  Shazeer  et al. 2017  "Outrageously Large Neural Networks"
  Zoph, Le, Shazeer 2022 "ST-MoE"
"""  # noqa
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import SolusConfig


@dataclass
class RouterOutput:
    """Structured return from SolusMoERouter.forward()."""
    logits: Tensor              # (B, seq, num_experts) -- raw gate logits
    weights: Tensor             # (B, seq, num_experts) -- softmax(detached grad)
    expert_indices: Tensor      # (B, seq, top_k)        -- int64
    topk_weights: Tensor        # (B, seq, top_k)         -- re-normalized over top-k
    aux_loss: Optional[Tensor]  # ()  scalar or None


class SolusMoERouter(nn.Module):
    """Lightweight token router: linear -> softmax -> top-k selection."""

    def __init__(self, cfg: SolusConfig):
        super().__init__()
        self.num_experts = cfg.moe_num_experts
        self.top_k       = cfg.moe_top_k
        self.jitter_eps  = cfg.moe_router_jitter_eps
        self.aux_alpha   = cfg.moe_aux_loss_alpha

        self.gate = nn.Linear(cfg.hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def aux_loss(self, probs: Tensor) -> Tensor:
        """
        Shazeer-style sigmoid-gate auxiliary loss.
        
        Uses mean routing probability p_i per expert (value):
          p_i = mean probs[..., i]  over (B, seq)
        For top-k=2 each token activates exactly 2 experts:
          f_i = 2 / num_experts   (1 / num_experts per external activation range)
          mean per-expert = avg(probs[..., i])
          L_aux = alpha * E * sum_i(p_i * f_i)
                   = alpha * E * sum_i(p_i) * (2/E) = 2*alpha*mean(sum_i p_i) = 2*alpha
                   (constant in balanced case -- helpful numerically)

        Returns scalar Tensor.
        """
        if self.aux_alpha == 0.0:
            return torch.tensor(0.0, device=probs.device)

        # E[i] = mean fraction of tokens going to expert i
        mean_per_expert = probs.mean(dim=(0, 1))      # (E,)
        # For top-k=2; each token activates 2 experts -> each contributes its prob / 2
        # to expert load:  f = 2 * mean_probs
        load = 2.0 * mean_per_expert
        # Scalar auxiliary
        return self.aux_alpha * self.num_experts * (mean_per_expert * load).sum()

    # ---------------------------------------------------------------
    def forward(self, hidden_states: Tensor) -> RouterOutput:
        """
        Parameters
        ----------
        hidden_states : (B, seq, hidden_size)

        Returns
        -------
        RouterOutput with:
          logits          (B, seq, num_experts)
          weights         (B, seq, num_experts)
          expert_indices  (B, seq, top_k)   -- int64
          topk_weights    (B, seq, top_k)   -- float (re-normalized over selected experts)
          aux_loss        () scalar
        """
        # Router jitter: add small noise during training to encourage exploration
        logits = self.gate(hidden_states)  # (B  seq, num_experts)

        if self.training and self.jitter_eps > 0:
            logits = logits + torch.empty_like(logits).uniform_(-self.jitter_eps, self.jitter_eps)

        # Softmax over expert dimension
        probs = F.softmax(logits, dim=-1)    # (B  seq, E)

        # Re-compute logits from detached probs to preserve straight-through estimator
        # (allows gradient through softmax, routing decisions are non-differentiable)
        logits = torch.log(probs + 1e-9).detach() + probs - probs.detach() + probs.detach()
        # Actually: Straight-through estimator
        #   fwd : hard (top-k argmax)
        #   bwd : softmax prob
        probs_detached = probs.detach()      # (B, seq, E)  nn.grad
        hard_probs = torch.zeros_like(probs).scatter_(-1, probs_detached.topk(self.top_k, dim=-1).indices, 1.0)
        hard_probs = hard_probs - probs + probs_detached  # STE: identity fwd, softmax bwd

        # Select
        topk_values, topk_indices = probs_detached.topk(self.top_k, dim=-1)  # (B, seq, K), (B, seq, K)

        # Re-normalize over top-k
        topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # (B, seq, K)

        aux = self.aux_loss(probs)

        return RouterOutput(
            logits=logits,
            weights=hard_probs,          # (B, seq, E) — STE weights
            expert_indices=topk_indices,
            topk_weights=topk_weights,
            aux_loss=aux,
        )