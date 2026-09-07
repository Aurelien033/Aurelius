"""AMC memory-aware training losses.

Three losses add memory-awareness to standard cross-entropy SFT:

1. surprise_prediction_loss — BCE for predicting important turns.
2. memory_consistency_loss — align current hidden state with retrieved memories.
3. promotion_reward_loss — REINFORCE-style signal for promotion gate decisions.

Combined with SFT:
  total = alpha*sft + beta*surprise + gamma*consistency + delta*promotion
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def surprise_prediction_loss(
    predicted_surprise: torch.Tensor,
    importance_labels: torch.Tensor,
    *,
    weight_by_importance: bool = True,
) -> torch.Tensor:
    """BCE loss for surprise prediction."""
    if predicted_surprise.dim() == 3:
        predicted_surprise = predicted_surprise.mean(dim=0)
        if importance_labels.dim() == 3:
            importance_labels = importance_labels.mean(dim=0)

    predicted = predicted_surprise.clamp(1e-7, 1 - 1e-7)
    target = importance_labels.float()

    if predicted.shape != target.shape:
        min_len = min(predicted.shape[-1], target.shape[-1])
        predicted = predicted[..., :min_len]
        target = target[..., :min_len]

    bce = F.binary_cross_entropy(predicted, target, reduction="none")

    if weight_by_importance:
        weight = torch.where(
            target > 0.5,
            torch.tensor(2.0, device=target.device, dtype=target.dtype),
            torch.tensor(1.0, device=target.device, dtype=target.dtype),
        )
        bce = bce * weight

    return bce.mean()


def memory_consistency_loss(
    current_hidden: torch.Tensor,
    retrieved_embeddings: torch.Tensor | None,
    *,
    embedding_proj: nn.Module | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cosine alignment loss between pooled hidden state and retrieved memories."""
    if retrieved_embeddings is None or retrieved_embeddings.numel() == 0:
        return torch.tensor(0.0, device=current_hidden.device, dtype=current_hidden.dtype)

    current_pooled = current_hidden.mean(dim=1)
    if embedding_proj is not None:
        current_pooled = embedding_proj(current_pooled)

    if current_pooled.shape[-1] != retrieved_embeddings.shape[-1]:
        raise ValueError(
            f"dimension mismatch: current={current_pooled.shape[-1]}, "
            f"retrieved={retrieved_embeddings.shape[-1]}"
        )

    current_norm = F.normalize(current_pooled, dim=-1)
    retrieved_norm = F.normalize(retrieved_embeddings, dim=-1)
    sim = torch.einsum("bd,bkd->bk", current_norm, retrieved_norm)
    attn = F.softmax(sim / temperature, dim=-1)
    weighted_sim = (attn * sim).sum(dim=-1)
    return (1 - weighted_sim).mean()


def promotion_reward_loss(
    store_soft: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """REINFORCE-style loss: reinforce gate when reward > 0."""
    if store_soft.dim() != 1 or rewards.dim() != 1:
        raise ValueError(f"expected 1D inputs, got {store_soft.shape}, {rewards.shape}")
    if store_soft.shape[0] != rewards.shape[0]:
        raise ValueError(f"shape mismatch: {store_soft.shape} vs {rewards.shape}")

    p = store_soft.clamp(1e-7, 1 - 1e-7)
    log_p = torch.log(p)
    masked_rewards = rewards * (rewards > 0).float()
    return -(log_p * masked_rewards).mean()


def total_amc_loss(
    sft_loss: torch.Tensor,
    surprise_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    promotion_loss: torch.Tensor,
    *,
    alpha: float = 0.70,
    beta: float = 0.15,
    gamma: float = 0.10,
    delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weighted sum of all AMC losses."""
    total = (
        alpha * sft_loss + beta * surprise_loss + gamma * consistency_loss + delta * promotion_loss
    )
    metrics = {
        "total_loss": float(total.detach().item()),
        "sft_loss": float(sft_loss.detach().item()),
        "surprise_loss": float(surprise_loss.detach().item()),
        "consistency_loss": float(consistency_loss.detach().item()),
        "promotion_loss": float(promotion_loss.detach().item()),
        "loss_weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
    }
    return total, metrics


__all__ = [
    "memory_consistency_loss",
    "promotion_reward_loss",
    "surprise_prediction_loss",
    "total_amc_loss",
]
