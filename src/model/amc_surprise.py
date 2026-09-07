"""Detachable, pretrainable surprise prediction head for AMC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SurpriseHeadConfig:
    d_model: int
    hidden_dim: int = 0
    dropout: float = 0.1
    n_layers: int = 2
    temperature: float = 1.0

    def resolve_hidden_dim(self) -> int:
        return self.hidden_dim or max(32, self.d_model // 4)


class SurpriseHead(nn.Module):
    """Standalone surprise prediction head.

    Usage in forward pass::

        surprise = head(x.detach())  # no grad to backbone
        store_decision, store_soft = gate(x, surprise)  # gate still gets grad
    """

    def __init__(self, config: SurpriseHeadConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.resolve_hidden_dim()

        layers: list[nn.Module] = []
        in_dim = config.d_model
        for _ in range(config.n_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self.temperature = config.temperature

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.net(hidden).squeeze(-1)
        return torch.sigmoid(out / self.temperature)

    @torch.no_grad()
    def predict(self, hidden: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        return self.forward(hidden) >= threshold


class SurprisePretrainer:
    """Offline pretrainer for the surprise head (focal BCE for class imbalance)."""

    def __init__(
        self,
        head: SurpriseHead,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        focal_gamma: float = 2.0,
    ) -> None:
        self.head = head
        self.optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.focal_gamma = focal_gamma

    def focal_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def train_step(
        self,
        hidden_states: torch.Tensor,
        importance_labels: torch.Tensor,
    ) -> dict[str, float]:
        self.head.train()
        self.optimizer.zero_grad()
        pred = self.head(hidden_states)
        loss = self.focal_bce(pred, importance_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            acc = ((pred > 0.5) == (importance_labels > 0.5)).float().mean()

        return {
            "loss": float(loss.detach()),
            "accuracy": float(acc),
            "positive_rate": float((importance_labels > 0.5).float().mean()),
            "mean_prediction": float(pred.mean()),
        }

    @torch.no_grad()
    def evaluate(
        self,
        hidden_states: torch.Tensor,
        importance_labels: torch.Tensor,
    ) -> dict[str, float]:
        self.head.eval()
        pred = self.head(hidden_states)
        loss = self.focal_bce(pred, importance_labels)
        acc = ((pred > 0.5) == (importance_labels > 0.5)).float().mean()
        return {"loss": float(loss.detach()), "accuracy": float(acc)}


def extract_surprise_heads(model: nn.Module) -> list[SurpriseHead]:
    """Return all :class:`SurpriseHead` instances from SSM layers for pretraining."""
    from src.model.amc_ssm_layer import AMCSSMLayer

    heads: list[SurpriseHead] = []
    for layer in getattr(model, "layers", []):
        if isinstance(layer, AMCSSMLayer):
            heads.append(layer.surprise_head)
    return heads


def sync_surprise_heads(heads: list[SurpriseHead], *, target: SurpriseHead) -> None:
    """Copy pretrained weights from *target* into every head in *heads*."""
    state = target.state_dict()
    for head in heads:
        head.load_state_dict(state)


__all__ = [
    "SurpriseHead",
    "SurpriseHeadConfig",
    "SurprisePretrainer",
    "extract_surprise_heads",
    "sync_surprise_heads",
]
