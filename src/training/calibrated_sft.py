"""
src/training/calibrated_sft.py — Calibration-Aware SFT loss.

Adds Expected Calibration Error (ECE) regularization to the standard SFT
loss. Uses the existing src/eval/calibration.py infrastructure for ECE
computation but provides a differentiable approximation suitable for
gradient-based training.

Reference: "Calibration-Aware SFT for Aurelius" (Aurelius, 2026)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CalibratedSFTConfig:
    """Configuration for calibration-aware SFT.

    Attributes:
        lambda_ece: Weight of ECE regularization loss.
        n_bins: Number of bins for ECE computation.
        temperature: Temperature for soft bin assignments (lower = harder).
        ece_frequency: Fraction of batches to compute ECE on.
        sample_ratio: Fraction of tokens to sample for ECE computation.
    """

    lambda_ece: float = 0.1
    n_bins: int = 10
    temperature: float = 0.1
    ece_frequency: float = 1.0
    sample_ratio: float = 0.2


class DifferentiableECE:
    """Soft/differentiable Expected Calibration Error for use as loss.

    Standard ECE uses hard bin assignments (non-differentiable). This
    implementation uses soft Gaussian bin assignments, making ECE
    differentiable and usable as a regularization term during SFT.
    """

    def __init__(self, config: CalibratedSFTConfig | None = None) -> None:
        self.config = config or CalibratedSFTConfig()

    def __call__(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute differentiable ECE loss.

        Args:
            logits: (B, T, V) raw model logits.
            labels: (B, T) target token IDs (-100 for padding).

        Returns:
            Differentiable ECE scalar loss.
        """
        probs = F.softmax(logits.float(), dim=-1)
        confidences, predictions = probs.max(dim=-1)  # (B, T), (B, T)
        accuracies = (predictions == labels).float()  # (B, T)

        # Flatten and ignore padding
        valid = labels != -100
        confidences = confidences[valid]
        accuracies = accuracies[valid]

        if confidences.numel() < self.config.n_bins:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Sample tokens to limit overhead
        if self.config.sample_ratio < 1.0:
            n_samples = max(
                int(confidences.numel() * self.config.sample_ratio), self.config.n_bins * 2
            )
            if n_samples < confidences.numel():
                indices = torch.randperm(confidences.numel(), device=confidences.device)[:n_samples]
                confidences = confidences[indices]
                accuracies = accuracies[indices]

        return self._soft_ece(confidences, accuracies)

    def _soft_ece(
        self,
        confidences: Tensor,
        accuracies: Tensor,
    ) -> Tensor:
        """Soft ECE with Gaussian bin assignments.

        Args:
            confidences: (N,) confidence values in [0, 1].
            accuracies: (N,) per-sample accuracy (0 or 1).

        Returns:
            Differentiable ECE loss (scalar).
        """
        n_bins = self.config.n_bins
        temp = self.config.temperature

        # Bin centers from 0 to 1
        bin_centers = torch.linspace(
            0.0,
            1.0,
            n_bins,
            device=confidences.device,
        )  # (n_bins,)

        # Soft bin assignments via RBF kernel
        diff = confidences.unsqueeze(-1) - bin_centers.unsqueeze(0)  # (N, n_bins)
        weights = F.softmax(-(diff**2) / max(temp, 1e-8), dim=-1)  # (N, n_bins)

        # Weighted accuracy and confidence per bin
        weighted_acc = (weights * accuracies.unsqueeze(-1)).sum(dim=0)  # (n_bins,)
        weighted_conf = (weights * confidences.unsqueeze(-1)).sum(dim=0)
        norm = weights.sum(dim=0).clamp(min=1.0)  # (n_bins,)

        bin_acc = weighted_acc / norm
        bin_conf = weighted_conf / norm

        # ECE = mean |acc - conf| over bins
        ece = (bin_acc - bin_conf).abs().mean()

        return ece


class CalibratedSFTLoss:
    """Combines standard cross-entropy with ECE regularization.

    Usage:
        loss_fn = CalibratedSFTLoss(config)
        total = loss_fn(logits, labels)  # = CE + lambda * ECE
    """

    def __init__(self, config: CalibratedSFTConfig | None = None) -> None:
        self.config = config or CalibratedSFTConfig()
        self.ece_fn = DifferentiableECE(config)

    def __call__(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute calibrated SFT loss.

        Args:
            logits: (B, T, V) model logits.
            labels: (B, T) target token IDs.

        Returns:
            (total_loss, components) where components contains
            individual loss terms for monitoring.
        """
        # Standard CE
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # ECE regularization
        ece_loss = self.ece_fn(logits, labels)

        total = ce_loss + self.config.lambda_ece * ece_loss

        return total, {
            "ce_loss": ce_loss.item() if hasattr(ce_loss, "item") else float(ce_loss),
            "ece_loss": ece_loss.item() if hasattr(ece_loss, "item") else float(ece_loss),
            "total_loss": total.item() if hasattr(total, "item") else float(total),
        }


# Registry entry
CALIBRATED_SFT_REGISTRY: dict[str, type] = {
    "calibrated_loss": CalibratedSFTLoss,
    "differentiable_ece": DifferentiableECE,
}
