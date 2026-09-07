"""
src/training/adaptive_sft.py — Adaptive data selection for SFT.

Extends the existing src/data/importance_sampler.py with SFT-specific
adaptive sampling: after each SFT epoch, measure per-example loss and
resample with weight = loss^temperature. Temperature decays from 10
(near-uniform early) to 1.0 (focused on hard examples late).

Reference: "Adaptive SFT Data Selection by Model Uncertainty" (Aurelius, 2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSFTConfig:
    """Configuration for adaptive SFT data selection.

    Attributes:
        temperature_start: Initial sampling temperature (higher = more uniform).
        temperature_end: Final sampling temperature (lower = more focused).
        total_steps: Total training steps for temperature decay schedule.
        ema_alpha: Smoothing factor for online weight updates.
        min_weight: Floor weight to prevent example starvation.
        compute_loss_every_n_steps: How often to recompute per-example losses.
        use_difficulty_decay: If True, decay temperature linearly.
    """

    temperature_start: float = 10.0
    temperature_end: float = 1.0
    total_steps: int = 1000
    ema_alpha: float = 0.3
    min_weight: float = 0.01
    compute_loss_every_n_steps: int = 100
    use_difficulty_decay: bool = True


class AdaptiveSFTDataSelector:
    """Adaptively selects SFT training examples by model uncertainty.

    After each epoch (or every N steps), measures per-example loss and
    reweights the sampling distribution. Easy examples get lower weight;
    hard examples get higher weight. Temperature controls how sharp the
    selection is.

    Usage:
        selector = AdaptiveSFTDataSelector(dataset, config)
        for epoch in range(n_epochs):
            sampler = selector.get_sampler(epoch)
            loader = DataLoader(dataset, sampler=sampler, batch_size=4)
            for batch in loader:
                loss = train_step(batch)
                selector.log_loss(batch_indices, per_sample_losses)
    """

    def __init__(
        self,
        n_examples: int,
        config: AdaptiveSFTConfig | None = None,
    ) -> None:
        self.config = config or AdaptiveSFTConfig()
        self.n_examples = n_examples
        self._perplexities = torch.ones(n_examples)
        self._weights = torch.ones(n_examples)
        self._steps = 0
        self._current_epoch = 0

    def get_temperature(self) -> float:
        """Return the current sampling temperature.

        Linearly decays from temperature_start to temperature_end
        over total_steps.
        """
        if not self.config.use_difficulty_decay:
            return self.config.temperature_end
        progress = min(self._steps / max(self.config.total_steps, 1), 1.0)
        return (
            self.config.temperature_start
            + (self.config.temperature_end - self.config.temperature_start) * progress
        )

    def set_weights_from_perplexities(
        self,
        perplexities: torch.Tensor,
    ) -> None:
        """Set sampling weights from per-example perplexity scores.

        weight_i = max(ppl_i ^ (1/T), min_weight)
        Then normalized to sum to n_examples.

        Args:
            perplexities: (N,) tensor of per-example perplexity scores.
        """
        ppl = perplexities.float().clamp(min=1.0)
        self._perplexities = ppl.clone()

        temp = self.get_temperature()
        exponent = 1.0 / max(temp, 1e-8)
        raw = ppl**exponent
        raw = raw.clamp(min=self.config.min_weight)

        self._weights = raw / raw.sum() * self.n_examples

    def log_loss(
        self,
        indices: list[int] | torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        """Log per-example losses and update weights via EMA.

        Args:
            indices: Example indices in the dataset.
            losses: Per-example loss values (same length as indices).
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        perplexities = losses.exp().clamp(min=1.0, max=1000.0)

        for idx, ppl in zip(indices, perplexities):
            old_ppl = self._perplexities[idx].item()
            new_ppl = self.config.ema_alpha * ppl.item() + (1 - self.config.ema_alpha) * old_ppl
            self._perplexities[idx] = new_ppl

        self.set_weights_from_perplexities(self._perplexities)
        self._steps += 1

    def get_sampler(
        self,
        num_samples: int | None = None,
    ) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for the current weights.

        Args:
            num_samples: Number of samples per epoch (default: n_examples).

        Returns:
            WeightedRandomSampler with replacement.
        """
        n = num_samples or self.n_examples
        return WeightedRandomSampler(
            weights=self._weights.tolist(),
            num_samples=n,
            replacement=True,
        )

    def get_epoch_sampler(
        self,
        epoch: int,
    ) -> WeightedRandomSampler:
        """Return a sampler for a given epoch.

        Epoch 0 always uses uniform sampling (no selection).
        Subsequent epochs use adaptive sampling.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            WeightedRandomSampler.
        """
        if epoch == 0:
            # Uniform sampling for first epoch
            return WeightedRandomSampler(
                weights=torch.ones(self.n_examples).tolist(),
                num_samples=self.n_examples,
                replacement=True,
            )
        return self.get_sampler()

    @property
    def weights(self) -> torch.Tensor:
        return self._weights.clone()

    @property
    def perplexities(self) -> torch.Tensor:
        return self._perplexities.clone()

    def summary(self) -> dict[str, float]:
        """Return summary statistics of current weight distribution."""
        w = self._weights
        return {
            "temperature": self.get_temperature(),
            "mean_weight": w.mean().item(),
            "std_weight": w.std().item(),
            "min_weight": w.min().item(),
            "max_weight": w.max().item(),
            "n_examples": self.n_examples,
            "weighted_entropy": (-(w / w.sum()) * (w / w.sum()).log()).sum().item(),
        }


# Registry entry
ADAPTIVE_SFT_REGISTRY: dict[str, type] = {
    "adaptive_selector": AdaptiveSFTDataSelector,
}
