"""
src/training/loss_masking.py — Token-Level Difficulty Masking for SFT.

Masks easy tokens during SFT forward pass to focus gradient signal
on hard (informative) tokens. Can be used as a drop-in loss function
for AureliusTrainer.

Reference: "Token-Level Difficulty Masking for Efficient SFT" (Aurelius, 2026)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TokenMaskConfig:
    """Configuration for token-level difficulty masking.

    Attributes:
        mask_ratio: Fraction of lowest-loss tokens to mask (0.0 = no masking).
        mask_ratio_schedule: If provided, overrides mask_ratio with a schedule.
            Callable takes step (int) and returns mask_ratio (float).
        temperature: Sharpness of the loss threshold (lower = sharper cutoff).
        min_valid_tokens: Minimum valid tokens before masking is applied.
        ignore_index: Token index to ignore in loss (-100 is standard).
    """

    mask_ratio: float = 0.5
    mask_ratio_schedule: Callable[[int], float] | None = None
    temperature: float = 1.0
    min_valid_tokens: int = 4
    ignore_index: int = -100


class TokenDifficultyMask:
    """Token-level difficulty masking loss function.

    During the SFT forward pass, computes per-token cross-entropy loss,
    identifies the bottom K% of tokens by loss magnitude ("easy" tokens),
    and masks their loss contributions. Only hard tokens contribute to
    the gradient.

    Usage:
        mask_fn = TokenDifficultyMask(config)
        loss = mask_fn(logits, labels, global_step=step)
    """

    def __init__(self, config: TokenMaskConfig | None = None) -> None:
        self.config = config or TokenMaskConfig()

    def __call__(
        self,
        logits: Tensor,
        labels: Tensor,
        global_step: int | None = None,
    ) -> Tensor:
        """Compute masked cross-entropy loss.

        Args:
            logits: (B, T, V) raw model logits.
            labels: (B, T) target token IDs with ignore_index for padding.
            global_step: Optional training step for schedule-based masking.

        Returns:
            Scalar loss tensor.
        """
        # Determine effective mask ratio
        if global_step is not None and self.config.mask_ratio_schedule is not None:
            mask_ratio = self.config.mask_ratio_schedule(global_step)
        else:
            mask_ratio = self.config.mask_ratio

        # Compute per-token CE loss
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        )  # (B * T,)

        ce = ce.view(labels.size(0), -1)  # (B, T)

        if mask_ratio <= 0.0:
            # Standard CE over valid tokens
            return ce[labels != self.config.ignore_index].mean()

        # Apply masking per sequence
        valid_mask = labels != self.config.ignore_index
        masked_loss = ce.clone()

        for b in range(labels.size(0)):
            valid = valid_mask[b]
            n_valid = valid.sum().item()

            if n_valid < self.config.min_valid_tokens:
                continue  # Keep all tokens when too few valid

            valid_losses = ce[b][valid]
            threshold = torch.quantile(
                valid_losses,
                mask_ratio,
            )

            # Zero out easy tokens (below threshold)
            easy_mask = (ce[b] < threshold) & valid
            masked_loss[b][easy_mask] = 0.0

        # Mean over non-masked, non-ignored positions
        n_effective = (masked_loss > 0.0).sum()
        _n_valid_total = valid_mask.sum()

        if n_effective < 1:
            return ce[valid_mask].mean()  # Fallback to standard CE

        return masked_loss.sum() / n_effective

    def get_mask_ratio(self, global_step: int) -> float:
        """Return the effective mask ratio for a given step."""
        if self.config.mask_ratio_schedule is not None:
            return self.config.mask_ratio_schedule(global_step)
        return self.config.mask_ratio


def linear_decay_schedule(
    start_ratio: float = 0.7,
    end_ratio: float = 0.3,
    total_steps: int = 1000,
) -> Callable[[int], float]:
    """Create a linear decay schedule for mask_ratio.

    Starts at start_ratio (mask many easy tokens early) and decays
    to end_ratio (mask fewer as training progresses).

    Args:
        start_ratio: Initial mask ratio at step 0.
        end_ratio: Final mask ratio at step total_steps.
        total_steps: Total training steps.

    Returns:
        Callable[[int], float] that returns mask_ratio at a given step.
    """

    def _schedule(step: int) -> float:
        progress = min(step / max(total_steps, 1), 1.0)
        return start_ratio + (end_ratio - start_ratio) * progress

    return _schedule


# Registry entry for integration with AureliusTrainer
LOSS_MASKING_REGISTRY: dict[str, type] = {"token_difficulty": TokenDifficultyMask}
