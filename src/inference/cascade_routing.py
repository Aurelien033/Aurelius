"""CascadeBank — pure-function router over DreamBank gate signals.

Zero trainable parameters.  CascadeRouter is a stateless function that maps
(bank_alpha, bank_confidence) from AMCModelOutput to a ComputePolicy decision.

Invariants:
- decision(None, *) and decision(*, None) always return the configured fallback.
- No gradients, no side effects, no heap allocations in hot path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch


class ComputePolicy(StrEnum):
    """Coarse inference-compute budget decision."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass(frozen=True)
class CascadeRouterConfig:
    """Routing thresholds.  Defaults chosen so balanced is the fail-safe class."""

    # If max(alpha_t) over the prompt >= alpha_thorough, escalate.
    alpha_thorough: float = 0.70
    # If mean(confidence_t) over the prompt < confidence_fast, downgrade.
    confidence_fast: float = 0.40
    # Require both signals present to upgrade to THOROUGH; otherwise stay at current level.
    require_confidence_for_thorough: bool = True
    # Minimum mean confidence to stay BALANCED when alpha is ambiguous.
    balanced_confidence_floor: float = 0.30
    # If DreamBank is disabled/empty, always return this policy (fail-safe).
    fallback: ComputePolicy = ComputePolicy.BALANCED


class CascadeRouter:
    """Pure-function routing on DreamBank gate signals.  Zero parameters."""

    def __init__(self, config: CascadeRouterConfig | None = None) -> None:
        self.config = config if config is not None else CascadeRouterConfig()

    def decision(
        self,
        bank_alpha: torch.Tensor | None,
        bank_confidence: torch.Tensor | None,
    ) -> ComputePolicy:
        """Pure-function routing.  O(1) per-token reduction.  Never raises.

        Args:
            bank_alpha: shape (B, T, 1) — per-token alignment gate from DreamBank.
            bank_confidence: shape (B, T, 1) — per-token confidence from DreamBank.

        Returns:
            ComputePolicy.  Falls back to `fallback` when either input is None.
        """
        if bank_alpha is None or bank_confidence is None:
            return self.config.fallback

        alpha_max = bank_alpha.max().item()
        conf_mean = bank_confidence.mean().item()

        # Step 1: fast gate — low confidence is strong signal to downgrade
        if conf_mean < self.config.confidence_fast:
            return ComputePolicy.FAST

        # Step 2: thorough gate — high alpha AND strong confidence
        if alpha_max >= self.config.alpha_thorough:
            if self.config.require_confidence_for_thorough:
                if conf_mean >= self.config.balanced_confidence_floor:
                    return ComputePolicy.THOROUGH
            else:
                return ComputePolicy.THOROUGH

        # Default: balanced
        return ComputePolicy.BALANCED

    def telemetry(
        self, histories: Sequence[ComputePolicy] | None = None
    ) -> dict[str, int]:
        """Class distribution over a history of decisions.  Pure function."""
        counts = {p.value: 0 for p in ComputePolicy}
        if histories:
            for h in histories:
                counts[h.value] = counts.get(h.value, 0) + 1
        counts["total"] = sum(counts.values())
        return counts


__all__ = [
    "CascadeRouter",
    "CascadeRouterConfig",
    "ComputePolicy",
]
