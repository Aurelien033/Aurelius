"""CascadeBank ablation harness — proves the router distributes decisions
across all three classes on a populated bank, and collapses to the fail-safe
class on an empty bank.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from src.inference.cascade_routing import CascadeRouter, ComputePolicy
from src.memory.hlm_bank import HLMPreferenceBank
from src.model.amc_transformer import AMCTransformer


@dataclass
class CascadeAblationResult:
    """Distribution of routing decisions across a batch of prompts."""

    decision_distribution: dict[str, int] = field(
        default_factory=lambda: {"fast": 0, "balanced": 0, "thorough": 0}
    )
    total_decisions: int = 0
    alpha_mean: float = 0.0
    confidence_mean: float = 0.0
    bank_fill: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_distribution": dict(self.decision_distribution),
            "total_decisions": self.total_decisions,
            "alpha_mean": self.alpha_mean,
            "confidence_mean": self.confidence_mean,
            "bank_fill": self.bank_fill,
        }


def _empty_distribution() -> dict[str, int]:
    return {policy.value: 0 for policy in ComputePolicy}


def run_cascade_ablation(
    model: AMCTransformer,
    prompts: Sequence[torch.Tensor],
    bank: HLMPreferenceBank | None = None,
    router: CascadeRouter | None = None,
) -> CascadeAblationResult:
    """Run the same prompts through the model with/without a bank, route each."""
    router = router or CascadeRouter()
    dist = _empty_distribution()
    total_alpha = 0.0
    total_conf = 0.0
    count = 0

    with torch.no_grad():
        for prompt in prompts:
            out = model(prompt, preference_bank=bank)
            decision = router.decision(out.bank_alpha, out.bank_confidence)
            dist[decision.value] += 1

            if out.bank_alpha is not None:
                total_alpha += float(out.bank_alpha.mean().item())
            if out.bank_confidence is not None:
                total_conf += float(out.bank_confidence.mean().item())
            count += 1

    n = max(count, 1)
    fill = (
        int(bank.strengths.gt(0).sum().item())
        if bank is not None and hasattr(bank, "strengths")
        else 0
    )

    return CascadeAblationResult(
        decision_distribution=dist,
        total_decisions=count,
        alpha_mean=total_alpha / n,
        confidence_mean=total_conf / n,
        bank_fill=fill,
    )
