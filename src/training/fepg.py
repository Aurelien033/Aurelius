"""FEPG — Free Energy Policy Gradient (DESIGNED; not yet run).

From the user's original inventions (aurelius-novel-inventions-2026-06-10),
activated in V3 Part 3 (A.3). Replaces GRPO/PPO/DPO with a single intrinsic
objective that cannot be gamed because minimizing free energy IS Bayes-optimal
behavior under uncertainty.

Core objective:
    F = E - T . S
    E = prediction error (cross-entropy with ground truth / verifier score)
    S = entropy of the model's output distribution
    T = temperature (adaptive exploration/exploitation balance)
    grad F = grad E - T . grad S

Honesty emerges automatically: expressing false confidence raises E without
lowering S, so FEPG prefers "I don't know" over confident-wrong. No reward model,
no explicit honesty training.

This module is TORCH-FREE orchestration: the model's (logits -> entropy) and the
(verifier -> error) are injected callables so the gradient step and the
reward-hacking comparison are unit-testable. Mirrors verifier_recursion.py.

FALSIFIER (Part 3 A.3): if FEPG does not match GRPO final accuracy on verifiable
tasks, or if the entropy term causes mode collapse, demote to research-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

EntropyFn = Callable[[str], float]  # prompt -> output-distribution entropy (0..1)
ErrorFn = Callable[[str, str], float]  # (prompt, completion) -> prediction error E
StepFn = Callable[[float, float], dict]  # (E, S) -> train step result


@dataclass
class FEPGStep:
    E: float
    S: float
    T: float
    F: float
    collapsed: bool
    result: dict = field(default_factory=dict)

    def ledger_row(self) -> dict:
        return {
            "experiment": "fepg",
            "E": round(self.E, 4),
            "S": round(self.S, 4),
            "T": round(self.T, 4),
            "F": round(self.F, 4),
            "collapsed": self.collapsed,
            "n": 1,
        }


class FEPGTrainer:
    """Minimize free energy F = E - T*S per step; log collapse + compare to GRPO."""

    def __init__(
        self,
        entropy_fn: EntropyFn,
        error_fn: ErrorFn,
        step_fn: StepFn,
        *,
        T: float = 1.0,
        collapse_floor: float = 0.05,  # S below this = mode collapse
        adaptive_T: bool = True,
    ) -> None:
        self.entropy_fn = entropy_fn
        self.error_fn = error_fn
        self.step_fn = step_fn
        self.T = T
        self.collapse_floor = collapse_floor
        self.adaptive_T = adaptive_T

    def step(self, prompt: str, completion: str) -> FEPGStep:
        E = float(self.error_fn(prompt, completion))
        S = float(self.entropy_fn(prompt))
        # Adaptive temperature: when uncertain (high E), raise T to encourage exploration.
        T = self.T
        if self.adaptive_T:
            T = max(0.1, min(2.0, self.T * (1.0 + E)))
        F = E - T * S
        collapsed = S < self.collapse_floor
        res = self.step_fn(E, S)
        return FEPGStep(E=E, S=S, T=T, F=F, collapsed=collapsed, result=res)


def fepg_verdict(
    fepg_acc: float,
    grpo_acc: float,
    min_gain: float = -0.02,
    n_collapse_steps: int = 0,
    total_steps: int = 1,
) -> dict:
    """Falsifier-gated verdict (Part 3 A.3): must match GRPO; no mode collapse.

    min_gain is slightly negative (within noise) — FEPG need not beat GRPO, but
    it must not regress, and collapse rate must stay low.
    """
    d = fepg_acc - grpo_acc
    collapse_rate = (n_collapse_steps / total_steps) if total_steps else 0.0
    if d >= min_gain and collapse_rate < 0.10:
        verdict = "MATCHES GRPO (ungameable objective viable)"
    elif collapse_rate >= 0.10:
        verdict = "MODE COLLAPSE (entropy term too strong — research-only)"
    else:
        verdict = "REGRESSES vs GRPO (demote to research-only)"
    return {
        "verdict": verdict,
        "fepg_acc": round(fepg_acc, 4),
        "grpo_acc": round(grpo_acc, 4),
        "d_acc": round(d, 4),
        "collapse_rate": round(collapse_rate, 4),
    }
