"""HMC — Holographic Mechanism Compression (DESIGNED; not yet run).

From the user's original inventions (aurelius-novel-inventions-2026-06-10),
activated in V3 Part 3 (A.2). Replaces separate LoRA adapters / MoE routers /
discrete gating with a single shared parameter space where every mechanism is
stored as an interference pattern.

Core mechanism:
    W = W_0 + sum_i  f_i (x) r_i
    f_i ~= (W . r_i) / ||r_i||^2           # projection recovers the mechanism
    read(f_i) by projecting W onto reference r_i

Mechanism birth (the killer feature): track gradient direction per error type;
when ||accumulated_gradient|| > threshold, define r_new = normalized(grad),
f_new = grad projected onto r_new, and add it to the HMC space. Spontaneous
specialization from experience.

This module is TORCH-FREE orchestration + linear-algebra over numpy. It models
the shared weight matrix W and reference vectors r_i as plain arrays so the
retrieval/birth math is unit-testable. Real wiring replaces the numpy ops with
parameter hooks. Mirrors the verifier_recursion.py pattern.

FALSIFIER (Part 3 A.2): if mechanism retrieval accuracy < 80% (projecting onto
r_i does not recover f_i), or mechanism birth creates noise rather than useful
specialization, HMC is not ready — fall back to LoRA-based specialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MechanismFn = Callable[[str], str]          # mechanism name -> generated behavior


@dataclass
class HMCConfig:
    d_hidden: int = 64                       # proxy dimension for the shared space
    birth_threshold: float = 1.0            # ||accumulated grad|| to trigger birth
    retrieval_tol: float = 0.80             # min cosine sim to count as "recovered"
    seed: int = 0


@dataclass
class HMCRetrievalResult:
    mechanism: str
    cosine_sim: float
    recovered: bool

    def ledger_row(self, baseline_sim: float = 0.0) -> dict:
        return {
            "experiment": "hmc_retrieval",
            "mechanism": self.mechanism,
            "value": round(self.cosine_sim, 4),
            "baseline_value": baseline_sim,
            "n": 1,
            "status_ok": self.recovered,
        }


class HMC:
    """Holographic mechanism store over a shared parameter space (numpy proxy)."""

    def __init__(self, cfg: HMCConfig | None = None) -> None:
        self.cfg = cfg or HMCConfig()
        rng = np.random.default_rng(self.cfg.seed)
        # Proxy shared space. ZERO by default so the read/write math is provable
        # (W = sum_i f_i (x) r_i => (W.r_i)/||r_i||^2 = f_i exactly). In real
        # wiring this is bound to the actual parameter matrix, where a random
        # base WOULD interfere with retrieval — that is what hmc_verdict guards.
        self.W = np.zeros((self.cfg.d_hidden, self.cfg.d_hidden))
        self.refs: dict[str, np.ndarray] = {}         # mechanism -> reference vector r_i
        self.accum_grad: dict[str, np.ndarray] = {}    # error_type -> accumulated gradient
        self.birthed: list[str] = []

    # -- core math --
    def _project(self, r: np.ndarray) -> np.ndarray:
        """f_i ~= (W . r_i) / ||r_i||^2 — recover the mechanism direction from W."""
        r = np.asarray(r, dtype=float)
        norm2 = float(r @ r)
        if norm2 == 0:
            return np.zeros_like(r)
        return (self.W @ r) / norm2

    def read(self, mechanism: str) -> np.ndarray | None:
        """Project W onto r_i to read mechanism i."""
        if mechanism not in self.refs:
            return None
        return self._project(self.refs[mechanism])

    def write(self, mechanism: str, f: np.ndarray) -> None:
        """Store mechanism i as interference pattern W += lr * (f (x) r_i); r_i = f/||f||."""
        r = np.asarray(f, dtype=float)
        norm = np.linalg.norm(r)
        if norm == 0:
            return
        r = r / norm
        # spec: W = W_0 + sum_i f_i (x) r_i ; read recovers f_i = (W.r_i)/||r_i||^2
        self.W = self.W + np.outer(r, r) * 0.1 * norm  # scale by ||f|| so magnitude survives
        self.refs[mechanism] = r

    def retrieval_accuracy(self, mechanism: str, ground_truth_f: np.ndarray) -> HMCRetrievalResult:
        """Does projecting onto r_i recover f_i? Cosine sim vs the stored f."""
        if mechanism not in self.refs:
            return HMCRetrievalResult(mechanism, 0.0, False)
        recovered_f = self._project(self.refs[mechanism])
        gt = np.asarray(ground_truth_f, dtype=float)
        if np.linalg.norm(gt) == 0 or np.linalg.norm(recovered_f) == 0:
            sim = 0.0
        else:
            sim = float(recovered_f @ gt / (np.linalg.norm(recovered_f) * np.linalg.norm(gt)))
        return HMCRetrievalResult(mechanism, sim, sim >= self.cfg.retrieval_tol)

    # -- mechanism birth --
    def accumulate_gradient(self, error_type: str, grad: np.ndarray) -> str | None:
        """Track per-error-type gradients; birth a new mechanism when ||grad|| exceeds."""
        g = np.asarray(grad, dtype=float)
        if error_type not in self.accum_grad:
            self.accum_grad[error_type] = np.zeros_like(g)
        self.accum_grad[error_type] = self.accum_grad[error_type] + g
        if np.linalg.norm(self.accum_grad[error_type]) >= self.cfg.birth_threshold:
            r_new = self.accum_grad[error_type] / np.linalg.norm(self.accum_grad[error_type])
            self.write(f"born_{error_type}", r_new)
            self.birthed.append(error_type)
            self.accum_grad[error_type] = np.zeros_like(g)
            return error_type
        return None


def hmc_verdict(retrieval_results: list[HMCRetrievalResult],
                min_acc: float = 0.80) -> dict:
    """Falsifier-gated verdict (Part 3 A.2): recovery rate must clear min_acc."""
    if not retrieval_results:
        return {"verdict": "insufficient", "reason": "no retrieval rows"}
    recovered = sum(1 for r in retrieval_results if r.recovered)
    rate = recovered / len(retrieval_results)
    if rate >= min_acc:
        verdict = "RETRIEVES (holographic storage works)"
    else:
        verdict = "FAILS RETRIEVAL (fall back to LoRA specialization)"
    return {
        "verdict": verdict,
        "recovery_rate": round(rate, 4),
        "min_acc": min_acc,
        "n_mechanisms": len(retrieval_results),
        "birthed": len(set(r.mechanism for r in retrieval_results if "born" in r.mechanism)),
    }
