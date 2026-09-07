"""RSVA — Recursive Self-Verifying Architecture (DESIGNED; not yet run).

From the user's original inventions (aurelius-novel-inventions-2026-06-10),
activated in V3 Part 3 (A.1). Highest-leverage inference-time mechanism.

Core loop (post-hoc verify -> accept/reject is replaced by iterative refinement):

    Input -> Model -> Output_0 + Verification_0
                       |
          Output_0 + V_0 -> Model -> Output_1 + Verification_1
                       |
                ... until ||Output_{t+1} - Output_t|| < eps

Mathematical claim (T422/T381): if the model is a contraction mapping w.r.t.
the verification-augmented input (Lipschitz L < 1), Banach fixed-point theorem
guarantees convergence to a unique y* with error <= L^k ||y0 - y*|| after k iters.

This module is TORCH-FREE orchestration: the model, verifier, and convergence
check are injected callables so the loop is unit-testable without a checkpoint.
Mirrors the established pattern in verifier_recursion.py. It emits TruthSurface-
style ledger rows (value, baseline, n, cost) for each run.

FALSIFIER (from Part 3 A.1): if RSVA-k accuracy does not exceed single-pass
accuracy by more than best-of-k independent sampling, it is merely sampling with
extra steps — kill and fall back to best-of-N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

Generate = Callable[[str], str]  # prompt -> completion
VerifyFn = Callable[
    [str, str], dict
]  # (prompt, completion) -> {score:float, verdict:bool, detail:str}
Converged = Callable[[str, str, dict, dict], bool]  # (prev, cur, vprev, vcur) -> converged?
LabelFn = Callable[[str, str], bool]  # optional held-out ground truth


@dataclass
class RSVAResult:
    iterations: int
    converged: bool
    final_output: str
    final_verify: dict
    scores: list[float] = field(default_factory=list)
    cost_generations: int = 0  # == iterations (one gen per iter)
    correct: bool | None = None  # needs label_fn

    def ledger_row(self) -> dict:
        """TruthSurface-style row. value = final verify score; baseline set by caller."""
        return {
            "experiment": "rsva",
            "iterations": self.iterations,
            "converged": self.converged,
            "value": round(self.final_verify.get("score", 0.0), 4),
            "n": 1,  # single task per RSVA run; aggregate across tasks
            "cost_generations": self.cost_generations,
            "correct": None if self.correct is None else bool(self.correct),
        }


class RSVALoop:
    """Iterative self-verification until convergence.

    The verify_fn must return a dict with at least `score` (float, higher=better)
    and `verdict` (bool). The default convergence check compares the verification
    dicts of consecutive iterations; override via `converged` for token-level diffs.
    """

    def __init__(
        self,
        generate: Generate,
        verify_fn: VerifyFn,
        *,
        max_iters: int = 10,
        converged: Converged | None = None,
        label_fn: LabelFn | None = None,
        eps: float = 0.01,
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        self.generate = generate
        self.verify_fn = verify_fn
        self.max_iters = max_iters
        self.eps = eps
        self.label_fn = label_fn
        self._converged = converged or self._default_converged

    def _default_converged(self, prev: str, cur: str, vprev: dict, vcur: dict) -> bool:
        # Convergence on the verification signal (absolute score delta) AND on the
        # output text (exact match). Both required: a stable-but-wrong plateau is
        # not a fixed point worth declaring.
        score_delta = abs(vcur.get("score", 0.0) - vprev.get("score", 0.0))
        return score_delta < self.eps and prev == cur

    def run(self, prompt: str) -> RSVAResult:
        prev = self.generate(prompt)
        vprev = self.verify_fn(prompt, prev)
        scores = [float(vprev.get("score", 0.0))]
        for it in range(1, self.max_iters):
            cur = self.generate(prompt)
            vcur = self.verify_fn(prompt, cur)
            scores.append(float(vcur.get("score", 0.0)))
            if self._converged(prev, cur, vprev, vcur):
                correct = None
                if self.label_fn is not None:
                    correct = self.label_fn(prompt, cur)
                return RSVAResult(
                    iterations=it + 1,
                    converged=True,
                    final_output=cur,
                    final_verify=vcur,
                    scores=scores,
                    cost_generations=it + 1,
                    correct=correct,
                )
            prev, vprev = cur, vcur
        # Did not converge within budget -> return last; caller applies falsifier.
        correct = None
        if self.label_fn is not None:
            correct = self.label_fn(prompt, prev)
        return RSVAResult(
            iterations=self.max_iters,
            converged=False,
            final_output=prev,
            final_verify=vprev,
            scores=scores,
            cost_generations=self.max_iters,
            correct=correct,
        )


def rsva_verdict(
    rsva_rows: list[dict], single_pass: float, best_of_k: float, min_gain: float = 0.03
) -> dict:
    """Falsifier-gated verdict (Part 3 A.1).

    RSVA must beat BOTH single-pass AND best-of-k independent sampling, else it is
    just sampling with extra steps. Returns the promotion decision + delta.
    """
    if not rsva_rows:
        return {"verdict": "insufficient", "reason": "no RSVA rows"}
    mean_score = sum(r["value"] for r in rsva_rows) / len(rsva_rows)
    d_single = mean_score - single_pass
    d_bok = mean_score - best_of_k
    if d_single >= min_gain and d_bok >= min_gain:
        verdict = "CONVERGES (beats single-pass AND best-of-k)"
    elif d_single <= -min_gain or d_bok <= -min_gain:
        verdict = "REGRESSES (killed — fall back to best-of-N)"
    else:
        verdict = "FLAT (within noise of baselines)"
    return {
        "verdict": verdict,
        "mean_rsva": round(mean_score, 4),
        "single_pass": single_pass,
        "best_of_k": best_of_k,
        "d_single": round(d_single, 4),
        "d_bok": round(d_bok, 4),
        "n_tasks": len(rsva_rows),
    }


# ---- real-wiring factory (guarded imports; not needed for tests) ----
def wire_real_rsva(
    model_generate, verifier, *, max_iters: int = 10, label_fn: LabelFn | None = None
) -> RSVALoop:
    """Assemble an RSVA loop from a model sampler + a verifier.

    model_generate(prompt) -> str          : your model sampler (single completion)
    verifier(prompt, completion) -> dict   : MultiDomainVerifier-shaped reward wrapper
    """
    return RSVALoop(model_generate, verifier, max_iters=max_iters, label_fn=label_fn)
