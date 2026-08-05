"""VGBS — Verifier-Guided Beam Search over reasoning/code steps.

A PRM-guided beam search: at each depth, expand each beam into candidate next
steps, score partial trajectories with a process verifier (PRM), keep the top-k.
Grounded in process-supervision literature (Let's Verify Step by Step 2305.20050;
Math-Shepherd 2312.08935): step-level verification beats outcome-only, and beam
search over a PRM reaches best-of-N parity at a fraction of the compute.

Cost (the reason to prefer it): a width-b depth-d search expands ~b·d·(1+prm_fac)
step-equivalents vs best-of-N's N·d. For b=4,d=30,N=64: 132 vs 1920 = ~14.5x
cheaper. `beam_cost` / `bon_cost` compute this so the gate is checkable.

Model-agnostic: inject `expand`, `score_step`, `is_complete`. The real wiring
uses a model's step-sampler for `expand` and MultiDomainVerifier (or a learned
PRM) for `score_step`; see `make_mdv_scorer`. Torch-free -> unit-testable.

GATE (roadmap): VGBS must clear greedy + ~12pp on MBPP-500, else keep best-of-N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# expand(prompt, prefix) -> list[str]      candidate next-step continuations
Expand = Callable[[str, str], list[str]]
# score_step(prompt, prefix) -> float      PRM score of a partial trajectory in [0,1]
ScoreStep = Callable[[str, str], float]
# is_complete(prefix) -> bool              trajectory terminal?
IsComplete = Callable[[str], bool]


@dataclass
class Beam:
    prefix: str
    score: float
    depth: int
    complete: bool = False


@dataclass
class VGBSResult:
    best: str                       # best complete (or deepest) trajectory text
    best_score: float
    depth_reached: int
    n_expansions: int               # measured step-equivalents (cost accounting)
    completed: bool
    all_finals: list[Beam] = field(default_factory=list)


def verifier_guided_beam_search(
    prompt: str,
    *,
    expand: Expand,
    score_step: ScoreStep,
    is_complete: IsComplete,
    beam_width: int = 4,
    max_depth: int = 30,
    init_prefix: str = "",
) -> VGBSResult:
    """Run PRM-guided beam search. Returns the best trajectory + cost stats.

    At each depth: expand every live beam, score each candidate with the PRM,
    keep the global top-`beam_width`. Completed beams are retained and compete
    on score. `n_expansions` counts PRM-scored candidates = the real cost.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    live: list[Beam] = [Beam(prefix=init_prefix, score=score_step(prompt, init_prefix), depth=0)]
    finals: list[Beam] = []
    n_expansions = 0

    for _depth in range(max_depth):
        if not live:
            break
        candidates: list[Beam] = []
        for b in live:
            if b.complete:
                candidates.append(b)
                continue
            for step in expand(prompt, b.prefix):
                new_prefix = b.prefix + step
                n_expansions += 1
                sc = score_step(prompt, new_prefix)
                candidates.append(Beam(
                    prefix=new_prefix, score=sc, depth=b.depth + 1,
                    complete=is_complete(new_prefix)))
        if not candidates:
            break
        # keep global top-k by score (stable: higher score, then deeper)
        candidates.sort(key=lambda x: (x.score, x.depth), reverse=True)
        kept = candidates[:beam_width]
        finals.extend(b for b in kept if b.complete)
        live = [b for b in kept if not b.complete]
        if not live:                      # all beams completed
            break

    pool = finals if finals else (live if live else [])
    if not pool:
        return VGBSResult(best=init_prefix, best_score=0.0, depth_reached=0,
                          n_expansions=n_expansions, completed=False)
    best = max(pool, key=lambda x: (x.score, x.depth))
    return VGBSResult(
        best=best.prefix, best_score=best.score, depth_reached=best.depth,
        n_expansions=n_expansions, completed=best.complete,
        all_finals=sorted(finals, key=lambda x: x.score, reverse=True))


def best_of_n(
    prompt: str,
    *,
    sample: Callable[[str], str],
    score_final: ScoreStep,
    n: int = 64,
) -> tuple[str, float, int]:
    """Best-of-N baseline: draw n full trajectories, pick the best by final score.
    Returns (best_text, best_score, n_scored). For head-to-head vs VGBS."""
    best_text, best_score = "", -1.0
    for _ in range(n):
        traj = sample(prompt)
        sc = score_final(prompt, traj)
        if sc > best_score:
            best_text, best_score = traj, sc
    return best_text, best_score, n


# ---- cost accounting (reproduces the roadmap's 14.5x claim; unit-tested) ----
def beam_cost(beam_width: int, depth: int, prm_fac: float = 0.1) -> float:
    """Step-equivalent cost of a VGBS run."""
    return beam_width * depth * (1.0 + prm_fac)


def bon_cost(n: int, depth: int) -> float:
    """Step-equivalent cost of best-of-N (n full rollouts of `depth` steps)."""
    return n * depth


def cost_ratio(beam_width: int = 4, depth: int = 30, n: int = 64, prm_fac: float = 0.1) -> float:
    """bon_cost / beam_cost — how many x cheaper VGBS is."""
    return bon_cost(n, depth) / beam_cost(beam_width, depth, prm_fac)


def verified_best_of_n(prompt: str, candidates: list[str],
                       verify: Callable[[str, str], float]) -> tuple[str, float, list[float]]:
    """The CODE-domain form of VGBS. An execution verifier is TERMINAL (0/1 on a
    complete program) — you cannot score a partial code prefix by running it — so
    step-level beam search degenerates to ranking whole candidates by the verifier
    and taking the best. With a perfect execution verifier this captures the full
    selection gap (== oracle) for free at inference. Returns (best, best_score, scores).

    This is deliberately the same operation as VerifierRecursionLoop's cycle-0
    selection: on code, the verifier IS ground truth, so best-of-N-verified is
    where the gap lives, and the recursion loop's job is folding it into greedy.
    """
    scores = [verify(prompt, c) for c in candidates]
    i = max(range(len(candidates)), key=lambda k: scores[k]) if candidates else 0
    return (candidates[i] if candidates else ""), (scores[i] if scores else 0.0), scores


# ---- real-wiring helpers (guarded import; not needed for tests) ----
def make_mdv_scorer(domain: str = "code", **verify_kwargs) -> ScoreStep:
    """Build a scorer from the repo's MultiDomainVerifier. Default domain=code
    (execution-verified is the honest home for these levers; the math probe was
    a measured null 2026-07-07).

    For code, pass ``test_runner=...`` (a (prompt, completion, task_id)->
    (passed, total, details) callable). For non-terminal domains used as a step
    PRM, a learned PRM head is the upgrade path.
    """
    from src.training.multi_domain_verifier import MultiDomainVerifier
    mdv = MultiDomainVerifier()

    def _score(prompt: str, completion: str) -> float:
        return mdv.verify(domain, prompt, completion, **verify_kwargs)

    return _score


def make_code_scorer(test_runner, **verify_kwargs) -> ScoreStep:
    """Convenience: an execution-verified CODE scorer (== oracle selector).
    ``test_runner(prompt, completion, task_id) -> (passed, total, details)``.
    """
    return make_mdv_scorer("code", test_runner=test_runner, **verify_kwargs)
