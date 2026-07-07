"""VerifierRecursionLoop — the compounding capability loop, built on real code.

The highest-EV compounding mechanism from the v3 roadmap (§9 step 5):

    base -> generate k candidates per task
         -> verifier ranks/filters to the correct ones
         -> RLVR train_step on the selected-correct set
         -> (optionally) retrain the verifier head
         -> repeat

Rationale (STaR 2203.14465; verifier-recursion math C=M*V): capacity M is the
floor, the VERIFIER V is what compounds. Each cycle the model absorbs its own
verifier-selected-correct outputs, and a better verifier -> better selection ->
more/cleaner training signal -> better base.

This wraps the CONFIRMED-REAL repo pieces via injected callables so the
orchestration is torch-free and unit-testable:
  generate(prompt, k) -> list[str]                     # real: model.generate
  verify_fn(prompt, completion) -> float in [0,1]      # real: mdv.reward_fn_wrap(domain)
  train_fn(selected) -> dict{"mean_reward": float,...}  # real: CurriculumRLVRTrainer.train_step
An optional `label_fn(task, completion) -> bool` (held-out ground truth) enables
verifier-PRECISION tracking — the V that must rise for the loop to compound.

CRITICAL GUARD (from the arc's receipts): naive winners-only self-distillation
REGRESSED greedy by 11pp. So the verdict here is strict: the loop only "works" if
the verifier-selected set is genuinely correct (precision high) AND the metric
rises across cycles without collapsing selection. `run` emits TruthSurface-style
ledger rows (cost + n + gate) for every cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Generate = Callable[[str, int], list[str]]
VerifyFn = Callable[[str, str], float]          # == MultiDomainVerifier.reward_fn_wrap output
TrainFn = Callable[[list[dict]], dict]          # selected -> {"mean_reward":..., ...}
LabelFn = Callable[[str, str], bool]            # optional held-out ground truth


@dataclass
class Task:
    task_id: str
    prompt: str
    answer: str = ""


@dataclass
class CycleResult:
    cycle: int
    n_tasks: int
    n_candidates: int
    n_selected: int
    selected_frac: float          # fraction of candidates passing the verifier threshold
    solve_rate: float             # fraction of TASKS with >=1 selected candidate (proxy for M)
    verifier_precision: float | None   # of selected, fraction actually correct (V) — needs label_fn
    train_mean_reward: float | None
    cost_generations: int         # candidates generated = the run cost

    def ledger_row(self) -> dict:
        """TruthSurface-style row: metric + cost + n + gate inputs."""
        return {
            "experiment": "verifier_recursion",
            "cycle": self.cycle,
            "n": self.n_tasks,
            "solve_rate": round(self.solve_rate, 4),
            "verifier_precision": None if self.verifier_precision is None
            else round(self.verifier_precision, 4),
            "selected_frac": round(self.selected_frac, 4),
            "train_mean_reward": self.train_mean_reward,
            "cost_generations": self.cost_generations,
        }


class VerifierRecursionLoop:
    """Orchestrates base -> verify -> select-correct -> RLVR, tracking compounding."""

    def __init__(
        self,
        generate: Generate,
        verify_fn: VerifyFn,
        train_fn: TrainFn,
        *,
        k: int = 8,
        select_threshold: float = 0.999,   # "correct" = verifier ~1.0 (execution pass)
        label_fn: LabelFn | None = None,
        max_selected_per_task: int = 1,     # keep best-scored; avoid over-weighting easy tasks
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.generate = generate
        self.verify_fn = verify_fn
        self.train_fn = train_fn
        self.k = k
        self.select_threshold = select_threshold
        self.label_fn = label_fn
        self.max_selected_per_task = max_selected_per_task

    def run_cycle(self, tasks: list[Task], cycle: int = 0) -> CycleResult:
        selected: list[dict] = []
        n_candidates = 0
        n_tasks_solved = 0
        correct_selected = 0

        for t in tasks:
            cands = self.generate(t.prompt, self.k)
            n_candidates += len(cands)
            scored = [(c, self.verify_fn(t.prompt, c)) for c in cands]
            passing = [(c, s) for c, s in scored if s >= self.select_threshold]
            passing.sort(key=lambda x: x[1], reverse=True)
            keep = passing[: self.max_selected_per_task]
            if keep:
                n_tasks_solved += 1
            for c, s in keep:
                selected.append({"task_id": t.task_id, "prompt": t.prompt,
                                 "response": c, "answer": t.answer, "verify_score": s})
                if self.label_fn is not None and self.label_fn(t, c):
                    correct_selected += 1

        n_sel = len(selected)
        precision = None
        if self.label_fn is not None and n_sel > 0:
            precision = correct_selected / n_sel

        train_reward = None
        if selected:
            res = self.train_fn(selected)
            train_reward = float(res.get("mean_reward")) if res and "mean_reward" in res else None

        return CycleResult(
            cycle=cycle, n_tasks=len(tasks), n_candidates=n_candidates,
            n_selected=n_sel,
            selected_frac=(n_sel / n_candidates) if n_candidates else 0.0,
            solve_rate=(n_tasks_solved / len(tasks)) if tasks else 0.0,
            verifier_precision=precision, train_mean_reward=train_reward,
            cost_generations=n_candidates,
        )

    def run(self, tasks: list[Task], n_cycles: int = 3) -> list[CycleResult]:
        return [self.run_cycle(tasks, cycle=i) for i in range(n_cycles)]


def recursion_verdict(cycles: list[CycleResult], min_gain: float = 0.03) -> dict:
    """Judge whether the loop COMPOUNDED (roadmap gate), guarding the winners-only
    −11pp failure mode: solve_rate must rise across cycles AND (if labels present)
    verifier precision must not collapse. Returns a verdict dict.
    """
    if len(cycles) < 2:
        return {"verdict": "insufficient", "reason": "need >=2 cycles"}
    first, last = cycles[0], cycles[-1]
    d_solve = last.solve_rate - first.solve_rate
    prec_ok = True
    prec_note = "no labels"
    if first.verifier_precision is not None and last.verifier_precision is not None:
        prec_ok = last.verifier_precision >= first.verifier_precision - 0.05
        prec_note = f"{first.verifier_precision:.3f}->{last.verifier_precision:.3f}"
    if d_solve >= min_gain and prec_ok:
        verdict = "COMPOUNDS"
    elif d_solve <= -min_gain:
        verdict = "REGRESSES (winners-only trap — do NOT ship)"
    else:
        verdict = "FLAT (selection gap not foldable into greedy by this loop)"
    return {
        "verdict": verdict,
        "d_solve_rate": round(d_solve, 4),
        "precision": prec_note,
        "precision_ok": prec_ok,
        "cycles": len(cycles),
    }


# ---- real-wiring factory (guarded imports; not needed for tests) ----
def wire_real_loop(model_generate, domain: str, curriculum_trainer, *, test_runner=None,
                   k: int = 8, **verify_kwargs) -> VerifierRecursionLoop:
    """Assemble a loop from the confirmed-real repo pieces.

    model_generate(prompt, k) -> list[str]         : your model sampler
    domain                    : 'code'|'math'|...   (MultiDomainVerifier)
    curriculum_trainer        : a CurriculumRLVRTrainer instance
    test_runner               : code executor for the code domain
    """
    from src.training.multi_domain_verifier import MultiDomainVerifier
    mdv = MultiDomainVerifier()
    verify = mdv.reward_fn_wrap(domain, **({"test_runner": test_runner} if test_runner else {}),
                                **verify_kwargs)

    def train_fn(selected: list[dict]) -> dict:
        # one grouped RLVR step over the selected-correct set (real train_step
        # takes tokenised prompt_ids; the caller adapts batching to their loop).
        rewards = []
        for ex in selected:
            r = curriculum_trainer.train_step(
                task_ids=[ex["task_id"]],
                prompt_ids=ex.get("prompt_ids"),
                prompt_text=ex["prompt"],
                answer=ex.get("answer", ""),
            )
            rewards.append(float(r.get("mean_reward", 0.0)))
        return {"mean_reward": sum(rewards) / len(rewards) if rewards else 0.0}

    return VerifierRecursionLoop(model_generate, verify, train_fn, k=k)
