"""Tests for VerifierRecursionLoop — torch-free orchestration with stubs.

Covers: selection by verifier threshold, solve-rate/precision tracking, the
compounding verdict, and the CRITICAL guard that catches the winners-only
regression trap (solve_rate falling across cycles).
"""
from __future__ import annotations

from src.training.verifier_recursion import (
    CycleResult, Task, VerifierRecursionLoop, recursion_verdict,
)


TASKS = [Task(f"t{i}", f"solve {i}", answer=str(i)) for i in range(10)]


def _make_loop(gen, verify, train, **kw):
    return VerifierRecursionLoop(gen, verify, train, **kw)


def test_selection_by_threshold_and_solve_rate():
    # generate k candidates; exactly one per task is "correct" (verify==1.0)
    def gen(prompt, k):
        return ["CORRECT"] + ["wrong"] * (k - 1)
    def verify(prompt, comp):
        return 1.0 if comp == "CORRECT" else 0.2
    trained = {}
    def train(selected):
        trained["n"] = len(selected)
        return {"mean_reward": 1.0}
    loop = _make_loop(gen, verify, train, k=8)
    res = loop.run_cycle(TASKS, cycle=0)
    assert res.n_tasks == 10
    assert res.n_candidates == 80
    assert res.n_selected == 10                 # 1 correct kept per task
    assert res.solve_rate == 1.0
    assert res.train_mean_reward == 1.0
    assert trained["n"] == 10
    row = res.ledger_row()
    assert row["experiment"] == "verifier_recursion" and row["cost_generations"] == 80


def test_precision_tracking_with_labels():
    # verifier APPROVES "plausible" (score 1.0) but only "CORRECT" is truly right
    def gen(prompt, k):
        return ["plausible"] * k
    def verify(prompt, comp):
        return 1.0                              # over-approves everything
    def label(task, comp):
        return comp == "CORRECT"                # nothing generated is actually correct
    loop = _make_loop(gen, verify, lambda s: {"mean_reward": 0.5}, k=4, label_fn=label)
    res = loop.run_cycle(TASKS)
    assert res.n_selected == 10                 # 1 per task kept
    assert res.verifier_precision == 0.0        # verifier is high-recall, zero-precision
    # this is exactly the low-precision over-approval the learned probe targets


def test_no_selection_when_nothing_passes():
    loop = _make_loop(lambda p, k: ["x"] * k, lambda p, c: 0.1,
                      lambda s: {"mean_reward": 0.0}, k=4)
    res = loop.run_cycle(TASKS)
    assert res.n_selected == 0 and res.solve_rate == 0.0
    assert res.train_mean_reward is None        # no train call when nothing selected


def test_run_multiple_cycles():
    loop = _make_loop(lambda p, k: ["CORRECT"] + ["w"] * (k - 1),
                      lambda p, c: 1.0 if c == "CORRECT" else 0.0,
                      lambda s: {"mean_reward": 1.0}, k=4)
    cycles = loop.run(TASKS, n_cycles=3)
    assert len(cycles) == 3 and all(c.solve_rate == 1.0 for c in cycles)


# --- verdict logic (the roadmap gate + winners-only guard) ---
def _cyc(cycle, solve, prec=None):
    return CycleResult(cycle=cycle, n_tasks=10, n_candidates=40, n_selected=10,
                       selected_frac=0.25, solve_rate=solve, verifier_precision=prec,
                       train_mean_reward=1.0, cost_generations=40)


def test_verdict_compounds():
    v = recursion_verdict([_cyc(0, 0.60), _cyc(1, 0.70), _cyc(2, 0.78)])
    assert v["verdict"] == "COMPOUNDS" and v["d_solve_rate"] > 0


def test_verdict_catches_winners_only_regression():
    # solve_rate falls across cycles -> the -11pp winners-only trap
    v = recursion_verdict([_cyc(0, 0.78), _cyc(1, 0.70), _cyc(2, 0.60)])
    assert "REGRESSES" in v["verdict"]


def test_verdict_flat():
    v = recursion_verdict([_cyc(0, 0.70), _cyc(1, 0.71), _cyc(2, 0.70)])
    assert "FLAT" in v["verdict"]


def test_verdict_precision_collapse_blocks_compound():
    # solve_rate rises BUT precision collapses -> not a clean compound
    v = recursion_verdict([_cyc(0, 0.60, prec=0.95), _cyc(2, 0.75, prec=0.40)])
    assert v["precision_ok"] is False and v["verdict"] != "COMPOUNDS"


def test_verdict_insufficient():
    assert recursion_verdict([_cyc(0, 0.6)])["verdict"] == "insufficient"
