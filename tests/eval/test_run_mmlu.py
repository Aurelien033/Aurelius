"""Tests for src/eval/run_mmlu.py (T22)."""

from __future__ import annotations

from src.eval.run_mmlu import run_benchmark, run_eval


def test_run_benchmark_oracle_is_perfect() -> None:
    scores = run_benchmark(mode="oracle", profile="smoke", n_samples=2, seed=0)
    assert scores == [1.0, 1.0]


def test_run_benchmark_mock_below_oracle() -> None:
    scores = run_benchmark(mode="mock", profile="smoke", n_samples=1, seed=0)
    assert scores[0] < 1.0


def test_run_eval_smoke_payload() -> None:
    payload = run_eval(profile="smoke", mode="oracle")
    assert payload["suite"] == "mmlu_57"
    assert payload["accuracy"] == 1.0
