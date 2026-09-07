"""Tests for src/eval/run_gsm8k.py (T22)."""

from __future__ import annotations

from src.eval.run_gsm8k import run_benchmark, run_eval


def test_run_benchmark_oracle_is_perfect() -> None:
    scores = run_benchmark(mode="oracle", n_samples=3, seed=0)
    assert scores == [1.0, 1.0, 1.0]


def test_run_benchmark_mock_is_zero() -> None:
    scores = run_benchmark(mode="mock", n_samples=2, seed=0)
    assert scores == [0.0, 0.0]


def test_run_eval_smoke_payload() -> None:
    payload = run_eval(profile="smoke", mode="oracle")
    assert payload["suite"] == "gsm8k"
    assert payload["accuracy"] == 1.0
    assert payload["score"] == 1.0
