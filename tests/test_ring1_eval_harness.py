"""Tests for Ring 1 evaluation harness (Tranche 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.eval.ring1_agent import Ring1Agent
from src.eval.ring1_eval_harness import (
    bootstrap_ci,
    compute_condition_metrics,
    run_eval_harness,
    split_traces,
    write_eval_artifacts,
)
from src.eval.ring1_trace_logger import Ring1TraceLogger, validate_trace


def _sample_config() -> dict:
    return {
        "seeds": {"base_seed": 1},
        "model": {"checkpoint_path": "checkpoints/aurelius-1.3b"},
        "trace_length_distribution": {"4": 1.0},
        "layer_range": [8, 16],
        "surprise_threshold": 0.5,
        "mcts": {"simulation_budget": 4},
    }


def test_bootstrap_ci_bounds() -> None:
    mean, low, high = bootstrap_ci([1.0, 0.0, 1.0, 1.0], n_resamples=200, seed=0)
    assert low <= mean <= high


def test_split_traces_holdout() -> None:
    traces = [{"trace_id": str(i), "seed": i} for i in range(10)]
    train, holdout = split_traces(traces, holdout_fraction=0.2, seed=0)
    assert len(holdout) == 2
    assert len(train) == 8


def test_run_eval_harness_on_generated_traces() -> None:
    config = _sample_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Ring1TraceLogger(Path(tmpdir), config)
        agent = Ring1Agent(config)
        jsonl = Path(tmpdir) / "traces.jsonl"
        for seed in range(10):
            trace = agent.generate_trace(seed=seed, logger=logger)
            validate_trace(trace)
            logger.write_trace(trace)

        report = run_eval_harness(
            amc_traces_path=jsonl,
            holdout_fraction=0.2,
            seed=0,
            dreambank_bank_fill=4,
            param_hash_unchanged=True,
        )
        assert len(report.ablation_table) == 3
        assert report.dreambank_lift is not None
        assert report.dreambank_lift.param_hash_unchanged

        metrics_dir = Path(tmpdir) / "metrics"
        write_eval_artifacts(report, metrics_dir)
        assert (metrics_dir / "aggregated_metrics.json").exists()


def test_compute_condition_metrics() -> None:
    traces = [
        {
            "trace_id": "t1",
            "final_outcome": "success",
            "total_steps": 6,
            "steps": [
                {
                    "memory_reads": [{"influenced_action": True}],
                    "memory_writes": [
                        {
                            "decision": "promote",
                            "verification_passed": True,
                            "proposed_content": "fact",
                        }
                    ],
                }
            ],
        }
    ]
    metrics = compute_condition_metrics(traces, "full_amc")
    assert metrics.task_success_rate == 1.0
    assert metrics.n_traces == 1
