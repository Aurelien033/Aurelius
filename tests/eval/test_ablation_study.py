"""Tests for AMC ablation study runner (T28)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ablation import (
    CONFIGS,
    AblationResult,
    bootstrap_paired_pvalue,
    run_ablation_study,
    run_benchmark_scores,
    summarize_results,
)


def test_bootstrap_paired_pvalue_identical_lists_high_p() -> None:
    scores = [0.8, 0.82, 0.79, 0.81, 0.8]
    p_value = bootstrap_paired_pvalue(scores, scores, n_bootstrap=500, random_seed=0)
    assert p_value >= 0.4


def test_bootstrap_paired_pvalue_returns_probability_in_unit_interval() -> None:
    baseline = [0.5, 0.52, 0.48, 0.51, 0.49]
    improved = [value + 0.2 for value in baseline]
    p_value = bootstrap_paired_pvalue(improved, baseline, n_bootstrap=2000, random_seed=1)
    assert 0.0 <= p_value <= 1.0


def test_bootstrap_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_paired_pvalue([1.0, 2.0], [1.0])


def test_oracle_amc_memory_monotonic_by_tier() -> None:
    order = ["baseline", "tier1_only", "tier12", "full_amc"]
    means = [
        float(np.mean(run_benchmark_scores(config_name=name, benchmark="amc_memory", mode="oracle")))
        for name in order
    ]
    assert means == sorted(means)
    assert means[-1] > means[0]


def test_oracle_gsm8k_within_five_percent_of_baseline() -> None:
    baseline = float(
        np.mean(run_benchmark_scores(config_name="baseline", benchmark="gsm8k", mode="oracle"))
    )
    full_amc = float(
        np.mean(run_benchmark_scores(config_name="full_amc", benchmark="gsm8k", mode="oracle"))
    )
    assert abs(full_amc - baseline) <= 0.05


def test_run_ablation_study_writes_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "ablation_scores.jsonl"
    results, written = run_ablation_study(
        "",
        benchmarks=("amc_memory", "gsm8k"),
        configs=("baseline", "full_amc"),
        output_dir=tmp_path,
        output_path=output_path,
        mode="oracle",
        n_bootstrap=200,
    )
    assert written == output_path
    assert len(results) == 4
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    payload = json.loads(lines[0])
    assert payload["config"] in CONFIGS
    assert "raw_scores" in payload


def test_full_amc_beats_baseline_on_amc_memory_oracle(tmp_path: Path) -> None:
    results, _ = run_ablation_study(
        "",
        benchmarks=("amc_memory",),
        configs=("baseline", "tier1_only", "tier12", "full_amc"),
        output_dir=tmp_path,
        output_path=tmp_path / "scores.jsonl",
        mode="oracle",
        n_bootstrap=500,
    )
    by_config = {row.config: row for row in results}
    assert by_config["full_amc"].score > by_config["tier12"].score
    assert by_config["tier12"].score > by_config["tier1_only"].score
    assert by_config["tier1_only"].score > by_config["baseline"].score
    assert by_config["full_amc"].p_value_vs_baseline is not None
    assert 0.0 <= by_config["full_amc"].p_value_vs_baseline <= 1.0


def test_summarize_results_includes_benchmark_header() -> None:
    rows = [
        AblationResult("baseline", "amc_memory", 0.5, 0.01, 5),
        AblationResult("full_amc", "amc_memory", 0.9, 0.01, 5, p_value_vs_baseline=0.01),
    ]
    text = summarize_results(rows)
    assert "amc_memory" in text
    assert "full_amc" in text
