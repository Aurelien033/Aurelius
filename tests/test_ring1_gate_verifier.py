"""Tests for Ring 1 gate verifier (Tranche 4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from src.eval.ring1_agent import Ring1Agent
from src.eval.ring1_dummy_agent import Ring1DummyAgent
from src.eval.ring1_gate_verifier import audit_memory_write_evidence, verify_all_gates, verify_r1_ga
from src.eval.ring1_trace_logger import Ring1TraceLogger, validate_trace


def _config() -> dict:
    return {
        "seeds": {"base_seed": 1},
        "model": {"checkpoint_path": "checkpoints/aurelius-1.3b"},
        "trace_length_distribution": {"4": 1.0},
        "layer_range": [8, 16],
        "surprise_threshold": 0.5,
        "mcts": {"simulation_budget": 4},
    }


def _write_traces(agent, count: int, seed_offset: int, tmp: Path) -> Path:
    config = _config()
    logger = Ring1TraceLogger(tmp, config)
    for index in range(count):
        trace = agent.generate_trace(seed=seed_offset + index, logger=logger)
        validate_trace(trace)
        logger.write_trace(trace)
    return tmp / "traces.jsonl"


def test_audit_memory_write_evidence_passes_for_integrated_agent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_traces(Ring1Agent(_config()), 5, 0, Path(tmpdir))
        # traces = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
        audit = audit_memory_write_evidence([json.loads(line) for line in Path(path).read_text().splitlines()])
        assert audit["sample_size"] >= 1
        assert audit["passed_traces"] >= 1


def test_verify_r1_ga_fails_with_insufficient_corpus() -> None:
    result = verify_r1_ga(
        amc_traces=[{"trace_id": "a", "steps": [], "metadata": {"tags": []}}],
        baseline_traces=[{"trace_id": "b", "steps": [], "metadata": {"tags": []}}],
        holdout_amc=[],
        holdout_baseline=[],
        min_traces_per_condition=200,
        min_success_delta_pp=5.0,
        config={"seeds": {}},
        command="test",
        failure_modes={},
    )
    assert not result.passed
    assert any(step.step == "trace_corpus" and not step.passed for step in result.steps)


def test_tranche4_config_exists() -> None:
    config = yaml.safe_load(Path("configs/ring1_tranche4.yaml").read_text(encoding="utf-8"))
    assert config["collection"]["min_traces_per_condition"] >= 200
    assert config["gates"]["min_traces_per_condition"] >= 200


def test_verify_all_gates_produces_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        amc_dir = tmp / "amc"
        base_dir = tmp / "base"
        amc_dir.mkdir()
        base_dir.mkdir()
        amc_path = _write_traces(Ring1Agent(_config()), 12, 0, amc_dir)
        base_path = _write_traces(Ring1DummyAgent(_config()), 12, 1000, base_dir)

        pack_dir = tmp / "pack"
        pack_dir.mkdir()
        (pack_dir / "README.md").write_text("# pack", encoding="utf-8")
        (pack_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (pack_dir / "config").mkdir()
        (pack_dir / "config" / "experiment_config.yaml").write_text("tranche: 4", encoding="utf-8")
        (pack_dir / "seeds").mkdir()
        (pack_dir / "seeds" / "master_seeds.json").write_text("{}", encoding="utf-8")
        (pack_dir / "verification").mkdir()
        (pack_dir / "verification" / "verify.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        dreambank = pack_dir / "dreambank_run"
        dreambank.mkdir()
        (dreambank / "pre_cycle_model_hash.txt").write_text("abc\n", encoding="utf-8")
        (dreambank / "post_cycle_model_hash.txt").write_text("abc\n", encoding="utf-8")
        traces = pack_dir / "traces" / "test_split"
        traces.mkdir(parents=True)
        (traces / "traces.jsonl").write_text("{}", encoding="utf-8")
        metrics = pack_dir / "metrics"
        metrics.mkdir()
        (metrics / "aggregated_metrics.json").write_text("{}", encoding="utf-8")

        config = yaml.safe_load(Path("configs/ring1_tranche4.yaml").read_text(encoding="utf-8"))
        config["gates"]["min_traces_per_condition"] = 10
        config["gates"]["min_ga_success_delta_pp"] = 0.0
        config["gates"]["min_gb_lift_pp"] = 0.0

        amc_traces = [json.loads(line) for line in amc_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        holdout = amc_traces[:3]
        holdout_base = [json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines() if line.strip()][:3]

        report = verify_all_gates(
            config=config,
            command="pytest",
            amc_traces_path=amc_path,
            baseline_traces_path=base_path,
            holdout_amc=holdout,
            holdout_baseline=holdout_base,
            eval_report={
                "failure_modes": {"failure_mode:successful_retrieval": 12},
                "dreambank_lift": {
                    "pre_success_rate": 0.8,
                    "post_success_rate": 0.85,
                    "absolute_lift_pp": 5.0,
                    "param_hash_unchanged": True,
                },
            },
            dreambank_result={
                "zero_grad_proof": {
                    "pre_cycle_param_hash": "abc",
                    "post_cycle_param_hash": "abc",
                    "param_hash_unchanged": True,
                },
                "preferences": [{"metadata_hash": "h1", "provenance": "dream_test"}],
            },
            pack_dir=pack_dir,
            shuffled_control_rate=0.1,
        )
        assert report.r1_ga.gate == "R1-GA"
        assert report.r1_gb.gate == "R1-GB"
        assert report.r1_gc.gate == "R1-GC"
