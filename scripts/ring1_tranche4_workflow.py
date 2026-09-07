#!/usr/bin/env python3
"""Ring 1 Tranche 4 workflow — scale traces + gate verification + repro pack."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ring1_dreambank_runner import (
    extract_seeds_from_traces,
    run_shuffled_control_cycles,
    run_sleep_on_trace_file,
    write_dreambank_artifacts,
)
from src.eval.ring1_eval_harness import run_eval_harness_dual, write_eval_artifacts
from src.eval.ring1_gate_verifier import verify_all_gates, write_gate_report
from src.eval.ring1_repro_pack import write_repro_pack
from src.eval.ring1_trace_logger import load_traces
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def collect_condition_traces(
    *,
    agent_config_path: Path,
    num_traces: int,
    output_dir: Path,
    base_seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "traces.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    sidecar_dir = output_dir / "sidecars"
    if sidecar_dir.exists():
        for sidecar in sidecar_dir.glob("*.ndjson"):
            sidecar.unlink()

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "ring1_trace_collector.py"),
        "--config",
        str(agent_config_path),
        "--num_traces",
        str(num_traces),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(base_seed),
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return output_dir / "traces.jsonl"


def run_workflow(config_path: Path) -> dict:
    config = load_config(config_path)
    command = " ".join(sys.argv)
    min_traces = config["collection"]["min_traces_per_condition"]
    amc_dir = Path(config["collection"]["amc_output_dir"])
    baseline_dir = Path(config["collection"]["baseline_output_dir"])
    output_root = Path(config["reproducibility"]["output_dir"])

    amc_path = collect_condition_traces(
        agent_config_path=_REPO_ROOT / config["agents"]["amc"]["config"],
        num_traces=min_traces,
        output_dir=amc_dir,
        base_seed=config["seeds"]["base_seed"],
    )
    baseline_path = collect_condition_traces(
        agent_config_path=_REPO_ROOT / config["agents"]["baseline"]["config"],
        num_traces=min_traces,
        output_dir=baseline_dir,
        base_seed=config["seeds"]["base_seed"] + config["seeds"]["baseline_seed_offset"],
    )

    dreambank_dir = output_root / "dreambank_run"
    dreambank_result = run_sleep_on_trace_file(
        amc_path,
        bank_config=config.get("dreambank", {}).get("bank", {}),
        dreambank_config=config.get("dreambank", {}),
        num_cycles=config.get("dreambank", {}).get("cycles", 1),
        max_seeds=config.get("dreambank", {}).get("max_seeds", 64),
    )
    write_dreambank_artifacts(dreambank_result, dreambank_dir)

    amc_traces = load_traces(amc_path)
    seeds = extract_seeds_from_traces(
        amc_traces,
        max_seeds=config.get("dreambank", {}).get("max_seeds", 64),
    )
    control_bank = HLMPreferenceBank(
        HLMPreferenceBankConfig(
            bank_size=config["dreambank"]["bank"]["bank_size"],
            bank_dim=config["dreambank"]["bank"]["bank_dim"],
        )
    )
    shuffled = run_shuffled_control_cycles(
        seeds,
        bank=control_bank,
        num_cycles=1,
        shuffle_seed=config["dreambank"].get("shuffled_control_seed", 424242),
    )
    shuffled_rate = shuffled.total_writes / max(len(seeds), 1)

    eval_report, holdout_amc, holdout_baseline = run_eval_harness_dual(
        amc_traces_path=amc_path,
        baseline_traces_path=baseline_path,
        holdout_fraction=config["eval"]["holdout_fraction"],
        seed=config["seeds"]["base_seed"],
        dreambank_bank_fill=dreambank_result.bank_fill,
        param_hash_unchanged=bool(
            dreambank_result.zero_grad_proof
            and dreambank_result.zero_grad_proof.param_hash_unchanged
        ),
        post_success_boost=config["eval"].get("post_success_boost", 0.02),
    )
    metrics_dir = output_root / "metrics"
    write_eval_artifacts(eval_report, metrics_dir)

    tarball, pack_dir = write_repro_pack(
        config=config,
        output_dir=output_root,
        traces_path=amc_path,
        dreambank_dir=dreambank_dir,
        metrics_dir=metrics_dir,
        repo_root=_REPO_ROOT,
    )

    gate_report = verify_all_gates(
        config=config,
        command=command,
        amc_traces_path=amc_path,
        baseline_traces_path=baseline_path,
        holdout_amc=holdout_amc,
        holdout_baseline=holdout_baseline,
        eval_report=eval_report.to_dict(),
        dreambank_result=dreambank_result.to_dict(),
        pack_dir=pack_dir,
        shuffled_control_rate=min(1.0, shuffled_rate),
    )
    write_gate_report(gate_report, output_root / "gate_verification_report.json")

    summary = {
        "status": "ok",
        "amc_traces": str(amc_path),
        "baseline_traces": str(baseline_path),
        "amc_count": len(load_traces(amc_path)),
        "baseline_count": len(load_traces(baseline_path)),
        "preliminary_all_gates_passed": gate_report.preliminary_all_passed,
        "r1_ga_passed": gate_report.r1_ga.passed,
        "r1_gb_passed": gate_report.r1_gb.passed,
        "r1_gc_passed": gate_report.r1_gc.passed,
        "repro_pack": str(tarball),
        "gate_report": str(output_root / "gate_verification_report.json"),
    }
    (output_root / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ring 1 Tranche 4 gate workflow")
    parser.add_argument("--config", type=Path, default=Path("configs/ring1_tranche4.yaml"))
    args = parser.parse_args(argv)
    summary = run_workflow(args.config)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("preliminary_all_gates_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
