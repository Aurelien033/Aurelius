#!/usr/bin/env python3
"""Ring 1 Tranche 5 workflow — checkpoint-attached DreamBank + measured lift."""

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
from src.eval.ring1_eval_harness import (
    run_eval_harness_dual,
    split_traces,
    split_traces_by_seed,
    write_eval_artifacts,
)
from src.eval.ring1_gate_verifier import verify_all_gates, write_gate_report
from src.eval.ring1_measured_lift import bank_from_dreambank_preferences, measure_dreambank_lift
from src.eval.ring1_model_loader import load_ring1_amc_model, resolve_checkpoint_weights_path
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
    return jsonl_path


def resolve_model_checkpoint_path(config: dict) -> Path:
    base = Path(config["model"]["checkpoint_path"])
    step = config["model"].get("checkpoint_step")
    if step:
        candidate = base / step
        if resolve_checkpoint_weights_path(candidate) is not None:
            return candidate
    return base


def ensure_traces(
    *,
    config: dict,
    min_traces: int,
) -> tuple[Path, Path]:
    amc_dir = Path(config["collection"]["amc_output_dir"])
    baseline_dir = Path(config["collection"]["baseline_output_dir"])
    amc_path = amc_dir / "traces.jsonl"
    baseline_path = baseline_dir / "traces.jsonl"

    reuse = config["collection"].get("reuse_existing_traces", True)
    amc_count = len(load_traces(amc_path)) if amc_path.exists() else 0
    baseline_count = len(load_traces(baseline_path)) if baseline_path.exists() else 0

    if reuse and amc_count >= min_traces and baseline_count >= min_traces:
        return amc_path, baseline_path

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
    return amc_path, baseline_path


def run_workflow(config_path: Path) -> dict:
    config = load_config(config_path)
    command = " ".join(sys.argv)
    min_traces = config["collection"]["min_traces_per_condition"]
    output_root = Path(config["reproducibility"]["output_dir"])

    amc_path, baseline_path = ensure_traces(config=config, min_traces=min_traces)

    checkpoint_path = resolve_model_checkpoint_path(config)
    device = config.get("model", {}).get("device", "cpu")
    model, weights_path, checkpoint_sha = load_ring1_amc_model(
        checkpoint_path,
        device=device,
        use_hlm_bank=True,
    )

    dreambank_dir = output_root / "dreambank_run"
    dreambank_result = run_sleep_on_trace_file(
        amc_path,
        bank_config=config.get("dreambank", {}).get("bank", {}),
        dreambank_config=config.get("dreambank", {}),
        num_cycles=config.get("dreambank", {}).get("cycles", 1),
        max_seeds=config.get("dreambank", {}).get("max_seeds", 64),
        model=model,
    )
    write_dreambank_artifacts(dreambank_result, dreambank_dir)

    amc_traces = load_traces(amc_path)
    baseline_traces = load_traces(baseline_path)
    _, holdout_amc = split_traces(
        amc_traces,
        holdout_fraction=config["eval"]["holdout_fraction"],
        seed=config["seeds"]["base_seed"],
    )
    holdout_seeds = {trace.get("seed") for trace in holdout_amc}
    _, holdout_baseline = split_traces_by_seed(baseline_traces, holdout_seeds)

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

    filled_bank = bank_from_dreambank_preferences(
        dreambank_result.preferences,
        bank_config={
            **config.get("dreambank", {}).get("bank", {}),
            "bank_dim": model.config.hlm_bank_dim or model.config.kv_lrank,
        },
    )

    measured_lift = None
    if config.get("eval", {}).get("use_measured_lift", True):
        measured_lift = measure_dreambank_lift(
            holdout_amc,
            model=model,
            filled_bank=filled_bank,
            lift_delta_scale=config["eval"].get("lift_delta_scale", 50.0),
            param_hash_unchanged=bool(
                dreambank_result.zero_grad_proof
                and dreambank_result.zero_grad_proof.param_hash_unchanged
            ),
            bank_fill=dreambank_result.bank_fill,
        )

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
        measured_lift=measured_lift,
    )
    metrics_dir = output_root / "metrics"
    write_eval_artifacts(eval_report, metrics_dir)

    config_with_sha = dict(config)
    config_with_sha.setdefault("model", {})["checkpoint_sha256"] = checkpoint_sha
    config_with_sha["model"]["weights_path"] = str(weights_path)

    tarball, pack_dir = write_repro_pack(
        config=config_with_sha,
        output_dir=output_root,
        traces_path=amc_path,
        dreambank_dir=dreambank_dir,
        metrics_dir=metrics_dir,
        repo_root=_REPO_ROOT,
    )

    gate_report = verify_all_gates(
        config=config_with_sha,
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
        "checkpoint_weights": str(weights_path),
        "checkpoint_sha256": checkpoint_sha,
        "param_hash_unchanged": dreambank_result.zero_grad_proof.param_hash_unchanged
        if dreambank_result.zero_grad_proof
        else None,
        "lift_mode": "measured" if measured_lift is not None else "proxy",
        "absolute_lift_pp": eval_report.dreambank_lift.absolute_lift_pp
        if eval_report.dreambank_lift
        else None,
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

    parser = argparse.ArgumentParser(description="Ring 1 Tranche 5 measured-lift workflow")
    parser.add_argument("--config", type=Path, default=Path("configs/ring1_tranche5.yaml"))
    args = parser.parse_args(argv)
    summary = run_workflow(args.config)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("preliminary_all_gates_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
