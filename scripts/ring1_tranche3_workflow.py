#!/usr/bin/env python3
"""Ring 1 Tranche 3 workflow — DreamBank sleep + eval harness + repro pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ring1_dreambank_runner import run_sleep_on_trace_file, write_dreambank_artifacts
from src.eval.ring1_eval_harness import run_eval_harness, write_eval_artifacts
from src.eval.ring1_repro_pack import validate_pack, write_repro_pack


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_workflow(config: dict, *, validate_only: bool = False) -> dict:
    traces_path = Path(config["eval"]["trace_input"])
    if not traces_path.exists():
        raise FileNotFoundError(
            f"Trace input not found: {traces_path}. "
            "Run ring1_trace_collector.py for Tranche 2 first."
        )

    output_root = Path(config["reproducibility"]["output_dir"])
    dreambank_dir = output_root / "dreambank_run"
    metrics_dir = output_root / "metrics"

    if validate_only:
        pack_dirs = [p for p in output_root.glob("ring1-repro-*") if p.is_dir()]
        if not pack_dirs:
            raise FileNotFoundError(f"No repro pack directory under {output_root}")
        errors = validate_pack(sorted(pack_dirs)[-1])
        if errors:
            raise RuntimeError("Pack validation failed: " + "; ".join(errors))
        return {"status": "validated", "pack": str(sorted(pack_dirs)[-1])}

    dreambank_result = run_sleep_on_trace_file(
        traces_path,
        bank_config=config.get("dreambank", {}).get("bank", {}),
        dreambank_config=config.get("dreambank", {}),
        num_cycles=config.get("dreambank", {}).get("cycles", 1),
        max_seeds=config.get("dreambank", {}).get("max_seeds", 32),
    )
    write_dreambank_artifacts(dreambank_result, dreambank_dir)

    eval_report = run_eval_harness(
        amc_traces_path=traces_path,
        holdout_fraction=config.get("eval", {}).get("holdout_fraction", 0.2),
        seed=config.get("seeds", {}).get("base_seed", 0),
        dreambank_bank_fill=dreambank_result.bank_fill,
        param_hash_unchanged=bool(
            dreambank_result.zero_grad_proof
            and dreambank_result.zero_grad_proof.param_hash_unchanged
        ),
    )
    write_eval_artifacts(eval_report, metrics_dir)

    tarball, pack_dir = write_repro_pack(
        config=config,
        output_dir=output_root,
        traces_path=traces_path,
        dreambank_dir=dreambank_dir,
        metrics_dir=metrics_dir,
        repo_root=_REPO_ROOT,
    )

    errors = validate_pack(pack_dir)
    if errors:
        raise RuntimeError("Pack validation failed: " + "; ".join(errors))

    summary = {
        "status": "ok",
        "dreambank_writes": dreambank_result.total_writes,
        "bank_fill": dreambank_result.bank_fill,
        "param_hash_unchanged": dreambank_result.zero_grad_proof.param_hash_unchanged
        if dreambank_result.zero_grad_proof
        else None,
        "eval": eval_report.to_dict(),
        "repro_pack": str(tarball),
    }
    (output_root / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ring 1 Tranche 3 end-to-end workflow")
    parser.add_argument("--config", type=Path, default=Path("configs/ring1_tranche3.yaml"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    summary = run_workflow(config, validate_only=args.validate_only)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
