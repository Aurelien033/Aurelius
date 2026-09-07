#!/usr/bin/env python3
"""Run post-training evaluation suite for Aurelius-Forge (T22).

Usage:
    python scripts/evaluate_forge.py \\
        --checkpoint logs/forge_1b_run_001/checkpoints/checkpoint-final.pt \\
        --output-dir logs/forge_1b_eval \\
        --profile smoke

    # Full CI-style suite (oracle/mock-friendly without GPU):
    python scripts/evaluate_forge.py --checkpoint path/to.ckpt --profile ci --mode oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.amc_memory_runner import run_benchmark as run_amc_memory
from src.eval.run_gsm8k import run_eval as run_gsm8k_eval
from src.eval.run_mmlu import run_eval as run_mmlu_eval


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Forge AMC checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument(
        "--config",
        default="configs/amc_forge_1b.yaml",
        help="Model config YAML (recorded in outputs)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON outputs")
    parser.add_argument(
        "--profile",
        choices=("smoke", "ci", "stress"),
        default="smoke",
        help="Benchmark difficulty profile",
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "mock", "engine"),
        default=None,
        help="Generator mode for GSM8K/MMLU (default: oracle for smoke, engine otherwise)",
    )
    parser.add_argument("--backend", default="mock", help="Serving backend when mode=engine")
    parser.add_argument("--skip-amc", action="store_true", help="Skip AMC-Memory benchmark")
    parser.add_argument("--skip-gsm8k", action="store_true", help="Skip GSM8K")
    parser.add_argument("--skip-mmlu", action="store_true", help="Skip MMLU")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    gsm_mmlu_profile = "smoke" if args.profile == "smoke" else "ci"

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "profile": args.profile,
        "results": {},
    }

    if not args.skip_amc:
        amc_generator = (
            "oracle" if args.profile == "smoke" and mode in (None, "oracle") else "engine"
        )
        amc_backend = args.backend if amc_generator == "engine" else "mock"
        amc_profile = args.profile if args.profile in ("smoke", "ci", "stress") else "ci"
        amc_payload = run_amc_memory(
            generator=amc_generator,
            profile=amc_profile,
            backend=amc_backend,
            model_path=args.checkpoint,
        )
        _write_json(out_dir / "amc_memory.json", amc_payload)
        summary["results"]["amc_memory"] = {
            "overall_score": amc_payload.get("overall_score"),
            "gate": amc_payload.get("gate"),
        }

    if not args.skip_gsm8k:
        gsm_payload = run_gsm8k_eval(
            checkpoint=args.checkpoint,
            config=args.config,
            profile=gsm_mmlu_profile,
            mode=mode,
            backend=args.backend,
        )
        _write_json(out_dir / "gsm8k.json", gsm_payload)
        summary["results"]["gsm8k"] = {
            "accuracy": gsm_payload.get("accuracy"),
            "score": gsm_payload.get("score"),
        }

    if not args.skip_mmlu:
        mmlu_payload = run_mmlu_eval(
            checkpoint=args.checkpoint,
            config=args.config,
            profile=gsm_mmlu_profile,
            mode=mode,
            backend=args.backend,
        )
        _write_json(out_dir / "mmlu.json", mmlu_payload)
        summary["results"]["mmlu"] = {
            "accuracy": mmlu_payload.get("accuracy"),
            "score": mmlu_payload.get("score"),
        }

    _write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    summary = run_suite(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote evaluation artifacts to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
