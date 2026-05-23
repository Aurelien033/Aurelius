#!/usr/bin/env python3
"""Run AMC ablation study (T28).

Usage:
    python scripts/run_ablation.py \\
        --checkpoint logs/forge_1b_run_001/checkpoint-final.pt \\
        --output docs/reproducibility/results/ablation_scores.jsonl \\
        --mode oracle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ablation import CONFIGS, DEFAULT_BENCHMARKS, run_ablation_study, summarize_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMC ablation study (4 configs × benchmarks)")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path (required for --mode engine)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: docs/reproducibility/results/ablation_scores.jsonl)",
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "engine"),
        default="oracle",
        help="Benchmark generator mode (oracle for CI/smoke)",
    )
    parser.add_argument("--backend", default="mock", help="Engine backend when mode=engine")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(CONFIGS.keys()),
        help="Subset of ablation configs",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to run",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default_out = _REPO_ROOT / "docs/reproducibility/results" / "ablation_scores.jsonl"
    output_path = args.output or default_out
    output_dir = output_path.parent

    results, written_path = run_ablation_study(
        args.checkpoint,
        configs=args.configs,
        benchmarks=args.benchmarks,
        output_dir=output_dir,
        output_path=output_path,
        mode=args.mode,
        backend=args.backend,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
        repo_root=_REPO_ROOT,
    )

    print(summarize_results(results))
    print(f"Wrote {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
