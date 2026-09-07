#!/usr/bin/env python3
"""Analyze ablation JSONL and optionally render a figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ablation import AblationResult, summarize_results


def _load(path: Path) -> list[AblationResult]:
    rows: list[AblationResult] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                AblationResult(
                    config=payload["config"],
                    benchmark=payload["benchmark"],
                    score=float(payload["score"]),
                    stderr=float(payload.get("stderr", 0.0)),
                    n_samples=int(payload.get("n_samples", 0)),
                    checkpoint=payload.get("checkpoint", ""),
                    config_path=payload.get("config_path", ""),
                    raw_scores=tuple(payload.get("raw_scores", ())),
                    p_value_vs_baseline=payload.get("p_value_vs_baseline"),
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional figure path (delegates to scripts/plot_ablation.py)",
    )
    args = parser.parse_args()

    results = _load(args.results)
    print(summarize_results(results))

    if args.output is not None:
        import importlib.util

        plot_path = _REPO_ROOT / "scripts" / "plot_ablation.py"
        spec = importlib.util.spec_from_file_location("plot_ablation", plot_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load {plot_path}")
        plot_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_mod)
        payload = plot_mod.load_results(args.results)
        plot_mod.plot_bar_chart(payload, args.output)
        plot_mod.plot_ascii_table(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
