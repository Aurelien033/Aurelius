#!/usr/bin/env python3
"""Generate reproducibility figures from bundled results (T30)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS = _REPO_ROOT / "docs" / "reproducibility" / "results"


def _load_plot_module():
    plot_path = _REPO_ROOT / "scripts" / "plot_ablation.py"
    spec = importlib.util.spec_from_file_location("plot_ablation", plot_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot AMC reproducibility results")
    parser.add_argument(
        "--ablation",
        type=Path,
        default=_RESULTS / "ablation_scores.jsonl",
        help="Ablation JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_RESULTS / "ablation_figure.pdf",
        help="Figure output path",
    )
    parser.add_argument(
        "--security",
        type=Path,
        default=_RESULTS / "security_audit.json",
        help="Security audit JSON path",
    )
    args = parser.parse_args()

    if args.ablation.is_file():
        plot_mod = _load_plot_module()
        results = plot_mod.load_results(args.ablation)
        print(f"Loaded {len(results)} ablation rows from {args.ablation}")
        plot_mod.plot_bar_chart(results, args.output)
        plot_mod.plot_ascii_table(results)
    else:
        print(f"Skip ablation plot: missing {args.ablation}", file=sys.stderr)

    if args.security.is_file():
        payload = json.loads(args.security.read_text(encoding="utf-8"))
        passed = payload.get("passed", 0)
        total = payload.get("total", 0)
        print(f"Security audit: {passed}/{total} passed")
        for row in payload.get("results", []):
            flag = "✓" if row.get("status") == "PASS" else "✗"
            print(f"  {flag} {row.get('test')}: {row.get('status')}")
    else:
        print(f"Skip security summary: missing {args.security}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
