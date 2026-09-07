#!/usr/bin/env python3
"""Audit tokenized .npy shards for C-19-style repetition corruption (NEW-04)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _row_signature(row: np.ndarray) -> bytes:
    trimmed = row[row != 0]
    return trimmed.tobytes()


def audit_shard_dir(
    shard_dir: Path,
    *,
    sample_shards: int = 10,
    min_diversity_ratio: float = 0.9,
) -> dict:
    shard_files = sorted(shard_dir.glob("shard_*.npy"))
    if not shard_files:
        return {
            "shard_dir": str(shard_dir),
            "status": "SKIP",
            "reason": "no shard_*.npy files found",
            "shard_count": 0,
        }

    sampled = shard_files[:sample_shards]
    shard_reports: list[dict] = []
    failing: list[str] = []

    for shard_path in sampled:
        arr = np.load(shard_path)
        if arr.ndim != 2 or arr.shape[0] == 0:
            failing.append(str(shard_path))
            shard_reports.append(
                {
                    "path": str(shard_path),
                    "samples": int(arr.shape[0]) if arr.ndim >= 1 else 0,
                    "unique_rows": 0,
                    "diversity_ratio": 0.0,
                    "status": "FAIL",
                }
            )
            continue

        signatures = {_row_signature(arr[i]) for i in range(arr.shape[0])}
        diversity = len(signatures) / arr.shape[0]
        status = "PASS" if diversity >= min_diversity_ratio else "FAIL"
        if status == "FAIL":
            failing.append(str(shard_path))
        shard_reports.append(
            {
                "path": str(shard_path),
                "samples": int(arr.shape[0]),
                "unique_rows": len(signatures),
                "diversity_ratio": round(diversity, 4),
                "status": status,
            }
        )

    overall = "PASS" if not failing else "FAIL"
    return {
        "shard_dir": str(shard_dir),
        "status": overall,
        "shard_count": len(shard_files),
        "sampled": len(sampled),
        "min_diversity_ratio": min_diversity_ratio,
        "failing_shards": failing,
        "shards": shard_reports,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit tokenized shard diversity (NEW-04)")
    parser.add_argument(
        "shard_dir",
        nargs="?",
        default="training_data/tokenized",
        help="Directory containing shard_*.npy files",
    )
    parser.add_argument("--sample", type=int, default=10, help="Max shards to sample")
    parser.add_argument(
        "--min-diversity",
        type=float,
        default=0.9,
        help="Minimum unique-row ratio per shard (default 0.9)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs/remediation/evidence/2026-05-27/NEW-04-shard-audit.json",
        help="JSON report path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_shard_dir(
        Path(args.shard_dir),
        sample_shards=args.sample,
        min_diversity_ratio=args.min_diversity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["status"] == "SKIP":
        print(f"\nWrote skip report to {args.output}")
        return 0
    if report["status"] != "PASS":
        print(f"\nFAIL — {len(report.get('failing_shards', []))} shard(s) below diversity threshold")
        print(f"Wrote report to {args.output}")
        return 1
    print(f"\nPASS — wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
