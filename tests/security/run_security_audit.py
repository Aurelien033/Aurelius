#!/usr/bin/env python3
"""Run AMC security audit and write JSON report (T29)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.security.amc_security_audit import AMCSecurityAudit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMC adversarial memory security audit")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path (recorded only)")
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs/reproducibility/results/security_audit.json",
        help="JSON output path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = AMCSecurityAudit()
    results = audit.run_all()
    payload = audit.to_json()
    payload["checkpoint"] = args.checkpoint or None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== SECURITY AUDIT ===")
    for result in results:
        flag = "✓" if result.status == "PASS" else "✗"
        suffix = f" ({result.note})" if result.note else ""
        print(f"  {flag} {result.test}: {result.status}{suffix}")

    failed = [result for result in results if result.status != "PASS"]
    print(f"\nResult: {len(results) - len(failed)}/{len(results)} PASS")
    if failed:
        print(f"FAILURES: {[result.test for result in failed]}")
        return 1

    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
