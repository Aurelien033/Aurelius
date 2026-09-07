#!/usr/bin/env python3
"""Run reproducibility cross-validation (T31).

Executes the smoke/oracle reproducibility path locally, compares ablation
scores to reference tolerances, and writes CROSS_VALIDATION_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ablation import CONFIGS, DEFAULT_BENCHMARKS, run_ablation_study
from src.security.amc_security_audit import AMCSecurityAudit

Status = Literal["PASS", "FAIL", "WARN"]

REFERENCE_AMC_MEMORY_MEANS: dict[str, float] = {
    "baseline": 0.58,
    "tier1_only": 0.72,
    "tier12": 0.86,
    "full_amc": 0.95,
}
SCORE_TOLERANCE = 0.08
CONFIG_ORDER = ("baseline", "tier1_only", "tier12", "full_amc")


@dataclass(frozen=True)
class CrossValidationCheck:
    name: str
    status: Status
    detail: str = ""


@dataclass
class CrossValidationResult:
    profile: str
    host: str
    python_version: str
    timestamp: str
    checks: list[CrossValidationCheck]

    @property
    def passed(self) -> bool:
        return all(check.status == "PASS" for check in self.checks)


def _run_pytest_bundle() -> CrossValidationCheck:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/reproducibility/test_reproducibility_bundle.py",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return CrossValidationCheck(
            "bundle_structure", "PASS", "reproducibility bundle tests passed"
        )
    tail = (proc.stdout + proc.stderr)[-500:]
    return CrossValidationCheck("bundle_structure", "FAIL", tail)


def _run_forge_config_validation() -> CrossValidationCheck:
    script = _REPO_ROOT / "scripts" / "validate_amc_forge_config.py"
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=_REPO_ROOT, capture_output=True, text=True
    )
    if proc.returncode == 0:
        return CrossValidationCheck("forge_config", "PASS", "validate_amc_forge_config.py ok")
    return CrossValidationCheck("forge_config", "FAIL", proc.stderr[-400:] or proc.stdout[-400:])


def _check_ablation_results(results_path: Path) -> list[CrossValidationCheck]:
    checks: list[CrossValidationCheck] = []
    if not results_path.is_file():
        return [
            CrossValidationCheck(
                "ablation_output",
                "FAIL",
                f"missing {results_path}",
            )
        ]

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_rows = len(CONFIGS) * len(DEFAULT_BENCHMARKS)
    if len(rows) != expected_rows:
        checks.append(
            CrossValidationCheck(
                "ablation_row_count",
                "FAIL",
                f"expected {expected_rows} rows, got {len(rows)}",
            )
        )
    else:
        checks.append(
            CrossValidationCheck(
                "ablation_row_count",
                "PASS",
                f"{len(rows)} config×benchmark rows",
            )
        )

    amc_scores = {
        row["config"]: float(row["score"]) for row in rows if row.get("benchmark") == "amc_memory"
    }
    if set(amc_scores) != set(CONFIG_ORDER):
        checks.append(
            CrossValidationCheck(
                "ablation_configs",
                "FAIL",
                f"unexpected configs: {sorted(amc_scores)}",
            )
        )
    else:
        ordered = [amc_scores[name] for name in CONFIG_ORDER]
        monotonic = all(left < right for left, right in zip(ordered, ordered[1:]))
        checks.append(
            CrossValidationCheck(
                "ablation_monotonicity",
                "PASS" if monotonic else "FAIL",
                " → ".join(f"{name}={amc_scores[name]:.3f}" for name in CONFIG_ORDER),
            )
        )

        drift_notes: list[str] = []
        for name, ref in REFERENCE_AMC_MEMORY_MEANS.items():
            actual = amc_scores.get(name)
            if actual is None:
                continue
            if abs(actual - ref) > SCORE_TOLERANCE:
                drift_notes.append(f"{name}: {actual:.3f} vs ref {ref:.3f}")
        if drift_notes:
            checks.append(
                CrossValidationCheck(
                    "ablation_score_drift",
                    "WARN",
                    "; ".join(drift_notes),
                )
            )
        else:
            checks.append(
                CrossValidationCheck(
                    "ablation_score_drift",
                    "PASS",
                    f"within ±{SCORE_TOLERANCE} of reference means",
                )
            )

    return checks


def _run_security_audit() -> CrossValidationCheck:
    audit = AMCSecurityAudit()
    results = audit.run_all()
    failed = [item for item in results if item.status != "PASS"]
    if failed:
        return CrossValidationCheck(
            "security_audit",
            "FAIL",
            ", ".join(item.test for item in failed),
        )
    return CrossValidationCheck("security_audit", "PASS", "6/6 probes passed")


def _check_script_paths() -> CrossValidationCheck:
    """Flag absolute home paths baked into reproducibility scripts."""
    scripts_dir = _REPO_ROOT / "docs" / "reproducibility" / "scripts"
    offenders: list[str] = []
    for path in scripts_dir.glob("*"):
        if path.suffix not in {".sh", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "/home/" in text:
            offenders.append(path.name)
    if offenders:
        return CrossValidationCheck(
            "hardcoded_paths",
            "FAIL",
            f"absolute paths in: {', '.join(offenders)}",
        )
    return CrossValidationCheck(
        "hardcoded_paths", "PASS", "no /Users or /home literals in bundle scripts"
    )


def run_cross_validation(*, profile: str = "smoke") -> CrossValidationResult:
    timestamp = datetime.now(UTC).isoformat()
    checks: list[CrossValidationCheck] = [
        _run_forge_config_validation(),
        _run_pytest_bundle(),
        _check_script_paths(),
    ]

    out_dir = _REPO_ROOT / "docs" / "reproducibility" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = out_dir / "cross_validation_ablation.jsonl"

    if profile == "smoke":
        run_ablation_study(
            "",
            mode="oracle",
            output_path=ablation_path,
            repo_root=_REPO_ROOT,
        )
        checks.extend(_check_ablation_results(ablation_path))
        checks.append(_run_security_audit())
    else:
        checks.append(
            CrossValidationCheck(
                "full_gpu_profile",
                "WARN",
                "not run in this invocation; use cloud instance per README",
            )
        )

    return CrossValidationResult(
        profile=profile,
        host=platform.node(),
        python_version=platform.python_version(),
        timestamp=timestamp,
        checks=checks,
    )


def render_report(result: CrossValidationResult) -> str:
    lines = [
        "# AMC Reproducibility Cross-Validation Report (T31)",
        "",
        f"- **Date (UTC):** {result.timestamp}",
        f"- **Host:** {result.host}",
        f"- **Python:** {result.python_version}",
        f"- **Profile:** `{result.profile}`",
        f"- **Overall:** {'PASS' if result.passed else 'FAIL'}",
        "",
        "## Summary",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for check in result.checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {check.status} | {detail} |")

    lines.extend(
        [
            "",
            "## Procedure",
            "",
            "1. Clone repository on a clean machine (no cached venv/data).",
            "2. `bash docs/reproducibility/scripts/install.sh`",
            "3. `bash docs/reproducibility/scripts/ablation.sh` (oracle smoke) or full GPU path.",
            "4. `python scripts/run_cross_validation.py --profile smoke`",
            "5. Compare `results/ablation_scores.jsonl` to reference ordering:",
            "   `baseline < tier1_only < tier12 < full_amc` on `amc_memory`.",
            "",
            "## Deviations",
            "",
        ]
    )
    warnings = [check for check in result.checks if check.status == "WARN"]
    failures = [check for check in result.checks if check.status == "FAIL"]
    if not warnings and not failures:
        lines.append("None for smoke profile on this host.")
    else:
        for check in failures + warnings:
            lines.append(f"- **{check.name}** ({check.status}): {check.detail}")

    lines.extend(
        [
            "",
            "## Full GPU cross-validation (manual)",
            "",
            "Repeat on a fresh 4×A100 instance with:",
            "",
            "```bash",
            "bash docs/reproducibility/scripts/install.sh",
            "bash docs/reproducibility/scripts/download_data.sh",
            "bash docs/reproducibility/scripts/train.sh",
            "bash docs/reproducibility/scripts/evaluate.sh logs/<run>/checkpoint-final.pt",
            "bash docs/reproducibility/scripts/ablation.sh logs/<run>/checkpoint-final.pt",
            "ABLATION_MODE=engine python scripts/run_cross_validation.py --profile full",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMC reproducibility cross-validation (T31)")
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke=oracle local CI; full=placeholder for GPU cloud run",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPO_ROOT / "docs" / "reproducibility" / "CROSS_VALIDATION_REPORT.md",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=_REPO_ROOT / "docs" / "reproducibility" / "results" / "cross_validation.json",
    )
    args = parser.parse_args(argv)

    result = run_cross_validation(profile=args.profile)
    args.report.write_text(render_report(result), encoding="utf-8")
    args.json.write_text(
        json.dumps(
            {
                "passed": result.passed,
                **{k: v for k, v in asdict(result).items() if k != "checks"},
                "checks": [asdict(check) for check in result.checks],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(render_report(result))
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
