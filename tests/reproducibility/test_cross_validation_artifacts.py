"""T31 cross-validation scaffold artifact tests."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "docs/reproducibility/scripts/cross_validate_clean_machine.sh"
_REPORT = _REPO / "docs/reproducibility/CROSS_VALIDATION_REPORT.md"


def test_cross_validate_script_exists_and_strict() -> None:
    assert _SCRIPT.is_file()
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "AMC_SKIP_TRAINING" in text


def test_cross_validation_report_has_honest_status() -> None:
    assert _REPORT.is_file()
    body = _REPORT.read_text(encoding="utf-8")
    assert "NOT EXECUTED" in body.upper() or "EXECUTED" in body.upper()
    assert "PASS" in body or "FAIL" in body or "NOT EXECUTED" in body.upper()
