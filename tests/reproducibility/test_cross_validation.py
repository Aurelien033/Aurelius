"""Tests for T31 cross-validation runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_cross_validation import (
    CONFIG_ORDER,
    run_cross_validation,
)


def test_smoke_cross_validation_passes() -> None:
    result = run_cross_validation(profile="smoke")
    assert result.passed, [(check.name, check.status, check.detail) for check in result.checks]


def test_render_report_contains_check_table() -> None:
    from scripts.run_cross_validation import CrossValidationCheck, CrossValidationResult, render_report

    result = CrossValidationResult(
        profile="smoke",
        host="test-host",
        python_version="3.12.0",
        timestamp="2026-05-23T00:00:00+00:00",
        checks=[
            CrossValidationCheck("example", "PASS", "ok"),
        ],
    )
    text = render_report(result)
    assert "ablation_monotonicity" not in text
    assert "| example | PASS | ok |" in text
    assert "Overall:** PASS" in text


def test_config_order_constant() -> None:
    assert list(CONFIG_ORDER) == ["baseline", "tier1_only", "tier12", "full_amc"]
