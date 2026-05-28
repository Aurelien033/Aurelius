"""Tests for the DreamBank dry-run CLI script."""

from __future__ import annotations

import json
import subprocess
import sys

_SCRIPT = "scripts/run_dreambank_cycle.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, _SCRIPT] + args,
        capture_output=True,
        text=True,
        timeout=30,
        cwd="/Users/christienantonio/aurelius",
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/Users/christienantonio/aurelius"},
    )


def test_dreambank_runner_dry_run_outputs_json() -> None:
    result = _run(["--dry-run", "--cycles", "1", "--seed", "test"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["dry_run"] is True
    assert "total_writes" in data
    assert "bank_fill" in data
    assert "mean_margin" in data


def test_dreambank_runner_respects_cycles() -> None:
    result = _run(["--dry-run", "--cycles", "3", "--seed", "repeat"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["cycles"] == 3


def test_dreambank_runner_rejects_negative_cycles() -> None:
    result = _run(["--dry-run", "--cycles", "-1", "--seed", "x"])
    assert result.returncode != 0
    assert "cycles" in result.stderr.lower()


def test_dreambank_runner_zero_cycles() -> None:
    result = _run(["--dry-run", "--cycles", "0", "--seed", "none"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["total_writes"] == 0
    assert data["bank_fill"] == 0


def test_dreambank_runner_multiple_seeds() -> None:
    result = _run(["--dry-run", "--cycles", "1", "--seed", "a", "b", "c"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert len(data["seeds"]) == 3
