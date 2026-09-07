"""Tests for scripts/monitor_training.py (T21 / S04)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_monitor():
    path = _REPO_ROOT / "scripts" / "monitor_training.py"
    spec = importlib.util.spec_from_file_location("monitor_training", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["monitor_training"] = module
    spec.loader.exec_module(module)
    return module


def test_check_nan_inf_detects_nan() -> None:
    monitor = _load_monitor().TrainingMonitor(Path("unused.jsonl"))
    assert monitor.check_nan_inf({"total_loss": float("nan")}) is not None
    assert monitor.check_nan_inf({"total_loss": 1.0}) is None


def test_check_promotion_rate_out_of_range() -> None:
    monitor = _load_monitor().TrainingMonitor(Path("unused.jsonl"))
    assert monitor.check_promotion_rate({"promotion_rate": 0.05}) is not None
    assert monitor.check_promotion_rate({"promotion_rate": 0.95}) is not None
    assert monitor.check_promotion_rate({"promotion_rate": 0.3}) is None


def test_check_surprise_accuracy_below_target() -> None:
    monitor = _load_monitor().TrainingMonitor(Path("unused.jsonl"))
    assert monitor.check_surprise_accuracy({"eval_surprise_accuracy": 0.4}) is not None
    assert monitor.check_surprise_accuracy({"eval_surprise_accuracy": 0.6}) is None


def test_check_losses_decreasing_requires_window() -> None:
    monitor = _load_monitor().TrainingMonitor(Path("unused.jsonl"))
    monitor.loss_history.extend([1.0] * 500)
    stalled = monitor.check_losses_decreasing()
    assert stalled is not None
    assert "stalled" in stalled


def test_process_entry_appends_loss_history() -> None:
    monitor = _load_monitor().TrainingMonitor(Path("unused.jsonl"))
    monitor.process_entry({"step": 1, "total_loss": 2.5, "promotion_rate": 0.25})
    assert len(monitor.loss_history) == 1
    assert monitor.loss_history[0] == 2.5
