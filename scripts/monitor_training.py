#!/usr/bin/env python3
"""Monitor a running AMC training job and alert on anomalies.

Usage:
    python scripts/monitor_training.py logs/<run>/training.jsonl

Alerts on:
- Loss NaN/Inf
- Loss not decreasing over N steps (stagnation)
- Surprise accuracy stuck below 55%
- Promotion rate outside [0.1, 0.9]
- Training stalled (no new log lines)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

STAGNATION_WINDOW = 500
ALERT_COOLDOWN = 300
PROMOTION_RATE_MIN = 0.1
PROMOTION_RATE_MAX = 0.9
SURPRISE_ACCURACY_MIN = 0.55
STALL_SECONDS = 600


class TrainingMonitor:
    """Tail training.jsonl and emit stderr alerts on metric anomalies."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.loss_history: deque[float] = deque(maxlen=STAGNATION_WINDOW)
        self.last_alert_time: dict[str, float] = {}
        self.last_line_time = time.time()

    def check_losses_decreasing(self) -> str | None:
        if len(self.loss_history) < STAGNATION_WINDOW:
            return None
        window = list(self.loss_history)
        quarter = STAGNATION_WINDOW // 4
        first_quarter_avg = sum(window[:quarter]) / quarter
        last_quarter_avg = sum(window[-quarter:]) / quarter
        if last_quarter_avg >= first_quarter_avg * 0.99:
            return (
                f"loss stalled: first 1/4 avg={first_quarter_avg:.4f}, "
                f"last 1/4 avg={last_quarter_avg:.4f}"
            )
        return None

    def check_nan_inf(self, metrics: dict[str, Any]) -> str | None:
        for key, value in metrics.items():
            if isinstance(value, float) and (value != value or value == float("inf")):
                return f"{key} is NaN/Inf: {value}"
        return None

    def check_promotion_rate(self, metrics: dict[str, Any]) -> str | None:
        rate = metrics.get("promotion_rate")
        if rate is not None:
            rate_f = float(rate)
            if rate_f < PROMOTION_RATE_MIN or rate_f > PROMOTION_RATE_MAX:
                return f"promotion rate out of range: {rate_f:.4f}"
        return None

    def check_surprise_accuracy(self, metrics: dict[str, Any]) -> str | None:
        acc = metrics.get("eval_surprise_accuracy")
        if acc is not None and float(acc) < SURPRISE_ACCURACY_MIN:
            return f"surprise accuracy below target: {float(acc):.4f}"
        return None

    def check_stalled(self) -> str | None:
        if time.time() - self.last_line_time > STALL_SECONDS:
            return f"no new log lines for {STALL_SECONDS}s"
        return None

    def maybe_alert(self, alert_key: str, message: str) -> bool:
        now = time.time()
        if now - self.last_alert_time.get(alert_key, 0) < ALERT_COOLDOWN:
            return False
        self.last_alert_time[alert_key] = now
        print(f"\n⚠️  {alert_key}: {message}\n", file=sys.stderr, flush=True)
        return True

    def process_entry(self, entry: dict[str, Any]) -> None:
        step = entry.get("step", "?")
        total = entry.get("total_loss")
        if total is not None:
            self.loss_history.append(float(total))

        checks: list[tuple[str, str | None]] = [
            ("nan_inf", self.check_nan_inf(entry)),
            ("loss_stagnated", self.check_losses_decreasing()),
            ("promotion_rate", self.check_promotion_rate(entry)),
            ("surprise_accuracy", self.check_surprise_accuracy(entry)),
        ]
        for name, result in checks:
            if result:
                self.maybe_alert(name, f"step {step}: {result}")

    def tail(self, check_interval: float = 5.0) -> None:
        """Follow the log and alert on anomalies."""
        if not self.log_path.is_file():
            raise FileNotFoundError(f"log file not found: {self.log_path}")

        print(f"Monitoring {self.log_path}...", flush=True)
        with self.log_path.open(encoding="utf-8") as handle:
            handle.seek(0, 2)
            try:
                while True:
                    line = handle.readline()
                    if line:
                        self.last_line_time = time.time()
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(entry, dict):
                            self.process_entry(entry)
                    else:
                        stalled = self.check_stalled()
                        if stalled:
                            self.maybe_alert("training_stalled", stalled)
                        time.sleep(check_interval)
            except KeyboardInterrupt:
                print("\nMonitor stopped.", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor AMC training.jsonl")
    parser.add_argument("log_path", type=Path, help="Path to training.jsonl")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between polls when log is idle",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    monitor = TrainingMonitor(args.log_path)
    monitor.tail(check_interval=args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
