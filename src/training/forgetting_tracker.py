"""
src/training/forgetting_tracker.py — Forgetting-Aware Multi-Task SFT.

Tracks per-task accuracy during multi-task SFT and schedules replay
data for tasks that show signs of catastrophic forgetting.

Reference: "Forgetting-Aware SFT for Aurelius" (Aurelius, 2026)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class ForgettingTrackerConfig:
    """Configuration for forgetting-aware training.

    Attributes:
        forgetting_threshold: Score above which replay is triggered.
        replay_proportion: Fraction of batch to allocate to replay tasks.
        max_task_history: Max accuracy measurements to keep per task.
        ema_alpha: Smoothing factor for accuracy measurements.
        min_measurements: Minimum measurements before forgetting is considered.
    """

    forgetting_threshold: float = 0.05
    replay_proportion: float = 0.3
    max_task_history: int = 50
    ema_alpha: float = 0.3
    min_measurements: int = 3


@dataclass
class TaskSnapshot:
    """Accuracy snapshot for a task family."""

    task: str
    accuracy: float
    examples_seen: int
    step: int


class ForgettingTracker:
    """Tracks per-task forgetting and computes replay proportions.

    For each task family, maintains an accuracy history. ForgettingScore =
    max(history) - current. When score > threshold, replay data for that
    task is mixed into subsequent training batches.

    Usage:
        tracker = ForgettingTracker(config)
        tracker.log_accuracy(task="code_tool", accuracy=0.85, step=100)
        replay_weights = tracker.get_replay_weights()
        # Mix replay_weights[tasks] proportion into next batch
    """

    def __init__(self, config: ForgettingTrackerConfig | None = None) -> None:
        self.config = config or ForgettingTrackerConfig()
        self._history: dict[str, list[float]] = defaultdict(list)
        self._snapshots: list[TaskSnapshot] = []
        self._current_step: int = 0

    def log_accuracy(
        self,
        task: str,
        accuracy: float,
        step: int | None = None,
        examples_seen: int = 0,
    ) -> None:
        """Record an accuracy measurement for a task family.

        Args:
            task: Task family name (e.g., 'code', 'tool', 'math', 'safety').
            accuracy: Measured accuracy in [0, 1].
            step: Global training step.
            examples_seen: Number of training examples seen for this task.
        """
        if step is not None:
            self._current_step = step

        history = self._history[task]
        if history and self.config.ema_alpha > 0:
            # Apply EMA smoothing
            smoothed = self.config.ema_alpha * accuracy + (1 - self.config.ema_alpha) * history[-1]
            history.append(smoothed)
        else:
            history.append(accuracy)

        # Trim history
        if len(history) > self.config.max_task_history:
            self._history[task] = history[-self.config.max_task_history :]

        self._snapshots.append(
            TaskSnapshot(
                task=task,
                accuracy=accuracy,
                examples_seen=examples_seen,
                step=self._current_step,
            )
        )

        logger.debug(
            "ForgettingTracker: task=%s accuracy=%.4f (history_len=%d)",
            task,
            accuracy,
            len(self._history[task]),
        )

    def forgetting_score(self, task: str) -> float:
        """Compute forgetting score for a task.

        score = max(history) - current_accuracy
        Higher score means more forgetting.
        """
        history = self._history.get(task, [])
        if len(history) < self.config.min_measurements:
            return 0.0
        return max(history) - history[-1]

    def is_forgetting(self, task: str) -> bool:
        """Check if a task is showing significant forgetting."""
        return self.forgetting_score(task) > self.config.forgetting_threshold

    def get_replay_weights(self) -> dict[str, float]:
        """Return replay proportion for each task.

        Tasks with forgetting > threshold get replay_proportion.
        Tasks without forgetting get 0 replay.
        Returns dict of {task_name: replay_weight}.
        """
        if not self._history:
            return {}

        forgetting_tasks = {
            task: self.forgetting_score(task)
            for task in self._history
            if self.forgetting_score(task) > self.config.forgetting_threshold
        }

        if not forgetting_tasks:
            return {task: 0.0 for task in self._history}

        # Normalize forgetting scores to [0, 1] and scale by replay_proportion
        max_score = max(forgetting_tasks.values())
        if max_score <= 0:
            return {task: 0.0 for task in self._history}

        weights = {
            task: (score / max_score) * self.config.replay_proportion
            for task, score in forgetting_tasks.items()
        }

        # Non-forgetting tasks get 0 replay
        for task in self._history:
            if task not in weights:
                weights[task] = 0.0

        return weights

    def get_tasks_needing_replay(self) -> list[str]:
        """Return list of tasks currently needing replay data."""
        return [
            task
            for task in self._history
            if self.forgetting_score(task) > self.config.forgetting_threshold
        ]

    def best_accuracy(self, task: str) -> float | None:
        """Return the best observed accuracy for a task."""
        history = self._history.get(task)
        if not history:
            return None
        return max(history)

    def current_accuracy(self, task: str) -> float | None:
        """Return the most recent accuracy for a task."""
        history = self._history.get(task)
        if not history:
            return None
        return history[-1]

    def summary(self) -> dict[str, dict[str, float]]:
        """Return a per-task summary of forgetting status."""
        return {
            task: {
                "current": self.current_accuracy(task) or 0.0,
                "best": self.best_accuracy(task) or 0.0,
                "delta": -(self.forgetting_score(task)),
                "needs_replay": 1.0 if self.is_forgetting(task) else 0.0,
            }
            for task in sorted(self._history.keys())
        }

    def save_state(self, path: str | Path) -> None:
        """Save tracker state for resumption."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "history": {k: v for k, v in self._history.items()},
            "snapshots": [
                {
                    "task": s.task,
                    "accuracy": s.accuracy,
                    "examples_seen": s.examples_seen,
                    "step": s.step,
                }
                for s in self._snapshots
            ],
            "current_step": self._current_step,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str | Path) -> None:
        """Load tracker state from disk."""
        path = Path(path)
        with open(path) as f:
            state = json.load(f)
        self._history = defaultdict(list, state["history"])
        self._snapshots = [TaskSnapshot(**s) for s in state["snapshots"]]
        self._current_step = state["current_step"]


class ReplayBuffer:
    """Simple replay buffer that stores examples per task family.

    When a task shows forgetting, the replay buffer provides
    past training examples for that task to mix into the current batch.
    """

    def __init__(self, max_per_task: int = 1000) -> None:
        self.max_per_task = max_per_task
        self._buffer: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(self, task: str, example: dict[str, Any]) -> None:
        """Add an example to the task's replay buffer."""
        buf = self._buffer[task]
        buf.append(example)
        if len(buf) > self.max_per_task:
            self._buffer[task] = buf[-self.max_per_task :]

    def sample(
        self,
        task: str,
        n: int,
    ) -> list[dict[str, Any]]:
        """Sample n examples from a task's replay buffer."""
        buf = self._buffer.get(task, [])
        if not buf:
            return []
        indices = torch.randperm(len(buf))[:n].tolist()
        return [buf[i] for i in indices]

    def __len__(self) -> int:
        return sum(len(v) for v in self._buffer.values())


# Registry entry
FORGETTING_TRACKER_REGISTRY: dict[str, type] = {
    "forgetting_tracker": ForgettingTracker,
    "replay_buffer": ReplayBuffer,
}
