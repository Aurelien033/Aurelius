"""
src/training/data_flywheel.py — Data flywheel from deployment.

Collects inference logs (prompts, outputs, verifier results), converts
failures into training pairs, and closes the loop: deploy -> log ->
convert -> train -> deploy. The pipeline that generates inference
failures IS the training data pipeline.

Reference: "Data Flywheel for Aurelius" (Aurelius, 2026)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class FlywheelConfig:
    """Configuration for the data flywheel.

    Attributes:
        log_dir: Directory to store inference logs.
        output_dir: Directory to store converted training pairs.
        min_accepted_for_training: Minimum pairs before training triggers.
        max_pairs_per_cycle: Maximum pairs to keep per cycle.
        task_families: List of task families to track.
        rotate_logs_every_n_calls: Rotate log file frequency.
    """

    log_dir: str = "data/flywheel/logs"
    output_dir: str = "data/flywheel/training"
    min_accepted_for_training: int = 50
    max_pairs_per_cycle: int = 1000
    task_families: tuple[str, ...] = (
        "code", "tool", "math", "safety", "long_context", "general"
    )
    rotate_logs_every_n_calls: int = 1000


@dataclass
class InferenceLogEntry:
    """A single inference call log entry."""

    timestamp: str = ""
    prompt: str = ""
    model_output: str = ""
    verifier_name: str = ""
    verifier_pass: bool = False
    verifier_detail: str = ""
    fallback_triggered: bool = False
    task_family: str = "general"
    latency_ms: float = 0.0
    model_name: str = ""
    temperature: float = 0.0
    prompt_hash: str = ""
    output_hash: str = ""


@dataclass
class TrainingPair:
    """A training pair derived from inference logs."""

    prompt: str
    chosen: str  # The correct/acceptable output
    rejected: str  # The failed output
    task_family: str = "general"
    source_log_hash: str = ""
    verifier_detail: str = ""
    created_at: str = ""


class InferenceLogger:
    """Logs inference calls to rotating JSONL files.

    Each line is an InferenceLogEntry in JSON format.
    """

    def __init__(self, config: FlywheelConfig | None = None) -> None:
        self.config = config or FlywheelConfig()
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0
        self._current_file = self._rotate_file()

    def _rotate_file(self) -> Path:
        """Create a new log file with timestamp."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"inference_log_{ts}.jsonl"
        logger.info("Rotating inference log to %s", path)
        return path

    def log(
        self,
        prompt: str,
        model_output: str,
        verifier_name: str = "",
        verifier_pass: bool = False,
        verifier_detail: str = "",
        fallback_triggered: bool = False,
        task_family: str = "general",
        latency_ms: float = 0.0,
        model_name: str = "",
        temperature: float = 0.0,
    ) -> None:
        """Log a single inference call."""
        self._call_count += 1

        import hashlib
        entry = InferenceLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=prompt,
            model_output=model_output,
            verifier_name=verifier_name,
            verifier_pass=verifier_pass,
            verifier_detail=verifier_detail,
            fallback_triggered=fallback_triggered,
            task_family=task_family,
            latency_ms=latency_ms,
            model_name=model_name,
            temperature=temperature,
            prompt_hash=hashlib.md5(prompt.encode()).hexdigest()[:16],
            output_hash=hashlib.md5(model_output.encode()).hexdigest()[:16],
        )

        with open(self._current_file, "a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")

        if self._call_count % self.config.rotate_logs_every_n_calls == 0:
            self._current_file = self._rotate_file()

    def get_log_files(self) -> list[Path]:
        """Return all log files sorted by modification time."""
        return sorted(self.log_dir.glob("inference_log_*.jsonl"))

    def get_recent_logs(
        self,
        n: int = 100,
        task_family: str | None = None,
    ) -> list[InferenceLogEntry]:
        """Return the N most recent log entries, optionally filtered."""
        entries: list[InferenceLogEntry] = []
        for log_file in reversed(self.get_log_files()):
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if task_family and data.get("task_family") != task_family:
                    continue
                entries.append(InferenceLogEntry(**data))
                if len(entries) >= n:
                    return entries
        return entries


class LogToTrainingConverter:
    """Converts inference logs into chosen/rejected training pairs.

    For each log entry where verifier_pass is False (failure):
    - If a fallback or retry produced a correct output, use it as 'chosen'.
    - The failed output is 'rejected'.
    - Creates a DPO-style training pair.
    """

    def __init__(self, config: FlywheelConfig | None = None) -> None:
        self.config = config or FlywheelConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(
        self,
        failed_entries: list[InferenceLogEntry],
        successful_fallbacks: dict[str, str] | None = None,
    ) -> list[TrainingPair]:
        """Convert log entries to training pairs.

        Args:
            failed_entries: Log entries where verifier_pass is False.
            successful_fallbacks: Optional dict mapping prompt_hash to the
                correct output (from a fallback, retry, or user correction).

        Returns:
            List of TrainingPair objects.
        """
        successful_fallbacks = successful_fallbacks or {}
        pairs: list[TrainingPair] = []

        for entry in failed_entries:
            # Try to find the correct output
            chosen = successful_fallbacks.get(entry.prompt_hash, "")
            if not chosen:
                # If we don't have a correction, we can't create a pair
                continue

            pairs.append(TrainingPair(
                prompt=entry.prompt,
                chosen=chosen,
                rejected=entry.model_output,
                task_family=entry.task_family,
                source_log_hash=entry.prompt_hash,
                verifier_detail=entry.verifier_detail,
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

        # Limit pairs
        if len(pairs) > self.config.max_pairs_per_cycle:
            pairs = pairs[:self.config.max_pairs_per_cycle]

        logger.info(
            "Converted %d log entries into %d training pairs",
            len(failed_entries), len(pairs),
        )
        return pairs

    def save_pairs(
        self,
        pairs: list[TrainingPair],
        cycle: int = 0,
    ) -> Path:
        """Save training pairs as JSONL.

        Args:
            pairs: Training pairs to save.
            cycle: Flywheel cycle number.

        Returns:
            Path to saved file.
        """
        path = self.output_dir / f"flywheel_cycle_{cycle:04d}.jsonl"
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps({
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "task_family": pair.task_family,
                    "source_log_hash": pair.source_log_hash,
                    "verifier_detail": pair.verifier_detail,
                    "created_at": pair.created_at,
                }) + "\n")
        logger.info("Saved %d training pairs to %s", len(pairs), path)
        return path


class DataFlywheel:
    """Full data flywheel loop: log -> convert -> train -> deploy -> repeat.

    Usage:
        flywheel = DataFlywheel(logger, converter, model, trainer)
        flywheel.run_cycle()
    """

    def __init__(
        self,
        logger: InferenceLogger,
        converter: LogToTrainingConverter,
        model: Any,
        trainer_fn: Callable | None = None,
        config: FlywheelConfig | None = None,
    ) -> None:
        self.logger = logger
        self.converter = converter
        self.model = model
        self.trainer_fn = trainer_fn
        self.config = config or FlywheelConfig()
        self._cycle = 0

    def run_cycle(
        self,
        failed_entries: list[InferenceLogEntry],
        corrections: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run a single flywheel cycle.

        Args:
            failed_entries: Log entries where inference failed.
            corrections: Map of prompt_hash -> correct_output.

        Returns:
            Cycle results with pair count, training outcome, etc.
        """
        self._cycle += 1

        # Convert logs to training pairs
        pairs = self.converter.convert(failed_entries, corrections)

        if len(pairs) < self.config.min_accepted_for_training:
            logger.info(
                "Cycle %d: %d pairs < min %d, skipping training",
                self._cycle, len(pairs), self.config.min_accepted_for_training,
            )
            return {
                "cycle": self._cycle,
                "pairs": len(pairs),
                "trained": False,
                "reason": "insufficient_pairs",
            }

        # Save pairs
        pairs_path = self.converter.save_pairs(pairs, self._cycle)

        # Train
        if self.trainer_fn is not None:
            self.model = self.trainer_fn(self.model, pairs)

        logger.info(
            "Flywheel cycle %d complete: %d pairs, model updated",
            self._cycle, len(pairs),
        )

        return {
            "cycle": self._cycle,
            "pairs": len(pairs),
            "pairs_path": str(pairs_path),
            "trained": True,
        }


# Registry entry
DATA_FLYWHEEL_REGISTRY: dict[str, type] = {
    "inference_logger": InferenceLogger,
    "log_converter": LogToTrainingConverter,
    "data_flywheel": DataFlywheel,
}
