"""
src/training/prm_training_data.py — PRM (Process Reward Model) Training-Data Pipeline.

Generates step-level correctness labels from code execution traces so that
the existing ``ProcessSupervision`` / ``ProcessRewardModel`` modules can
be trained without hand-labelled data.

Pipeline
--------
1. **Trajectory collection**  — run model on tasks, record each generated
   step (token or CoT sentence) together with the post-step execution state.
2. **Auto-labelling**          — run test suite at each check-point; a step
   is labelled ``y_t = 1`` iff all tests pass after that step.
3. **Filtering**               — drop steps where the execution state is
   identical to the previous step (no progress) or where the label is
   ambiguous (e.g. syntax error prevented execution).
4. **Packing**                 — group into (task_id, steps, labels) tuples
   compatible with ``ProcessSupervision`` (B, T, d_model) inputs.

The output format is consumed by ``PRMTrainer`` in ``process_reward_model.py``.

REF: Lightman et al. arXiv:2305.20050; v5 Process Reward Model mechanism.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PRMTrainingDataConfig:
    """Configuration for PRM training-data generation.

    Attributes:
        checkpoint_every:   Execute tests every N generated tokens/steps.
                            Smaller means denser labels but higher compute cost.
        min_steps:          Minimum number of labelled steps in a trajectory
                            for it to be retained.  Default 2.
        drop_no_progress:   Drop steps where execution state is identical
                            to the previous step's state.
        drop_ambiguous:     Drop steps where execution raised an exception
                            before any test could run.
        output_dir:         Directory for ``.jsonl`` output shards.
        shard_size:         Maximum trajectories per output shard file.
    """

    checkpoint_every: int = 1
    min_steps: int = 2
    drop_no_progress: bool = True
    drop_ambiguous: bool = True
    output_dir: str | Path = ".prm_data"
    shard_size: int = 500


# ---------------------------------------------------------------------------
# Trajectory / label types
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One step in a generated solution trajectory.

    Attributes:
        step_idx:   Position in the trajectory (0-indexed).
        text:       Text generated at this step (cumulative output so far).
        test_state: Snapshot of test execution after this step, as a
                    ``dict`` returned by the test runner (must include
                    ``passed``, ``total``, ``error`` keys).
        label:      Binary correctness label (1 = all tests pass, 0 otherwise).
                    Populated by ``auto_label``.
    """

    step_idx: int
    text: str
    test_state: dict[str, Any] = field(default_factory=dict)
    label: int | None = None

    @property
    def state_hash(self) -> str:
        """Hash of the test state for progress/no-progress detection."""
        payload = json.dumps(self.test_state, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class LabelledTrajectory:
    """Fully labelled trajectory ready for PRM training.

    Attributes:
        task_id:      Unique task identifier.
        prompt:       Original prompt text.
        steps:        List of ``StepRecord`` objects (all with label set).
        final_score:  Fraction of tests passing in the final state.
        metadata:     Arbitrary metadata dict (model name, timestamp, etc.).
    """

    task_id: str
    prompt: str
    steps: list[StepRecord] = field(default_factory=list)
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def is_valid(self, min_steps: int = 2) -> bool:
        """Trajectory is valid iff all steps are labelled and ≥ min_steps."""
        return len(self.steps) >= min_steps and all(s.label is not None for s in self.steps)


# ---------------------------------------------------------------------------
# Labelling helpers
# ---------------------------------------------------------------------------


def _default_label_fn(test_state: dict[str, Any]) -> int | None:
    """Default auto-label: 1 if all tests pass, 0 if not, None if ambiguous.

    A step is ambiguous when:
      * execution raised an exception (``error`` key is non-empty), or
      * the test suite has not yet been run (``total`` key is absent / 0).
    """
    if not test_state:
        return None
    error = test_state.get("error")
    if error:
        return None  # ambiguous — couldn't test
    total = test_state.get("total", 0)
    passed = test_state.get("passed", 0)
    if total == 0:
        return None  # no tests available
    return int(passed == total)


def auto_label(
    steps: list[StepRecord],
    label_fn: Callable[[dict[str, Any]], int | None] | None = None,  # type: ignore[valid-type]
) -> list[StepRecord]:
    """Assign ``label`` to each step in ``steps`` using the label function.

    Args:
        steps:   List of ``StepRecord`` objects (test_state must be populated).
        label_fn: Callable that maps test_state dict to 0 or 1, or None if
                  ambiguous.  Defaults to ``_default_label_fn``.

    Returns:
        List of ``StepRecord`` objects with ``label`` field populated.
    """
    fn = label_fn or _default_label_fn
    for step in steps:
        step.label = fn(step.test_state)
    return steps


def filter_steps(
    steps: list[StepRecord],
    *,
    drop_no_progress: bool = True,
    drop_ambiguous: bool = True,
) -> list[StepRecord]:
    """Remove unwanted steps from a trajectory.

    Args:
        steps:               List of ``StepRecord`` objects.
        drop_no_progress:    Remove steps with identical state hash to previous step.
        drop_ambiguous:      Remove steps with ``label is None``.

    Returns:
        Filtered list (preserves original order).
    """
    filtered: list[StepRecord] = []
    prev_hash: str | None = None

    for step in steps:
        if drop_ambiguous and step.label is None:
            logger.debug("filter_steps: dropping ambiguous step at idx=%d", step.step_idx)
            continue
        if drop_no_progress and step.state_hash == prev_hash:
            logger.debug("filter_steps: dropping no-progress step at idx=%d", step.step_idx)
            continue
        filtered.append(step)
        prev_hash = step.state_hash

    return filtered


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------


class TrajectoryBuilder:
    """Collects step records during a model generation run.

    The caller drives generation; the builder records each step together
    with the post-execution state.  A typical usage pattern::

        builder = TrajectoryBuilder(task_id="human_eval_0", prompt=prompt)
        for token in model.generate(prompt):
            text_so_far = tokenizer.decode(all_tokens)
            test_result = test_runner(prompt, text_so_far, task_id)
            builder.record_step(text=text_so_far, test_state=test_result)
        trajectory = builder.build()
    """

    def __init__(self, task_id: str, prompt: str, metadata: dict | None = None) -> None:
        self.task_id = task_id
        self.prompt = prompt
        self.steps: list[StepRecord] = []
        self.metadata = metadata or {}

    def record_step(
        self,
        text: str,
        test_state: dict[str, Any] | None = None,
    ) -> None:
        """Record one step in the trajectory.

        Args:
            text:       Cumulative completion text at this step.
            test_state: Execution result dict (populated by test runner).
        """
        step = StepRecord(
            step_idx=len(self.steps),
            text=text,
            test_state=test_state or {},
        )
        self.steps.append(step)

    def build(
        self,
        final_score: float = 0.0,
        *,
        auto_label_steps: bool = True,
        filter_steps_after_label: bool = True,
        min_steps: int = 2,
    ) -> LabelledTrajectory | None:
        """Finalise and optionally label/filter the trajectory.

        Args:
            final_score:         Fraction of tests passing at the end.
            auto_label_steps:    Run ``auto_label`` before returning.
            filter_steps_after_label: Run ``filter_steps`` after labelling.
            min_steps:           Minimum steps required for a valid trajectory.

        Returns:
            ``LabelledTrajectory``, or ``None`` if the trajectory fails
            the ``is_valid`` check after optional filtering.
        """
        if auto_label_steps:
            auto_label(self.steps)
        if filter_steps_after_label:
            self.steps = filter_steps(
                self.steps,
                drop_no_progress=True,
                drop_ambiguous=True,
            )
        traj = LabelledTrajectory(
            task_id=self.task_id,
            prompt=self.prompt,
            steps=list(self.steps),
            final_score=final_score,
            metadata=dict(self.metadata),
        )
        return traj if traj.is_valid(min_steps=min_steps) else None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Sharded writer
# ---------------------------------------------------------------------------


class PRMDataWriter:
    """Writes labelled trajectories to ``.jsonl`` shards on disk.

    Each line in a shard is a JSON object with the keys:

        task_id, prompt, steps, final_score, metadata, n_steps

    Steps are encoded as a list of ``{step_idx, text, label, test_state}``
    dicts for downstream consumption.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        shard_size: int = 500,
        config: PRMTrainingDataConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or (config.output_dir if config else ".prm_data"))
        self.shard_size = shard_size if config is None else config.shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._shard_idx = 0
        self._total = 0

    def add(self, trajectory: LabelledTrajectory) -> None:
        """Add a labelled trajectory to the write buffer.

        Flushes to disk when the buffer reaches ``shard_size``.
        """
        record = self._trajectory_to_dict(trajectory)
        self._buffer.append(record)
        self._total += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _trajectory_to_dict(self, t: LabelledTrajectory) -> dict[str, Any]:
        return {
            "task_id": t.task_id,
            "prompt": t.prompt,
            "final_score": t.final_score,
            "metadata": t.metadata,
            "n_steps": len(t.steps),
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "text": s.text,
                    "label": s.label,
                    "test_state": s.test_state,
                }
                for s in t.steps
            ],
        }

    def _flush(self) -> None:
        if not self._buffer:
            return
        path = self.output_dir / f"prm_shard_{self._shard_idx:05d}.jsonl"
        with path.open("w") as fh:
            for record in self._buffer:
                fh.write(json.dumps(record) + "\n")
        logger.info("PRMDataWriter: flushed %d records → %s", len(self._buffer), path)
        self._buffer = []
        self._shard_idx += 1

    def close(self) -> None:
        """Flush remaining buffered records to disk."""
        self._flush()

    @property
    def total_written(self) -> int:
        """Total trajectories written across all shards."""
        return self._total

    def __enter__(self) -> PRMDataWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
