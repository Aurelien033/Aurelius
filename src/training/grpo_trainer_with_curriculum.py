"""
src/training/grpo_trainer_with_curriculum.py — Curriculum-Sampled RLVR Trainer.

Wraps the existing ``RLVRTrainer`` from ``rlvr.py`` with a
``CurriculumRLSampler`` from ``curriculum_rl.py`` so that task batches
are drawn from the learnability band (medium-accuracy tasks) with
per-step accuracy feedback driving the difficulty EMA.

v5 mechanism: execution-grounded credit assignment with adaptive prompt
difficulty sampling (replaces round-robin mixing).

Key design decisions
--------------------
* The wrapper composes (not replaces) the existing ``RLVRTrainer``.
  When no sampler is provided, behaviour is identical to standalone
  RLVR training — zero breaking changes for existing callers.
* After each ``train_step``, the mean reward is converted to a binary
  ``is_correct`` and fed back to all ``task_ids`` in the batch,
  keeping the difficulty EMA current per-train-step.
* ``sample_batch`` exposes the sampler as a public API so callers
  can query available tasks before constructing a training batch.

Usage::

    sampler = CurriculumRLSampler(CurriculumRLConfig())
    for task_id, prompt, answer in tasks:
        sampler.register_task(task_id, difficulty=0.5)

    trainer = CurriculumRLVRTrainer(
        policy_model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        reward_fn=mdv.reward_fn_wrap("code", test_runner=tool.run_tests),
        config=rlvr_config,
        sampler=sampler,
    )
    batch_ids = trainer.sample_batch(n=8)
    result = trainer.train_step(
        task_ids=batch_ids,
        prompt_ids=batch_prompt_ids,
        prompt_text=batch_prompt_text,
        answer=batch_answer,
    )
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from src.alignment.rlvr import RLVRTrainer, RLVRConfig
from src.training.curriculum_rl import CurriculumRLConfig, CurriculumRLSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurriculumRLVRConfig:
    """Configuration for curriculum-driven RLVR training.

    Attributes:
        rlvr:          RLVRConfig forwarded to RLVRTrainer.
        curriculum:    CurriculumRLConfig forwarded to CurriculumRLSampler.
        verbose:       Log sampler statistics after each train_step.
    """

    rlvr: RLVRConfig | None = None
    curriculum: CurriculumRLConfig | None = None
    verbose: bool = False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CurriculumRLVRTrainer:
    """RLVR trainer with curriculum-sampled task batches.

    Maintains an ``RLVRTrainer`` for the actual GRPO step, plus a
    ``CurriculumRLSampler`` for adaptive task selection and accuracy-
    driven difficulty EMA updates.

    Args:
        rlvr_trainer:  Pre-built ``RLVRTrainer`` instance.  If None, a new
                       one is constructed from ``(policy_model, ref_model,
                       optimizer, reward_fn, rlvr_config)``.
        sampler:       ``CurriculumRLSampler`` instance.
        config:        ``CurriculumRLVRConfig``.
    """

    def __init__(
        self,
        rlvr_trainer: RLVRTrainer | None = None,
        *,
        sampler: CurriculumRLSampler | None = None,
        policy_model: Any = None,
        ref_model: Any = None,
        optimizer: Any = None,
        reward_fn: Any = None,
        rlvr_config: RLVRConfig | None = None,
        config: CurriculumRLVRConfig | None = None,
    ) -> None:
        if rlvr_trainer is not None:
            self.rlvr_trainer = rlvr_trainer
        else:
            if None in (policy_model, ref_model, optimizer, reward_fn):
                raise ValueError(
                    "Either rlvr_trainer must be provided, or all of "
                    "(policy_model, ref_model, optimizer, reward_fn)."
                )
            self.rlvr_trainer = RLVRTrainer(
                policy_model=policy_model,
                ref_model=ref_model,
                reward_fn=reward_fn,
                config=rlvr_config or RLVRConfig(),
                optimizer=optimizer,
            )
        self.sampler = sampler
        self.config = config or CurriculumRLVRConfig()

    # ------------------------------------------------------------------ #
    # Core train_step                                                     #
    # ------------------------------------------------------------------ #

    def train_step(
        self,
        task_ids: Sequence[str],
        prompt_ids: torch.Tensor,
        prompt_text: str,
        answer: str = "",
        **reward_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one RLVR training step with sampler feedback.

        Args:
            task_ids:      Task identifiers (must be registered in the sampler).
            prompt_ids:    Shape ``(1, S)`` tokenised prompt.
            prompt_text:   Raw task prompt.
            answer:        Ground-truth answer string.
            **reward_kwargs: Forwarded to ``rlvr_trainer.train_step``.

        Returns:
            Dict with keys from the underlying ``rlvr_trainer.train_step``
            (``loss``, ``mean_reward``, ``n_samples``) plus
            ``sampler_stats`` when ``config.verbose`` is True.
        """
        if self.sampler is not None:
            self._warn_outside_zone(task_ids)

        result: dict[str, Any] = dict(
            self.rlvr_trainer.train_step(
                prompt_ids=prompt_ids,
                prompt_text=prompt_text,
                ground_truth=answer,
            )
        )

        if self.sampler is not None:
            mean_reward = float(result.get("mean_reward", 0.0))
            is_correct = mean_reward >= 0.5
            for tid in task_ids:
                try:
                    self.sampler.update(tid, is_correct=is_correct)
                except KeyError:
                    logger.debug("train_step: task %r not in sampler — skipped.", tid)

            if self.config.verbose:
                try:
                    result["sampler_stats"] = self.sampler.statistics()
                except Exception as exc:
                    logger.debug("Could not read sampler statistics: %s", exc)

        return result

    # ------------------------------------------------------------------ #
    # Batch selection helpers                                             #
    # ------------------------------------------------------------------ #

    def sample_batch(
        self,
        n: int,
        rng_seed: int | None = None,
    ) -> list[str]:
        """Return ``n`` task IDs sampled from the curriculum sampler.

        Args:
            n:        Number of task IDs to draw.
            rng_seed: Optional seed for reproducibility.

        Returns:
            Ordered list of task ID strings.

        Raises:
            RuntimeError: If no sampler is configured.
        """
        if self.sampler is None:
            raise RuntimeError(
                "CurriculumRLVRTrainer.sample_batch() requires a sampler. "
                "Construct with sampler=... keyword argument."
            )
        return self.sampler.sample(n=n, rng_seed=rng_seed)

    # ------------------------------------------------------------------ #
    # Sampler introspection                                               #
    # ------------------------------------------------------------------ #

    def sampler_statistics(self) -> dict[str, Any] | None:
        """Return current sampler statistics or None if no sampler configured."""
        if self.sampler is None:
            return None
        return self.sampler.statistics()

    def task_summary(self, task_id: str) -> dict[str, Any]:
        """Return accuracy summary for a single registered task."""
        if self.sampler is None:
            raise RuntimeError("No sampler configured.")
        return self.sampler.task_summary(task_id)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _warn_outside_zone(self, task_ids: Sequence[str]) -> None:
        """Log warnings for tasks currently outside the learning zone."""
        for tid in task_ids:
            try:
                if not self.sampler.in_learning_zone(tid):  # type: ignore[union-attr]
                    logger.warning(
                        "CurriculumRLVRTrainer: task %r is outside the learning zone. "
                        "Current thresholds: easy=%.2f, hard=%.2f. "
                        "Consider retraining the difficulty EMA or adjusting thresholds.",
                        tid,
                        self.sampler.config.easy_threshold,  # type: ignore[union-attr]
                        self.sampler.config.hard_threshold,  # type: ignore[union-attr]
                    )
            except KeyError:
                logger.debug("_warn_outside_zone: task %r not registered.", tid)
