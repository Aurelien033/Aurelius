"""
tests/training/test_grpo_trainer_with_curriculum.py

Tests:
  1.  test_default_config_values
  2.  test_init_with_rlvr_trainer_no_models_required
  3.  test_init_without_trainer_or_models_raises
  4.  test_init_stores_sampler_and_trainer
  5.  test_sample_batch_raises_without_sampler
  6.  test_sample_batch_returns_task_ids
  7.  test_train_step_delegates_to_rlvr_trainer
  8.  test_train_step_updates_sampler_on_success
  9.  test_train_step_skips_unregistered_tasks
  10. test_verbose_includes_sampler_stats
  11. test_sampler_statistics_returns_none_without_sampler
  12. test_enforce_learning_zone_warns_on_hard_task
  13. test_policy_and_ref_model_stored_via_trainer
  14. test_config_exposes_nested_grpo_and_curriculum_configs
"""

from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest

from src.alignment.rlvr import RLVRConfig, RLVRTrainer
from src.training.curriculum_rl import CurriculumRLConfig, CurriculumRLSampler
from src.training.grpo_trainer_with_curriculum import (
    CurriculumRLVRConfig,
    CurriculumRLVRTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_models():
    policy = MagicMock()
    ref = MagicMock()
    optimizer = MagicMock()
    return policy, ref, optimizer


def _make_sampler_with_tasks(n_tasks: int = 5) -> CurriculumRLSampler:
    sampler = CurriculumRLSampler(CurriculumRLConfig())
    for i in range(n_tasks):
        sampler.register_task(f"task_{i}", difficulty=0.5)
    return sampler


def _make_rlvr_trainer():
    policy, ref, opt = _make_dummy_models()
    cfg = RLVRConfig(n_samples=2, max_new_tokens=8)
    t = RLVRTrainer(
        policy_model=policy,
        ref_model=ref,
        reward_fn=lambda p, c, a="": 0.5,
        config=cfg,
        optimizer=opt,
    )
    return t


# ---------------------------------------------------------------------------
# 1-4. Construction
# ---------------------------------------------------------------------------


def test_default_config_values():
    cfg = CurriculumRLVRConfig()
    assert cfg.verbose is False
    assert cfg.rlvr is None
    assert cfg.curriculum is None


def test_init_with_rlvr_trainer_no_models_required():
    # Pass a pre-built trainer — no model plumbing needed
    trainer = _make_rlvr_trainer()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=_make_sampler_with_tasks())
    assert wrapped.rlvr_trainer is trainer
    assert wrapped.sampler is not None


def test_init_without_trainer_or_models_raises():
    with pytest.raises(ValueError, match="Either rlvr_trainer must be provided"):
        CurriculumRLVRTrainer(sampler=_make_sampler_with_tasks())


def test_init_stores_sampler_and_trainer():
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler)
    assert wrapped.rlvr_trainer is trainer
    assert wrapped.sampler is sampler
    assert wrapped.config.verbose is False


# ---------------------------------------------------------------------------
# 5-6. Batch sampling
# ---------------------------------------------------------------------------


def test_sample_batch_raises_without_sampler():
    trainer = _make_rlvr_trainer()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer)
    with pytest.raises(RuntimeError, match="requires a sampler"):
        wrapped.sample_batch(5)


def test_sample_batch_returns_task_ids():
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks(n_tasks=10)
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler)
    ids = wrapped.sample_batch(4, rng_seed=0)
    assert len(ids) == 4
    assert all(isinstance(t, str) for t in ids)


# ---------------------------------------------------------------------------
# 7-9. train_step integration
# ---------------------------------------------------------------------------


def test_train_step_delegates_to_rlvr_trainer():
    """train_step returns whatever RLVRTrainer.train_step returns."""
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler)

    prompt_ids = torch.zeros((1, 10), dtype=torch.long)
    # Patch RLVRTrainer.train_step so it doesn't run real model forward
    trainer.train_step = MagicMock(return_value={"loss": 1.2, "mean_reward": 0.8, "n_samples": 2})

    result = wrapped.train_step(
        task_ids=["task_0"],
        prompt_ids=prompt_ids,
        prompt_text="solve this",
        answer="42",
    )
    assert result["loss"] == 1.2
    assert result["mean_reward"] == 0.8
    trainer.train_step.assert_called_once_with(
        prompt_ids=prompt_ids,
        prompt_text="solve this",
        ground_truth="42",
    )


def test_train_step_updates_sampler_on_success():
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler)
    prompt_ids = torch.zeros((1, 5), dtype=torch.long)

    trainer.train_step = MagicMock(return_value={"loss": 0.5, "mean_reward": 1.0, "n_samples": 1})

    acc_before = sampler.task_summary("task_0")["accuracy"]
    wrapped.train_step(
        task_ids=["task_0"],
        prompt_ids=prompt_ids,
        prompt_text="p",
        answer="a",
    )
    acc_after = sampler.task_summary("task_0")["accuracy"]
    assert acc_after >= acc_before


def test_train_step_skips_unregistered_tasks():
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks(n_tasks=3)  # only task_0..task_2 registered
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler)
    prompt_ids = torch.zeros((1, 3), dtype=torch.long)

    trainer.train_step = MagicMock(return_value={"loss": 0.1, "mean_reward": 0.0, "n_samples": 1})

    # ghost_task not registered — should not raise
    result = wrapped.train_step(
        task_ids=["task_0", "ghost_task"],
        prompt_ids=prompt_ids,
        prompt_text="p",
        answer="a",
    )
    assert result["loss"] == 0.1


# ---------------------------------------------------------------------------
# 10-12. Verbose mode, introspection, zone enforcement
# ---------------------------------------------------------------------------


def test_verbose_includes_sampler_stats():
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks()
    CONFIG = CurriculumRLVRConfig(verbose=True)
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler, config=CONFIG)
    prompt_ids = torch.zeros((1, 3), dtype=torch.long)

    trainer.train_step = MagicMock(return_value={"loss": 0.5, "mean_reward": 1.0, "n_samples": 1})
    result = wrapped.train_step(
        task_ids=["task_0"], prompt_ids=prompt_ids, prompt_text="p", answer="a"
    )
    assert "sampler_stats" in result
    assert isinstance(result["sampler_stats"], dict)


def test_sampler_statistics_returns_none_without_sampler():
    trainer = _make_rlvr_trainer()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer)
    assert wrapped.sampler_statistics() is None


def test_enforce_learning_zone_warns_on_hard_task():
    trainer = _make_rlvr_trainer()
    sampler = _make_sampler_with_tasks()
    sampler.register_task("hard_task", difficulty=0.01)  # near-zero accuracy
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, sampler=sampler)
    prompt_ids = torch.zeros((1, 3), dtype=torch.long)

    trainer.train_step = MagicMock(return_value={"loss": 0.1, "mean_reward": 1.0, "n_samples": 1})
    # Should not raise — just log a warning
    result = wrapped.train_step(
        task_ids=["hard_task"], prompt_ids=prompt_ids, prompt_text="p", answer="a"
    )
    assert result is not None


# ---------------------------------------------------------------------------
# 13-14. Introspection
# ---------------------------------------------------------------------------


def test_policy_and_ref_model_stored_via_trainer():
    trainer = _make_rlvr_trainer()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer)
    assert wrapped.rlvr_trainer.policy_model is not None
    assert wrapped.rlvr_trainer.ref_model is not None


def test_config_exposes_nested_grpo_and_curriculum_configs():
    grpo_cfg = MagicMock()
    curr_cfg = CurriculumRLConfig()
    config = CurriculumRLVRConfig(rlvr=grpo_cfg, curriculum=curr_cfg)
    trainer = _make_rlvr_trainer()
    wrapped = CurriculumRLVRTrainer(rlvr_trainer=trainer, config=config)
    assert wrapped.config.rlvr is grpo_cfg
    assert wrapped.config.curriculum is curr_cfg
