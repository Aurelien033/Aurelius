"""
tests/training/test_prm_training_data.py

Tests:
  1.  test_config_defaults
  2.  test_step_record_state_hash_changes_with_state
  3.  test_step_record_state_hash_same_for_same_state
  4.  test_labelled_trajectory_valid_with_minimum_steps
  5.  test_labelled_trajectory_invalid_too_few_steps
  6.  test_labelled_trajectory_invalid_unlabelled_steps
  7.  test_auto_label_exact_match
  8.  test_auto_label_ambiguous_no_tests
  9.  test_auto_label_error_state
  10. test_filter_steps_removes_ambiguous
  11. test_filter_steps_removes_no_progress
  12. test_filter_steps_keeps_valid_steps
  13. test_filter_steps_preserves_order
  14. test_trajectory_builder_record_step
  15. test_trajectory_builder_build_valid
  16. test_trajectory_builder_build_drops_invalid
  17. test_prm_data_writer_adds_and_flushes
  18. test_prm_data_writer_context_manager
  19. test_prm_data_writer_total_written
  20. test_prm_data_writer_jsonl_format
"""

from __future__ import annotations

import json
import os

import pytest

from src.training.prm_training_data import (
    LabelledTrajectory,
    PRMDataWriter,
    PRMTrainingDataConfig,
    StepRecord,
    TrajectoryBuilder,
    _default_label_fn,
    auto_label,
    filter_steps,
)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PRMTrainingDataConfig()
    assert cfg.checkpoint_every == 1
    assert cfg.min_steps == 2
    assert cfg.drop_no_progress is True
    assert cfg.drop_ambiguous is True
    assert cfg.shard_size == 500


# ---------------------------------------------------------------------------
# 2-3. StepRecord hashing
# ---------------------------------------------------------------------------


def test_step_record_state_hash_changes_with_state():
    s1 = StepRecord(step_idx=0, text="a", test_state={"passed": 0, "total": 1})
    s2 = StepRecord(step_idx=0, text="a", test_state={"passed": 1, "total": 1})
    assert s1.state_hash != s2.state_hash


def test_step_record_state_hash_same_for_same_state():
    s1 = StepRecord(step_idx=0, text="a", test_state={"passed": 1, "total": 2})
    s2 = StepRecord(step_idx=0, text="a", test_state={"passed": 1, "total": 2})
    assert s1.state_hash == s2.state_hash


# ---------------------------------------------------------------------------
# 4-6. LabelledTrajectory validity
# ---------------------------------------------------------------------------


def _make_trajectory(n_steps: int, labelled: bool = True):
    steps = [
        StepRecord(step_idx=i, text=f"step_{i}", test_state={"passed": i, "total": n_steps}, label=int(i > 0) if labelled else None)
        for i in range(n_steps)
    ]
    return LabelledTrajectory(
        task_id="t1", prompt="p", steps=steps, final_score=1.0
    )


def test_labelled_trajectory_valid_with_minimum_steps():
    t = _make_trajectory(n_steps=2, labelled=True)
    assert t.is_valid(min_steps=2) is True


def test_labelled_trajectory_invalid_too_few_steps():
    t = _make_trajectory(n_steps=1, labelled=True)
    assert t.is_valid(min_steps=2) is False


def test_labelled_trajectory_invalid_unlabelled_steps():
    steps = [
        StepRecord(step_idx=0, text="s", label=None),
        StepRecord(step_idx=1, text="s", label=None),
    ]
    t = LabelledTrajectory(task_id="t1", prompt="p", steps=steps)
    assert t.is_valid(min_steps=2) is False


# ---------------------------------------------------------------------------
# 7-9. auto_label
# ---------------------------------------------------------------------------


def test_auto_label_exact_match():
    steps = [StepRecord(0, "", {"passed": 3, "total": 3}), StepRecord(1, "", {"passed": 2, "total": 3})]
    auto_label(steps)
    assert steps[0].label == 1
    assert steps[1].label == 0


def test_auto_label_ambiguous_no_tests():
    steps = [StepRecord(0, "oops", {})]
    auto_label(steps)
    assert steps[0].label is None


def test_auto_label_error_state():
    steps = [StepRecord(0, "err", {"passed": 0, "total": 0, "error": "SyntaxError"})]
    auto_label(steps)
    assert steps[0].label is None


# ---------------------------------------------------------------------------
# 10-13. filter_steps
# ---------------------------------------------------------------------------


def test_filter_steps_removes_ambiguous():
    steps = [
        StepRecord(0, "a", {"passed": 1, "total": 1}, label=1),
        StepRecord(1, "b", {"passed": 0, "total": 1, "error": "err"}, label=None),
        StepRecord(2, "c", {"passed": 1, "total": 1}, label=1),
    ]
    out = filter_steps(steps, drop_ambiguous=True, drop_no_progress=False)
    assert len(out) == 2
    assert out[0].step_idx == 0
    assert out[1].step_idx == 2


def test_filter_steps_removes_no_progress():
    state = {"passed": 1, "total": 1}
    steps = [
        StepRecord(0, "a", state),
        StepRecord(1, "b", state),  # identical state
        StepRecord(2, "c", {"passed": 2, "total": 1}),  # different state
    ]
    auto_label(steps)
    out = filter_steps(steps, drop_no_progress=True, drop_ambiguous=False)
    assert len(out) == 2
    assert out[0].step_idx == 0
    assert out[1].step_idx == 2


def test_filter_steps_keeps_valid_steps():
    steps = [
        StepRecord(0, "x", {"passed": 1, "total": 2}, label=0),
        StepRecord(1, "y", {"passed": 2, "total": 2}, label=1),
    ]
    out = filter_steps(steps, drop_ambiguous=True, drop_no_progress=True)
    assert len(out) == 2


def test_filter_steps_preserves_order():
    steps = [
        StepRecord(i, f"s{i}", {"passed": i, "total": 5}, label=1)
        for i in range(5)
    ]
    out = filter_steps(steps)
    assert [s.step_idx for s in out] == list(range(5))


# ---------------------------------------------------------------------------
# 14-16. TrajectoryBuilder
# ---------------------------------------------------------------------------


def test_trajectory_builder_record_step():
    b = TrajectoryBuilder(task_id="t1", prompt="solve this")
    b.record_step(text="def foo():", test_state={"passed": 0, "total": 1})
    assert b.n_steps == 1
    assert b.steps[0].text == "def foo():"


def test_trajectory_builder_build_valid():
    b = TrajectoryBuilder(task_id="t1", prompt="solve this")
    # Use distinct test states so filter_steps does not drop the second step
    b.record_step(text="step0", test_state={"passed": 0, "total": 2})
    b.record_step(text="step1", test_state={"passed": 2, "total": 2})
    traj = b.build(final_score=1.0, auto_label_steps=True, min_steps=2)
    assert traj is not None
    assert len(traj.steps) == 2
    assert traj.final_score == 1.0


def test_trajectory_builder_build_drops_invalid():
    b = TrajectoryBuilder(task_id="t1", prompt="p")
    # Only one step — will be filtered out by min_steps=2
    b.record_step(text="only", test_state={"passed": 1, "total": 1})
    traj = b.build(min_steps=2)
    assert traj is None


# ---------------------------------------------------------------------------
# 17-20. PRMDataWriter
# ---------------------------------------------------------------------------


def test_prm_data_writer_adds_and_flushes(tmp_path):
    writer = PRMDataWriter(output_dir=tmp_path, shard_size=3)
    t = _make_trajectory(n_steps=2, labelled=True)
    writer.add(t)
    assert writer.total_written == 1

    # Shard not yet written (buffer not full)
    shards_before = list(tmp_path.glob("*.jsonl"))
    assert len(shards_before) == 0

    # Add two more to trigger flush
    writer.add(_make_trajectory(n_steps=2, labelled=True))
    writer.add(_make_trajectory(n_steps=2, labelled=True))
    assert writer.total_written == 3
    shards = list(tmp_path.glob("*.jsonl"))
    assert len(shards) == 1  # one shard flushed


def test_prm_data_writer_context_manager(tmp_path):
    t = _make_trajectory(n_steps=2, labelled=True)
    with PRMDataWriter(output_dir=tmp_path, shard_size=100) as writer:
        writer.add(t)
        assert writer.total_written == 1
    # context exit flushes remaining buffer
    shards = list(tmp_path.glob("*.jsonl"))
    assert len(shards) == 1


def test_prm_data_writer_total_written(tmp_path):
    writer = PRMDataWriter(output_dir=tmp_path, shard_size=10)
    for _ in range(7):
        writer.add(_make_trajectory(n_steps=2, labelled=True))
    assert writer.total_written == 7


def test_prm_data_writer_jsonl_format(tmp_path):
    writer = PRMDataWriter(output_dir=tmp_path, shard_size=2)
    t = _make_trajectory(n_steps=2, labelled=True)
    writer.add(t)
    writer.add(_make_trajectory(n_steps=2, labelled=True))
    writer.close()

    shard_path = tmp_path / "prm_shard_00000.jsonl"
    assert shard_path.exists()
    records = [json.loads(line) for line in shard_path.read_text().splitlines()]
    assert len(records) == 2
    for rec in records:
        assert "task_id" in rec
        assert "steps" in rec
        assert isinstance(rec["steps"], list)
        assert all("label" in s for s in rec["steps"])
