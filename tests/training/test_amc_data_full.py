"""Tests for AMC training data pipeline (T16)."""

from __future__ import annotations

import torch

from src.training.amc_data import (
    AMCDataCollator,
    annotate_importance,
    build_amc_training_batch,
    retrieval_ground_truth_from_prior_session,
    session_boundary_flags,
)


def _fake_tokenizer(text: str) -> list[int]:
    return list(range(len(text.split())))


def test_annotate_corrections_are_high() -> None:
    msgs = [{"role": "user", "content": "Actually, that's wrong — the correct way is X."}]
    scores = annotate_importance(msgs)
    assert scores[0] >= 0.8


def test_annotate_greetings_are_low() -> None:
    msgs = [{"role": "user", "content": "Hi there!"}]
    scores = annotate_importance(msgs)
    assert scores[0] < 0.3


def test_annotate_tool_messages() -> None:
    msgs = [{"role": "tool", "content": "Result from API call: OK"}]
    scores = annotate_importance(msgs)
    assert 0.4 < scores[0] < 0.7


def test_batch_shape_matches_max_seq_len() -> None:
    transcript = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    batch = build_amc_training_batch(transcript, _fake_tokenizer, max_seq_len=32)
    assert batch.input_ids.shape == (1, 32)
    assert batch.importance_labels.shape == (1, 32)


def test_batch_target_is_shifted() -> None:
    transcript = [{"role": "user", "content": "test message"}]
    batch = build_amc_training_batch(transcript, _fake_tokenizer, max_seq_len=8)
    assert torch.allclose(batch.target_ids[:, :-1], batch.input_ids[:, 1:])


def test_collator_batches_multiple() -> None:
    transcripts = [[{"role": "user", "content": f"msg {i}"}] for i in range(4)]
    batches = [
        build_amc_training_batch(transcript, _fake_tokenizer, max_seq_len=16)
        for transcript in transcripts
    ]
    collator = AMCDataCollator()
    batched = collator(batches)
    assert batched.input_ids.shape[0] == 4


def test_session_boundary_detection() -> None:
    messages = [
        {"role": "user", "content": "a", "session_id": "s1"},
        {"role": "assistant", "content": "b", "session_id": "s1"},
        {"role": "user", "content": "c", "session_id": "s2"},
    ]
    flags = session_boundary_flags(messages)
    assert flags == [False, False, True]


def test_retrieval_ground_truth_from_prior_session() -> None:
    prior = [
        {"role": "user", "content": "Remember to always use dark mode."},
        {"role": "user", "content": "Hi"},
    ]
    ground_truth = retrieval_ground_truth_from_prior_session(prior)
    assert len(ground_truth) == 1
    assert "dark mode" in ground_truth[0]
