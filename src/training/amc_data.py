"""AMC training data: tokenize transcripts and annotate importance.

Produces :class:`AMCTrainBatch` with input/target ids, per-token importance labels,
session scope, and optional retrieved memory embeddings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AMCTrainBatch:
    """One training batch for the AMC trainer."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    importance_labels: torch.Tensor
    session_id: str | list[str]
    step: int | list[int]
    retrieved_embeddings: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> AMCTrainBatch:
        return AMCTrainBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            importance_labels=self.importance_labels.to(device),
            session_id=self.session_id,
            step=self.step,
            retrieved_embeddings=(
                self.retrieved_embeddings.to(device)
                if self.retrieved_embeddings is not None
                else None
            ),
        )


_CORRECTION_MARKERS = frozenset(
    {
        "correct",
        "actually",
        "actually,",
        "no,",
        "wrong",
        "fix",
        "instead",
        "rather",
        "than",
        "don't",
        "shouldn't",
    }
)
_FACT_MARKERS = frozenset(
    {
        "remember",
        "store",
        "always",
        "never",
        "prefer",
        "use",
        "configuration",
        "architecture",
        "policy",
    }
)
_GREETING_MARKERS = frozenset({"hi", "hello", "hey", "thanks", "ok", "okay"})


def annotate_importance(
    messages: list[dict[str, Any]],
    *,
    correction_weight: float = 0.9,
    fact_weight: float = 0.7,
    tool_weight: float = 0.5,
    greeting_weight: float = 0.1,
    filler_weight: float = 0.05,
) -> list[float]:
    """Heuristic importance labels for each message in a transcript."""
    scores: list[float] = []
    for msg in messages:
        content = str(msg.get("content", "")).lower()
        role = str(msg.get("role", "user"))
        words = set(content.split())

        if words & _CORRECTION_MARKERS:
            scores.append(correction_weight)
        elif words & _FACT_MARKERS:
            scores.append(fact_weight)
        elif role == "tool":
            scores.append(tool_weight)
        elif words & _GREETING_MARKERS and len(content.split()) < 8:
            scores.append(greeting_weight)
        else:
            scores.append(filler_weight)
    return scores


def session_boundary_flags(messages: list[dict[str, Any]]) -> list[bool]:
    """Return True when a message begins a new session."""
    flags: list[bool] = []
    previous_session: str | None = None
    for msg in messages:
        session = str(msg.get("session_id", "default"))
        flags.append(previous_session is not None and session != previous_session)
        previous_session = session
    return flags


def retrieval_ground_truth_from_prior_session(
    prior_messages: list[dict[str, Any]],
    *,
    min_importance: float = 0.7,
) -> list[str]:
    """High-importance prior-session contents used as retrieval supervision."""
    ground_truth: list[str] = []
    for msg in prior_messages:
        if annotate_importance([msg])[0] < min_importance:
            continue
        content = str(msg.get("content", "")).strip()
        if content:
            ground_truth.append(content)
    return ground_truth


def build_amc_training_batch(
    transcript: list[dict[str, Any]],
    tokenizer: Callable[[str], list[int]],
    *,
    max_seq_len: int = 2048,
    pad_token_id: int = 0,
    session_id: str = "default",
    base_step: int = 0,
) -> AMCTrainBatch:
    """Build a training batch from a multi-turn transcript."""
    text_parts: list[str] = []
    importance_per_msg = annotate_importance(transcript)
    token_importance: list[float] = []

    for msg, importance in zip(transcript, importance_per_msg, strict=True):
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        full = f"<|{role}|>\n{content}\n\n"
        tokens = tokenizer(full)
        text_parts.append(full)
        token_importance.extend([importance] * len(tokens))

    full_text = "".join(text_parts)
    input_ids = tokenizer(full_text)

    if len(token_importance) != len(input_ids):
        min_len = min(len(token_importance), len(input_ids))
        input_ids = input_ids[:min_len]
        token_importance = token_importance[:min_len]

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        token_importance = token_importance[:max_seq_len]

    pad_len = max_seq_len - len(input_ids)
    input_ids = input_ids + [pad_token_id] * pad_len
    token_importance = token_importance + [0.0] * pad_len

    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    target_tensor = torch.cat([input_tensor[1:], torch.tensor([pad_token_id])])
    importance_tensor = torch.tensor(token_importance, dtype=torch.float32)

    return AMCTrainBatch(
        input_ids=input_tensor.unsqueeze(0),
        target_ids=target_tensor.unsqueeze(0),
        importance_labels=importance_tensor.unsqueeze(0),
        session_id=session_id,
        step=base_step,
    )


class AMCDataCollator:
    """Collate :class:`AMCTrainBatch` instances into a batched training input."""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[AMCTrainBatch]) -> AMCTrainBatch:
        inputs = torch.cat([item.input_ids for item in batch], dim=0)
        targets = torch.cat([item.target_ids for item in batch], dim=0)
        importances = torch.cat([item.importance_labels for item in batch], dim=0)
        session_ids = [item.session_id for item in batch]
        steps = [item.step for item in batch]

        max_k = max(
            (
                item.retrieved_embeddings.shape[1]
                if item.retrieved_embeddings is not None
                else 0
            )
            for item in batch
        )
        padded_retrieved: torch.Tensor | None
        if max_k > 0:
            d_embed = next(
                item.retrieved_embeddings.shape[-1]
                for item in batch
                if item.retrieved_embeddings is not None
            )
            padded_retrieved = torch.zeros(len(batch), max_k, d_embed)
            for index, item in enumerate(batch):
                if item.retrieved_embeddings is not None:
                    count = item.retrieved_embeddings.shape[1]
                    padded_retrieved[index, :count] = item.retrieved_embeddings
        else:
            padded_retrieved = None

        return AMCTrainBatch(
            input_ids=inputs,
            target_ids=targets,
            importance_labels=importances,
            session_id=session_ids,
            step=steps,
            retrieved_embeddings=padded_retrieved,
        )


__all__ = [
    "AMCDataCollator",
    "AMCTrainBatch",
    "annotate_importance",
    "build_amc_training_batch",
    "retrieval_ground_truth_from_prior_session",
    "session_boundary_flags",
]
