"""Tests for AMCTrainer and optimizer groups (T17)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.model.amc_transformer import AMCModelOutput
from src.training.amc_data import AMCTrainBatch
from src.training.amc_trainer import (
    AMCTrainConfig,
    AMCTrainer,
    CheckpointManager,
    build_optimizer_groups,
)


class _MockAMCModel(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.surprise_head = nn.Linear(d_model, 1)
        self.promotion_gate = nn.Linear(d_model, 1)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._tier2_hook = _Tier2Stub()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        session_id: str | list[str] | None = None,
        step: int | list[int] = 0,
        use_amc: bool = True,
        return_memory: bool = False,
    ) -> AMCModelOutput:
        _ = (session_id, step, use_amc, return_memory)
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        surprise = torch.sigmoid(self.surprise_head(hidden).squeeze(-1))
        store_soft = torch.sigmoid(self.promotion_gate(hidden[:, -1, :])).squeeze(-1)
        store_hard = (store_soft > 0.5).float()
        promotion_loss = store_soft.mean() * 0.01
        return AMCModelOutput(
            logits=logits,
            hidden_states=hidden,
            surprise_scores=surprise.unsqueeze(0),
            gate_outputs=[(store_hard, store_soft)],
            promotion_loss=promotion_loss,
        )


@dataclass
class _Tier2Stub:
    episodic: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episodic)


def _batch() -> AMCTrainBatch:
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target_ids = torch.tensor([[2, 3, 4, 0]], dtype=torch.long)
    importance = torch.tensor([[0.9, 0.1, 0.8, 0.0]], dtype=torch.float32)
    return AMCTrainBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        importance_labels=importance,
        session_id="s1",
        step=1,
    )


def test_trainer_initializes_with_config() -> None:
    model = _MockAMCModel()
    config = AMCTrainConfig(gradient_accumulation=1, max_steps=10)
    trainer = AMCTrainer(model, iter([_batch()]), None, config, Path(tempfile.mkdtemp()))
    assert trainer.config.learning_rate == 3e-4
    assert trainer.config.promotion_gate_lr == 1e-4
    assert trainer.config.surprise_head_lr == 1e-5


def test_train_step_returns_all_metrics() -> None:
    trainer = AMCTrainer(
        _MockAMCModel(),
        iter([_batch()]),
        None,
        AMCTrainConfig(gradient_accumulation=1),
        Path(tempfile.mkdtemp()),
    )
    metrics = trainer.train_step(_batch())
    for key in (
        "sft_loss",
        "surprise_loss",
        "consistency_loss",
        "promotion_loss",
        "total_loss",
        "lr",
        "promotion_rate",
        "mean_surprise",
        "tier2_entries",
    ):
        assert key in metrics


def test_loss_weights_are_used() -> None:
    trainer = AMCTrainer(
        _MockAMCModel(),
        iter([_batch()]),
        None,
        AMCTrainConfig(
            gradient_accumulation=1,
            loss_weights={"sft": 1.0, "surprise": 0.0, "consistency": 0.0, "promotion": 0.0},
        ),
        Path(tempfile.mkdtemp()),
    )
    metrics = trainer.train_step(_batch())
    assert metrics["surprise_loss"] >= 0.0
    assert metrics["total_loss"] >= metrics["sft_loss"] * 0.99


def test_optimizer_param_groups_separated() -> None:
    model = _MockAMCModel()
    optimizers, _schedulers = build_optimizer_groups(
        model,
        lr_main=3e-4,
        lr_gate=1e-4,
        lr_surprise=1e-5,
        weight_decay=0.01,
        warmup_steps=10,
        max_steps=100,
    )
    main_ids = {id(p) for p in optimizers["main"].param_groups[0]["params"]}
    gate_ids = {id(p) for p in optimizers["gate"].param_groups[0]["params"]}
    surprise_ids = {id(p) for p in optimizers["surprise"].param_groups[0]["params"]}
    assert main_ids.isdisjoint(gate_ids)
    assert main_ids.isdisjoint(surprise_ids)
    assert gate_ids.isdisjoint(surprise_ids)
    assert optimizers["main"].defaults["lr"] == 3e-4
    assert optimizers["gate"].defaults["lr"] == 1e-4
    assert optimizers["surprise"].defaults["lr"] == 1e-5


def test_gradient_accumulation() -> None:
    trainer = AMCTrainer(
        _MockAMCModel(),
        iter([_batch(), _batch()]),
        None,
        AMCTrainConfig(gradient_accumulation=2),
        Path(tempfile.mkdtemp()),
    )
    first = trainer.train_step(_batch())
    second = trainer.train_step(_batch())
    assert first["optimizer_stepped"] == 0.0
    assert second["optimizer_stepped"] == 1.0


def test_eval_mode_does_not_train() -> None:
    model = _MockAMCModel()
    trainer = AMCTrainer(
        model,
        iter([_batch()]),
        iter([_batch()]),
        AMCTrainConfig(gradient_accumulation=1),
        Path(tempfile.mkdtemp()),
    )
    before = [p.clone() for p in model.parameters()]
    trainer.evaluate()
    after = list(model.parameters())
    for param_before, param_after in zip(before, after, strict=True):
        assert torch.allclose(param_before, param_after)


def test_checkpoint_save_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _MockAMCModel()
        optimizers, schedulers = build_optimizer_groups(
            model,
            lr_main=3e-4,
            lr_gate=1e-4,
            lr_surprise=1e-5,
            weight_decay=0.01,
            warmup_steps=5,
            max_steps=50,
        )
        manager = CheckpointManager(Path(tmpdir) / "ckpts", keep_last_n=2)
        path = manager.save(
            "test",
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            step=7,
            metrics={"eval_surprise_accuracy": 0.8},
        )
        loaded = manager.load(path)
        assert loaded["step"] == 7
        assert "model" in loaded
        assert set(loaded["optimizers"].keys()) == {"main", "gate", "surprise"}
