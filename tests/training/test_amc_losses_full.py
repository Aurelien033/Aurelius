"""Tests for AMC memory-aware training losses (T15)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.amc_losses import (
    memory_consistency_loss,
    promotion_reward_loss,
    surprise_prediction_loss,
    total_amc_loss,
)


def test_surprise_loss_perfect_prediction() -> None:
    pred = torch.tensor([[0.9, 0.1, 0.8]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    loss = surprise_prediction_loss(pred, labels, weight_by_importance=False)
    assert loss.item() < 0.3


def test_surprise_loss_bad_prediction() -> None:
    pred = torch.tensor([[0.1, 0.9, 0.2]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    loss = surprise_prediction_loss(pred, labels, weight_by_importance=False)
    assert loss.item() > 0.5


def test_surprise_loss_weights_positives() -> None:
    pred = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([[1.0, 0.0]])
    weighted = surprise_prediction_loss(pred, labels, weight_by_importance=True)
    unweighted = surprise_prediction_loss(pred, labels, weight_by_importance=False)
    assert weighted.item() > unweighted.item()


def test_surprise_loss_3d_input() -> None:
    pred = torch.rand(3, 2, 8)
    labels = torch.rand(3, 2, 8)
    loss = surprise_prediction_loss(pred, labels)
    assert loss.shape == ()


def test_surprise_loss_handles_all_zeros_labels() -> None:
    pred = torch.tensor([[0.2, 0.3, 0.1]])
    labels = torch.zeros_like(pred)
    loss = surprise_prediction_loss(pred, labels, weight_by_importance=False)
    assert torch.isfinite(loss)


def test_consistency_loss_zero_when_no_retrievals() -> None:
    hidden = torch.randn(2, 8, 64)
    loss = memory_consistency_loss(hidden, None)
    assert loss.item() == 0.0


def test_consistency_loss_matches_when_aligned() -> None:
    hidden = torch.randn(2, 8, 64)
    pooled = hidden.mean(dim=1)
    retrieved = pooled.unsqueeze(1)
    loss = memory_consistency_loss(hidden, retrieved)
    assert loss.item() < 0.1


def test_consistency_loss_diverges_when_misaligned() -> None:
    hidden = torch.randn(2, 8, 64)
    retrieved = torch.randn(2, 3, 64)
    loss = memory_consistency_loss(hidden, retrieved)
    assert loss.item() > 0.3


def test_consistency_loss_with_projection() -> None:
    proj = nn.Linear(64, 32)
    hidden = torch.randn(2, 8, 64)
    retrieved = torch.randn(2, 2, 32)
    loss = memory_consistency_loss(hidden, retrieved, embedding_proj=proj)
    assert loss.item() >= 0.0


def test_promotion_reward_loss_reinforces_good_decisions() -> None:
    store_soft = torch.tensor([0.9, 0.9, 0.1, 0.1], requires_grad=True)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = promotion_reward_loss(store_soft, rewards)
    loss.backward()
    assert torch.isfinite(loss)
    assert store_soft.grad is not None


def test_promotion_loss_zero_when_no_rewards() -> None:
    store_soft = torch.tensor([0.5, 0.5])
    rewards = torch.tensor([0.0, 0.0])
    loss = promotion_reward_loss(store_soft, rewards)
    assert loss.item() == 0.0


def test_total_loss_weights_matter() -> None:
    sft = torch.tensor(1.0)
    sup = torch.tensor(2.0)
    con = torch.tensor(3.0)
    pro = torch.tensor(4.0)
    total, _metrics = total_amc_loss(sft, sup, con, pro)
    expected = 0.7 * 1.0 + 0.15 * 2.0 + 0.10 * 3.0 + 0.05 * 4.0
    assert abs(total.item() - expected) < 1e-5


def test_total_loss_returns_metrics() -> None:
    sft = torch.tensor(0.5)
    sup = torch.tensor(0.3)
    con = torch.tensor(0.2)
    pro = torch.tensor(0.1)
    _, metrics = total_amc_loss(sft, sup, con, pro)
    assert "loss_weights" in metrics
    assert abs(sum(metrics["loss_weights"].values()) - 1.0) < 1e-6


def test_losses_finite_check() -> None:
    pred = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([[1.0, 0.0]])
    assert torch.isfinite(surprise_prediction_loss(pred, labels))

    hidden = torch.randn(2, 4, 8)
    retrieved = torch.randn(2, 1, 8)
    assert torch.isfinite(memory_consistency_loss(hidden, retrieved))

    store = torch.tensor([0.6, 0.7], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0])
    promo = promotion_reward_loss(store, rewards)
    assert torch.isfinite(promo)
    promo.backward()
    assert torch.isfinite(store.grad).all()
