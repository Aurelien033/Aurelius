"""Tests for standalone AMC surprise prediction head."""

from __future__ import annotations

import torch

from src.model.amc_surprise import (
    SurpriseHead,
    SurpriseHeadConfig,
    SurprisePretrainer,
    extract_surprise_heads,
    sync_surprise_heads,
)
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def test_surprise_head_output_range() -> None:
    head = SurpriseHead(SurpriseHeadConfig(d_model=64))
    x = torch.randn(8, 64)
    out = head(x)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out <= 1).all()


def test_surprise_predict_threshold() -> None:
    head = SurpriseHead(SurpriseHeadConfig(d_model=64))
    x = torch.randn(8, 64)
    pred = head.predict(x, threshold=0.5)
    assert pred.dtype == torch.bool


def test_focal_bce_handles_class_imbalance() -> None:
    pretrainer = SurprisePretrainer(
        SurpriseHead(SurpriseHeadConfig(d_model=64)),
        focal_gamma=2.0,
    )
    hidden = torch.randn(100, 64)
    labels = torch.zeros(100)
    labels[:10] = 1.0
    metrics = pretrainer.train_step(hidden, labels)
    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1


def test_pretraining_improves_accuracy() -> None:
    torch.manual_seed(42)
    head = SurpriseHead(SurpriseHeadConfig(d_model=64))
    pretrainer = SurprisePretrainer(head, lr=1e-3)
    hidden = torch.randn(200, 64)
    labels = torch.zeros(200)
    labels[:20] = 1.0
    initial = pretrainer.evaluate(hidden, labels)["accuracy"]
    for _ in range(200):
        pretrainer.train_step(hidden, labels)
    final = pretrainer.evaluate(hidden, labels)["accuracy"]
    assert final > initial + 0.1, f"pretraining did not improve accuracy: {initial} -> {final}"


def test_sync_surprise_heads_copies_weights() -> None:
    heads = [SurpriseHead(SurpriseHeadConfig(d_model=64)) for _ in range(3)]
    target = SurpriseHead(SurpriseHeadConfig(d_model=64))
    with torch.no_grad():
        target.net[0].weight.fill_(0.123)
    sync_surprise_heads(heads, target=target)
    for head in heads:
        assert torch.allclose(head.net[0].weight, torch.full_like(head.net[0].weight, 0.123))


def test_extract_surprise_heads_from_transformer() -> None:
    cfg = AMCTransformerConfig(
        vocab_size=100,
        d_model=64,
        n_layers=4,
        n_heads=4,
        ssm_d_state=32,
        ssm_headdim=32,
        ssm_expand=2,
    )
    model = AMCTransformer(cfg)
    heads = extract_surprise_heads(model)
    assert len(heads) == model.ssm_layer_count


def test_surprise_head_deterministic_in_eval() -> None:
    head = SurpriseHead(SurpriseHeadConfig(d_model=64, dropout=0.5))
    head.eval()
    x = torch.randn(4, 64)
    a = head(x).tolist()
    b = head(x).tolist()
    assert a == b


def test_temperature_scaling() -> None:
    cfg = SurpriseHeadConfig(d_model=64, temperature=0.1)
    head = SurpriseHead(cfg)
    cfg2 = SurpriseHeadConfig(d_model=64, temperature=10.0)
    head2 = SurpriseHead(cfg2)
    head2.load_state_dict(head.state_dict())
    x = torch.randn(16, 64)
    out_sharp = head(x)
    out_soft = head2(x)
    sharp_spread = (out_sharp - 0.5).abs().mean()
    soft_spread = (out_soft - 0.5).abs().mean()
    assert sharp_spread > soft_spread
