"""Tests for Tier-1 → Tier-2 promotion gate."""

from __future__ import annotations

import torch

from src.memory.amc_tier2 import AMCTier2Config, AMCTier2Hook
from src.model.amc_promotion import PromotionGate, tier1_to_tier2_promotion
from src.model.amc_ssm_layer import AMCSSMConfig, AMCSSMLayer


def test_initialization() -> None:
    gate = PromotionGate(d_model=64, temperature=0.5)
    assert gate.temperature == 0.5
    assert len(gate.gate_net) == 3


def test_forward_returns_hard_and_soft() -> None:
    gate = PromotionGate(d_model=64)
    gate.train()
    hidden = torch.randn(3, 64)
    surprise = torch.rand(3)
    hard, soft = gate(hidden, surprise)
    assert hard.shape == (3,)
    assert soft.shape == (3,)


def test_hard_is_binary() -> None:
    gate = PromotionGate(d_model=64)
    gate.train()
    hard, _ = gate(torch.randn(8, 64), torch.rand(8))
    assert torch.all((hard == 0.0) | (hard == 1.0))


def test_soft_is_in_unit_interval() -> None:
    gate = PromotionGate(d_model=64)
    gate.train()
    _, soft = gate(torch.randn(8, 64), torch.rand(8))
    assert soft.min() >= 0.0
    assert soft.max() <= 1.0


def test_straight_through_gradient() -> None:
    gate = PromotionGate(d_model=64)
    gate.train()
    hidden = torch.randn(4, 64, requires_grad=True)
    surprise = torch.rand(4, requires_grad=True)
    hard, soft = gate(hidden, surprise)
    st = gate.straight_through_store(hard, soft)
    st.sum().backward()
    assert gate.gate_net[0].weight.grad is not None
    assert gate.gate_net[0].weight.grad.norm().item() > 0.0


def test_eval_mode_is_deterministic() -> None:
    gate = PromotionGate(d_model=64)
    gate.eval()
    hidden = torch.randn(4, 64)
    surprise = torch.tensor([0.2, 0.5, 0.8, 0.1])
    hard_a, soft_a = gate(hidden, surprise)
    hard_b, soft_b = gate(hidden, surprise)
    assert torch.equal(hard_a, hard_b)
    assert torch.equal(soft_a, soft_b)


def test_promote_loss_decreases_with_training() -> None:
    gate = PromotionGate(d_model=64)
    gate.train()
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
    hidden = torch.randn(8, 64)
    surprise = torch.rand(8)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    initial = None
    for _ in range(10):
        opt.zero_grad()
        _, soft = gate(hidden, surprise)
        loss = gate.promote_loss(soft, target)
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < initial


def test_tier1_to_tier2_promotion_integration() -> None:
    layer = AMCSSMLayer(AMCSSMConfig(d_model=64, d_state=32, headdim=32), layer_index=0)
    gate = PromotionGate(d_model=64)
    gate.eval()
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
    out = layer(torch.randn(2, 4, 64), step=0)
    count, mean_soft, promoted = tier1_to_tier2_promotion(
        out,
        gate,
        hook,
        promote_threshold=0.0,
    )
    assert mean_soft.ndim == 0
    assert count == len(promoted)
    assert count >= 0


def test_promotion_respects_threshold() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
    layer = AMCSSMLayer(AMCSSMConfig(d_model=64, d_state=32, headdim=32), layer_index=1)
    gate = PromotionGate(d_model=64)
    gate.eval()
    out = layer(torch.randn(2, 4, 64), step=0)
    count_all, _, _ = tier1_to_tier2_promotion(
        out, gate, hook, promote_threshold=-0.1
    )
    count_none, _, _ = tier1_to_tier2_promotion(
        out, gate, hook, promote_threshold=1.1
    )
    assert count_all >= count_none


def test_no_promotion_when_below_threshold() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
    layer = AMCSSMLayer(AMCSSMConfig(d_model=64, d_state=32, headdim=32), layer_index=0)
    gate = PromotionGate(d_model=64)
    gate.eval()
    out = layer(torch.randn(2, 4, 64), step=0)
    count, _, promoted = tier1_to_tier2_promotion(
        out, gate, hook, promote_threshold=1.1
    )
    assert count == 0
    assert promoted == []


def test_temperature_effect() -> None:
    """Higher temperature yields softer (less peaked) store probabilities."""
    gate = PromotionGate(d_model=64, temperature=2.0)
    gate.train()
    with torch.no_grad():
        gate.gate_net[2].weight.zero_()
        gate.gate_net[2].bias.zero_()
    hidden = torch.zeros(4, 64)
    surprise = torch.zeros(4)

    gate.temperature = 5.0
    _, soft_high = gate(hidden, surprise)
    gate.temperature = 0.05
    _, soft_low = gate(hidden, surprise)

    assert (soft_high - 0.5).abs().mean() < (soft_low - 0.5).abs().mean()
