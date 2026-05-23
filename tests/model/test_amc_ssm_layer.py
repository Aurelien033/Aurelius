"""Tests for AMC Tier-1 SSM working memory layer."""

from __future__ import annotations

import torch

from src.model.amc_ssm_layer import AMCForwardOutput, AMCSSMConfig, AMCSSMLayer


def _small_config(**overrides: object) -> AMCSSMConfig:
    defaults: dict[str, object] = dict(
        d_model=64,
        d_state=32,
        d_conv=4,
        expand=2,
        headdim=32,
    )
    defaults.update(overrides)
    return AMCSSMConfig(**defaults)


def test_initialization() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=2)
    assert layer.layer_index == 2
    assert layer.ssm is not None
    assert layer.get_state() is None


def test_forward_returns_amc_output() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    x = torch.randn(2, 8, 64)
    out = layer(x, step=0)
    assert isinstance(out, AMCForwardOutput)
    assert out.hidden.shape == (2, 8, 64)


def test_surprise_head_outputs_in_unit_interval() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    out = layer(torch.randn(2, 4, 64), step=0)
    assert out.surprise_scores.min() >= 0.0
    assert out.surprise_scores.max() <= 1.0


def test_surprise_head_no_gradient() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x, step=0)
    out.hidden.sum().backward()
    for param in layer.surprise_head.parameters():
        assert param.grad is None


def test_gates_have_gradient() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x, step=0)
    out.hidden.sum().backward()
    assert layer.gates.decay.net[0].weight.grad is not None
    assert layer.gates.decay.net[0].weight.grad.norm().item() > 0.0


def test_memory_block_tier_is_1() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    out = layer(torch.randn(1, 4, 64), step=0)
    assert out.memory_block.tier == 1


def test_memory_block_surprise_matches_head_output() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    out = layer(torch.randn(2, 6, 64), step=0)
    expected = float(out.surprise_scores[:, -1].mean().item())
    assert abs(out.memory_block.surprise_score - expected) < 1e-6


def test_amc_tensor_state_correct_shape() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=3)
    out = layer(torch.randn(2, 5, 64), step=10)
    ts = out.tensor_state
    assert ts.layer_index == 3
    assert ts.token_count == 15
    assert len(ts.kvs) == 1
    state = ts.kvs[0]
    assert state.shape == (2, 4, 32, 32)


def test_state_continuity_across_steps() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    x1 = torch.randn(2, 4, 64)
    x2 = torch.randn(2, 4, 64)

    out1, state1 = layer(x1, step=0, return_state=True)
    out2, state2 = layer(
        x2,
        step=4,
        prev_state=state1["ssm_state"],
        return_state=True,
    )

    assert out1.hidden.shape == (2, 4, 64)
    assert out2.hidden.shape == (2, 4, 64)
    assert state2["ssm_state"].shape == (2, 4, 32, 32)
    assert layer.get_state() is not None
    assert tuple(layer.get_state().shape) == (2, 4, 32, 32)


def test_reset_clears_state() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=0)
    layer(torch.randn(1, 4, 64), step=0)
    assert layer.get_state() is not None
    layer.reset_state()
    assert layer.get_state() is None


def test_layer_index_stamped_correctly() -> None:
    layer = AMCSSMLayer(_small_config(), layer_index=7)
    out = layer(torch.randn(1, 2, 64), step=1)
    assert out.tensor_state.layer_index == 7
    assert out.memory_block.metadata["layer_index"] == 7
    assert out.memory_block.block_id.startswith("amc_l7_")
