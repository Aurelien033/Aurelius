"""Tests for AMCTransformer hybrid attention + SSM model."""

from __future__ import annotations

import torch

from src.memory.amc_tier2 import AMCTier2Config, AMCTier2Hook
from src.model.amc_ssm_layer import AMCSSMLayer
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def _cfg(**overrides: object) -> AMCTransformerConfig:
    base: dict[str, object] = dict(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        ssm_d_state=32,
        ssm_headdim=32,
        ssm_expand=2,
        max_seq_len=128,
    )
    base.update(overrides)
    return AMCTransformerConfig(**base)


def test_initialization_counts() -> None:
    model = AMCTransformer(_cfg(n_layers=6))
    assert model.ssm_layer_count + model.attention_layer_count == 6
    assert model.ssm_layer_count == 3
    assert model.attention_layer_count == 3


def test_ssm_layer_indices_default() -> None:
    cfg = _cfg(n_layers=8)
    assert cfg.get_ssm_layer_indices() == {1, 3, 5, 7}


def test_ssm_layer_indices_custom() -> None:
    cfg = _cfg(n_layers=6, ssm_layers_at=(0, 2))
    assert cfg.get_ssm_layer_indices() == {0, 2}
    model = AMCTransformer(cfg)
    assert model.ssm_layer_count == 2
    assert model.attention_layer_count == 4


def test_forward_returns_logits_correct_shape() -> None:
    model = AMCTransformer(_cfg())
    x = torch.randint(0, 1000, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 1000)


def test_forward_returns_memory_blocks_when_requested() -> None:
    model = AMCTransformer(_cfg())
    out = model(torch.randint(0, 1000, (2, 8)), return_memory=True)
    assert len(out.memory_blocks) == model.ssm_layer_count
    assert out.surprise_scores is not None
    assert out.surprise_scores.shape[0] == model.ssm_layer_count


def test_forward_no_memory_when_use_amc_false() -> None:
    model = AMCTransformer(_cfg())
    out = model(torch.randint(0, 1000, (2, 8)), use_amc=False, return_memory=True)
    assert out.memory_blocks == []
    assert out.surprise_scores is None
    assert out.gate_outputs == []


def test_memory_blocks_all_tier1() -> None:
    model = AMCTransformer(_cfg())
    out = model(torch.randint(0, 1000, (1, 4)), return_memory=True)
    assert all(block.tier == 1 for block in out.memory_blocks)


def test_promotion_loss_is_scalar() -> None:
    model = AMCTransformer(_cfg())
    model.train()
    out = model(torch.randint(0, 1000, (2, 8)), use_amc=True)
    assert out.promotion_loss is not None
    assert out.promotion_loss.ndim == 0


def test_promotion_loss_zero_when_not_training() -> None:
    model = AMCTransformer(_cfg())
    model.eval()
    out = model(torch.randint(0, 1000, (2, 8)), use_amc=True)
    assert out.promotion_loss is None


def test_gradient_flows_to_all_components() -> None:
    model = AMCTransformer(_cfg())
    model.train()
    out = model(torch.randint(0, 1000, (2, 8)), use_amc=True)
    loss = out.logits.sum()
    if out.promotion_loss is not None:
        loss = loss + out.promotion_loss
    loss.backward()

    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.embed.weight.grad.norm().item() > 0.0

    for layer_idx, layer in enumerate(model.layers):
        if layer_idx in model.ssm_layer_indices:
            assert isinstance(layer, AMCSSMLayer)
            assert layer.decay_net.weight.grad is not None
        else:
            assert layer.attn.q_proj.weight.grad is not None

    for gate in model.promotion_gates.values():
        assert gate.gate_net[0].weight.grad is not None


def test_reset_amc_state_clears_all_ssm_layers() -> None:
    model = AMCTransformer(_cfg())
    x = torch.randint(0, 1000, (1, 4))
    model(x, step=0)
    for layer in model.layers:
        if hasattr(layer, "get_state"):
            assert layer.get_state() is not None
    model.reset_amc_state()
    for layer in model.layers:
        if hasattr(layer, "get_state"):
            assert layer.get_state() is None


def test_tie_embeddings() -> None:
    model = AMCTransformer(_cfg(tie_embeddings=True))
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_wire_amc_hooks_attaches_hooks() -> None:
    model = AMCTransformer(_cfg())
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
    model.wire_amc_hooks(tier2=hook)
    assert model._tier2_hook is hook
