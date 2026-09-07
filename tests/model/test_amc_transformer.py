"""Tests for AMCTransformer hybrid attention + SSM model."""

from __future__ import annotations

import torch

from src.memory.amc_tier2 import AMCTier2Config, AMCTier2Hook
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
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
            assert layer.gates.decay.net[0].weight.grad is not None
        else:
            assert layer.mla.q_down.weight.grad is not None

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


# ── DreamBank wiring (DB-03) ─────────────────────────────────────────────


def _filled_bank(**kw) -> HLMPreferenceBank:
    dim = kw.pop("bank_dim", 64)
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=dim, **kw))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim) * 2, strength=1.0))
    return bank


def test_hlm_bank_disabled_by_default_preserves_output_shape() -> None:
    model = AMCTransformer(_cfg())
    x = torch.randint(0, 1000, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 1000)
    assert out.bank_alpha is None
    assert out.bank_confidence is None
    assert out.bank_telemetry is None


def test_hlm_bank_output_fields_absent_when_disabled() -> None:
    model = AMCTransformer(_cfg())
    x = torch.randint(0, 1000, (1, 8))
    out = model(x)
    assert not hasattr(out, "bank_alpha") or out.bank_alpha is None
    assert not hasattr(out, "bank_confidence") or out.bank_confidence is None
    assert not hasattr(out, "bank_telemetry") or out.bank_telemetry is None


def test_hlm_bank_enabled_with_empty_bank_preserves_logits_close() -> None:
    model = AMCTransformer(_cfg(kv_lrank=64))
    x = torch.randint(0, 1000, (2, 8))
    out_no_bank = model(x)
    out_empty_bank = model(x, preference_bank=HLMPreferenceBank(HLMPreferenceBankConfig(bank_dim=64)))
    assert torch.allclose(out_no_bank.logits, out_empty_bank.logits, atol=1e-6)


def test_hlm_bank_enabled_with_written_slot_returns_bank_telemetry() -> None:
    model = AMCTransformer(_cfg(kv_lrank=64, use_hlm_bank=True))
    x = torch.randint(0, 1000, (1, 8))
    bank = _filled_bank(bank_dim=64)
    out = model(x, preference_bank=bank)
    assert out.bank_telemetry is not None
    assert out.bank_telemetry["filled_slots"] == 1


def test_hlm_bank_forward_changes_logits_when_nonempty() -> None:
    model = AMCTransformer(_cfg(kv_lrank=64, use_hlm_bank=True))
    x = torch.randint(0, 1000, (2, 8))
    out_baseline = model(x)
    bank = _filled_bank(bank_dim=64)
    out_with_bank = model(x, preference_bank=bank)
    assert not torch.allclose(out_baseline.logits, out_with_bank.logits, atol=1e-5)


# ── use_hlm_bank config gate tests ──────────────────────────────────────


def test_use_hlm_bank_defaults_to_false() -> None:
    cfg = _cfg()
    assert cfg.use_hlm_bank is False


def test_passing_bank_without_flag_does_not_change_logits() -> None:
    model = AMCTransformer(_cfg(kv_lrank=64))  # use_hlm_bank defaults to False
    x = torch.randint(0, 1000, (2, 8))
    out_no_bank = model(x)
    bank = _filled_bank(bank_dim=64)
    out_with_bank = model(x, preference_bank=bank)
    # Bank should be ignored since use_hlm_bank=False
    assert torch.allclose(out_no_bank.logits, out_with_bank.logits, atol=1e-6)
    assert out_with_bank.bank_alpha is None


def test_passing_bank_with_flag_true_changes_logits() -> None:
    model = AMCTransformer(_cfg(kv_lrank=64, use_hlm_bank=True))
    x = torch.randint(0, 1000, (2, 8))
    out_no_bank = model(x)
    bank = _filled_bank(bank_dim=64)
    out_with_bank = model(x, preference_bank=bank)
    assert not torch.allclose(out_no_bank.logits, out_with_bank.logits, atol=1e-5)
    assert out_with_bank.bank_alpha is not None
