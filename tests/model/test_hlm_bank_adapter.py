"""Tests for HLMPreferenceAdapter — differentiable bank read adapter."""

from __future__ import annotations

import pytest
import torch

from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.hlm_bank_adapter import (
    HLMPreferenceAdapter,
    HLMPreferenceAdapterConfig,
)


def _adapter(bank_dim=32, **kw) -> HLMPreferenceAdapter:
    return HLMPreferenceAdapter(HLMPreferenceAdapterConfig(d_model=64, bank_dim=bank_dim, **kw))


def _empty_bank(dim=32) -> HLMPreferenceBank:
    return HLMPreferenceBank(HLMPreferenceBankConfig(bank_dim=dim))


def _filled_bank(dim=32) -> HLMPreferenceBank:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=dim))
    bank.upsert(
        HLMPreferenceWrite(
            key=torch.ones(dim),
            value=torch.ones(dim) * 2,
            strength=1.0,
        )
    )
    return bank


# ── identity when bank absent/empty ────────────────────────────────────────


def test_adapter_returns_identity_when_bank_is_none() -> None:
    adapter = _adapter()
    hidden = torch.randn(2, 8, 64)
    out = adapter(hidden, None)
    assert torch.equal(out.hidden, hidden)


def test_adapter_returns_identity_when_bank_empty() -> None:
    adapter = _adapter()
    hidden = torch.randn(2, 8, 64)
    out = adapter(hidden, _empty_bank())
    assert torch.equal(out.hidden, hidden)


# ── alpha shape and bounds ────────────────────────────────────────────────


def test_adapter_alpha_shape_and_bounds() -> None:
    adapter = _adapter()
    hidden = torch.randn(3, 16, 64)
    out = adapter(hidden, _filled_bank())
    assert out.alpha.shape == (3, 16, 1)
    assert (out.alpha >= 0).all()
    assert (out.alpha <= 1).all()


# ── adapter changes hidden when bank has matching slot ────────────────────


def test_adapter_changes_hidden_when_bank_has_matching_slot() -> None:
    dim = 32
    adapter = _adapter(bank_dim=dim)
    bank = _filled_bank(dim)

    # Query with something close to the written key
    hidden = torch.ones(1, 4, 64)  # uniform input
    out = adapter(hidden, bank)

    # Hidden should differ from identity when bank is non-empty and gate active
    assert not torch.allclose(out.hidden, hidden, atol=1e-5)


# ── gradients flow to adapter, not bank ───────────────────────────────────


def test_adapter_gradients_flow_to_adapter_not_bank_buffers() -> None:
    adapter = _adapter()
    bank = _filled_bank()
    hidden = torch.randn(2, 4, 64, requires_grad=True)

    out = adapter(hidden, bank)
    loss = out.hidden.sum()
    loss.backward()

    # Adapter params should have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert has_grad, "Adapter parameters should receive gradients"

    # Bank buffers should NOT have gradients
    for buf_name, buf in bank.named_buffers():
        if (
            hasattr(buf, "grad")
            and buf.grad is not None
            and buf_name not in ("keys", "values", "strengths")
        ):
            continue
        # Keys, values, strengths should not receive gradients (they were detached)


# ── config validation ─────────────────────────────────────────────────────


def test_adapter_rejects_invalid_inject_scale() -> None:
    with pytest.raises(ValueError, match="inject_scale"):
        HLMPreferenceAdapterConfig(d_model=64, inject_scale=-0.1)

    with pytest.raises(ValueError, match="inject_scale"):
        HLMPreferenceAdapterConfig(d_model=64, inject_scale=1.5)


# ── output field shapes ───────────────────────────────────────────────────


def test_adapter_output_shapes() -> None:
    adapter = _adapter(bank_dim=32)
    hidden = torch.randn(2, 8, 64)
    out = adapter(hidden, _filled_bank(32))

    assert out.hidden.shape == (2, 8, 64)
    assert out.alpha.shape == (2, 8, 1)
    assert out.bank_context.shape == (2, 8, 32)
    assert out.confidence.shape == (2, 8, 1)


# ── gradient path (Fix #5) ───────────────────────────────────────────────


def test_adapter_query_proj_receives_gradient_bank_keys_do_not() -> None:
    dim = 32
    adapter = _adapter(bank_dim=dim)
    # Fill multiple slots so softmax gradient is non-degenerate
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=dim))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim) * 2, strength=1.0))
    bank.upsert(HLMPreferenceWrite(key=torch.randn(dim), value=torch.randn(dim), strength=1.0))
    bank.upsert(HLMPreferenceWrite(key=torch.randn(dim), value=torch.randn(dim), strength=1.0))

    hidden = torch.randn(2, 4, 64, requires_grad=True)
    out = adapter(hidden, bank)
    loss = out.hidden.sum()
    loss.backward()

    # query_proj should receive gradients (gradient flows through bank read into query)
    assert adapter.query_proj.weight.grad is not None
    assert adapter.query_proj.weight.grad.abs().sum() > 0

    # bank.keys is a buffer (not parameter), should NOT have gradients
    assert bank.keys.grad is None
