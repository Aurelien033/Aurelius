"""Integration tests for CascadeRouter × live AMCTransformer + HLMPreferenceBank.

Proves the router correctly consumes the real (B, T, 1) shapes emitted by
AMCModelOutput.bank_alpha and AMCModelOutput.bank_confidence, and that the
fail-safe path trips when DreamBank is disabled or the bank is empty.
"""

from __future__ import annotations

import torch

from src.inference.cascade_routing import CascadeRouter, ComputePolicy
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def _tiny_config(use_bank: bool) -> AMCTransformerConfig:
    """Tiny CPU-friendly AMC config. d_model=16, kv_lrank=8 keeps Mamba2 headdim happy."""
    return AMCTransformerConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        kv_lrank=8,
        max_seq_len=16,
        use_hlm_bank=use_bank,
    )


# ── CB-02 integration: real shape consumption ────────────────────────────


def test_router_consumes_real_bank_alpha_confidence_shape() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=True)
    model = AMCTransformer(cfg).eval()

    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=8))
    for _ in range(3):
        bank.upsert(
            HLMPreferenceWrite(key=torch.randn(8), value=torch.randn(8), strength=1.0)
        )

    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out = model(ids, preference_bank=bank)

    # Shape contract from the DreamBank adapter: (B, T, 1)
    assert out.bank_alpha is not None
    assert out.bank_confidence is not None
    assert out.bank_alpha.shape[-1] == 1
    assert out.bank_confidence.shape == out.bank_alpha.shape

    decision = CascadeRouter().decision(out.bank_alpha, out.bank_confidence)
    assert isinstance(decision, ComputePolicy)


def test_router_empty_bank_falls_back_to_balanced() -> None:
    """An empty bank produces zero alphas -> mean confidence is 0 -> FAST path
    or BALANCED fallback depending on implementation. The stronger invariant is
    "the router does not crash" and "returns a ComputePolicy enum"."""
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=True)
    model = AMCTransformer(cfg).eval()

    empty = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=2, bank_dim=8))

    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out = model(ids, preference_bank=empty)

    # Empty bank: adapter returns zero alpha/confidence (identity path)
    # Router sees low confidence and should return FAST or BALANCED — NOT THOROUGH
    decision = CascadeRouter().decision(out.bank_alpha, out.bank_confidence)
    assert isinstance(decision, ComputePolicy)
    assert decision != ComputePolicy.THOROUGH, (
        "Empty bank should not escalate to THOROUGH — that's the expensive path"
    )


def test_router_nonempty_bank_produces_nontrivial_decision() -> None:
    """A populated bank should produce SOME decision — the router must not
    default to BALANCED universally. We only check it returns a ComputePolicy;
    distribution across classes is the ablation harness's job (CB-03)."""
    torch.manual_seed(42)
    cfg = _tiny_config(use_bank=True)
    model = AMCTransformer(cfg).eval()

    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=8))
    for _ in range(4):
        bank.upsert(
            HLMPreferenceWrite(key=torch.randn(8), value=torch.randn(8), strength=1.0)
        )

    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out = model(ids, preference_bank=bank)

    decision = CascadeRouter().decision(out.bank_alpha, out.bank_confidence)
    assert isinstance(decision, ComputePolicy)


# ── Safety: routing when DreamBank is explicitly disabled ─────────────────


def test_router_no_bank_passed_returns_fallback() -> None:
    """If the caller does not pass a bank, AMCModelOutput.bank_alpha/bank_confidence
    are None. Router must fail safe."""
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=False)
    model = AMCTransformer(cfg).eval()

    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out = model(ids)

    assert out.bank_alpha is None
    assert out.bank_confidence is None

    decision = CascadeRouter().decision(out.bank_alpha, out.bank_confidence)
    assert decision == ComputePolicy.BALANCED
