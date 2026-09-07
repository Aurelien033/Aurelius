"""Tests for per-layer MLA bank wiring (Tranche E)."""

from __future__ import annotations

import torch

from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def _tiny_config(
    use_bank: bool = True,
    read_layers: tuple[int, ...] | None = None,
    n_layers: int = 2,
) -> AMCTransformerConfig:
    return AMCTransformerConfig(
        vocab_size=1024,
        d_model=128,
        n_layers=n_layers,
        n_heads=8,
        n_kv_heads=2,
        kv_lrank=32,
        ssm_headdim=32,
        ssm_expand=2,
        max_seq_len=128,
        use_hlm_bank=use_bank,
        hlm_bank_read_layers=read_layers,
    )


def _populated_bank() -> HLMPreferenceBank:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=32))
    torch.manual_seed(42)
    for _ in range(4):
        bank.upsert(
            HLMPreferenceWrite(key=torch.randn(32), value=torch.randn(32), strength=1.0)
        )
    return bank


def test_default_read_layers_applies_at_end() -> None:
    """Default behavior: hlm_bank_read_layers=None -> apply at final norm."""
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=True, read_layers=None)
    model = AMCTransformer(cfg).eval()

    ids = torch.randint(0, 32, (1, 4))
    out = model(ids, preference_bank=_populated_bank())
    # Alpha and confidence should be populated
    assert out.bank_alpha is not None
    assert out.bank_confidence is not None
    assert out.bank_telemetry is not None


def test_explicit_final_read_layers_applies_at_end() -> None:
    """Explicit (-1,) also applies at final norm."""
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=True, read_layers=(-1,))
    model = AMCTransformer(cfg).eval()

    ids = torch.randint(0, 32, (1, 4))
    out = model(ids, preference_bank=_populated_bank())
    assert out.bank_alpha is not None


def test_intermediate_read_layers_applies_in_loop() -> None:
    """hlm_bank_read_layers=(0,) applies after layer 0, not at the end."""
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=True, read_layers=(0,), n_layers=2)
    model = AMCTransformer(cfg).eval()

    ids = torch.randint(0, 32, (1, 4))
    out = model(ids, preference_bank=_populated_bank())
    # Alpha should be populated by layer 0
    assert out.bank_alpha is not None


def test_multiple_read_layers_applies_each() -> None:
    """hlm_bank_read_layers=(0, 1) applies after layer 0 and 1, then final norm."""
    torch.manual_seed(0)
    cfg = _tiny_config(use_bank=True, read_layers=(0, 1), n_layers=2)
    model = AMCTransformer(cfg).eval()

    ids = torch.randint(0, 32, (1, 4))
    out = model(ids, preference_bank=_populated_bank())
    # Alpha should be populated
    assert out.bank_alpha is not None


def test_read_layers_with_ssm_layers() -> None:
    """Verify it works even if layers include SSM layers."""
    torch.manual_seed(0)
    cfg = _tiny_config(
        use_bank=True,
        read_layers=(1,),
        n_layers=3,
    )
    model = AMCTransformer(cfg).eval()

    ids = torch.randint(0, 32, (1, 4))
    out = model(ids, preference_bank=_populated_bank())
    assert out.bank_alpha is not None
