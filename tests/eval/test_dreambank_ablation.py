"""Tests for DreamBank ablation harness."""

from __future__ import annotations

import json

import torch

from src.eval.dreambank_ablation import AblationResult, run_ablation
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def _tiny_model() -> AMCTransformer:
    cfg = AMCTransformerConfig(
        vocab_size=500,
        d_model=64,
        n_layers=4,
        n_heads=4,
        kv_lrank=64,
        ssm_d_state=32,
        ssm_headdim=32,
        ssm_expand=2,
        max_seq_len=64,
        use_hlm_bank=True,
    )
    return AMCTransformer(cfg).eval()


def _tiny_filled_bank() -> HLMPreferenceBank:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=64))
    bank.upsert(HLMPreferenceWrite(
        key=torch.ones(64), value=torch.ones(64) * 3, strength=1.0,
    ))
    return bank


def test_ablation_returns_zeroish_delta_for_empty_bank() -> None:
    model = _tiny_model()
    x = torch.randint(0, 500, (1, 16))
    result = run_ablation(model, x)
    assert result.empty_bank
    assert result.mean_logit_delta < 1e-4  # essentially zero
    assert isinstance(result, AblationResult)


def test_ablation_returns_positive_delta_for_nonempty_bank() -> None:
    model = _tiny_model()
    x = torch.randint(0, 500, (1, 16))
    bank = _tiny_filled_bank()
    result = run_ablation(model, x, bank=bank)
    assert not result.empty_bank
    assert result.mean_logit_delta > 0.0


def test_ablation_result_is_json_serializable() -> None:
    model = _tiny_model()
    x = torch.randint(0, 500, (1, 16))
    result = run_ablation(model, x, bank=_tiny_filled_bank())
    d = result.to_dict()
    # Should not raise
    json.dumps(d)
    assert "mean_logit_delta" in d
    assert "bank_fill" in d
