"""Tests for Ring 1 measured DreamBank lift (Tranche 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.eval.ring1_measured_lift import (
    bank_from_dreambank_preferences,
    measure_dreambank_lift,
    trace_prompt_to_input_ids,
)
from src.eval.ring1_model_loader import load_ring1_amc_model, resolve_checkpoint_weights_path
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite

CHECKPOINT_ROOT = Path("checkpoints/aurelius-1.3b")


def test_trace_prompt_to_input_ids_is_deterministic() -> None:
    trace = {"trace_id": "abc", "steps": [{"observation": "find entity alpha beta"}]}
    first = trace_prompt_to_input_ids(trace, vocab_size=256)
    second = trace_prompt_to_input_ids(trace, vocab_size=256)
    assert torch.equal(first, second)
    assert first.shape == (1, 4)


def test_bank_from_dreambank_preferences_fills_slots() -> None:
    prefs = [
        {"provenance": "dream_a:step_1:observation", "strength": 0.8, "metadata_hash": "aaa"},
        {"provenance": "dream_b:step_2:reflection", "strength": 0.6, "metadata_hash": "bbb"},
    ]
    bank = bank_from_dreambank_preferences(prefs, bank_config={"bank_size": 14, "bank_dim": 64})
    assert int(bank.strengths.gt(0).sum().item()) == 2


@pytest.mark.skipif(
    resolve_checkpoint_weights_path(CHECKPOINT_ROOT) is None,
    reason="local checkpoint not present",
)
def test_measured_lift_exceeds_proxy_on_holdout() -> None:
    model, _, _ = load_ring1_amc_model(CHECKPOINT_ROOT)
    holdout = [
        {
            "trace_id": f"t{i}",
            "final_outcome": "success" if i % 2 == 0 else "failure",
            "steps": [{"observation": f"query task variant {i} with context"}],
        }
        for i in range(20)
    ]
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=14, bank_dim=64))
    for index in range(6):
        vector = torch.randn(64)
        bank.upsert(
            HLMPreferenceWrite(
                key=vector,
                value=vector,
                strength=0.7 + index * 0.03,
                provenance=f"pref-{index}",
                metadata_hash=f"hash-{index}",
            )
        )

    measured = measure_dreambank_lift(
        holdout,
        model=model,
        filled_bank=bank,
        lift_delta_scale=50.0,
        param_hash_unchanged=True,
        bank_fill=6,
    )
    assert measured.absolute_lift_pp >= 1.0
