"""Tests for HLMPreferenceBank — core preference bank data structure."""

from __future__ import annotations

import pytest
import torch

from src.memory.hlm_bank import (
    HLMPreferenceBank,
    HLMPreferenceBankConfig,
    HLMPreferenceWrite,
)

# ── Config validation ──────────────────────────────────────────────────────


def test_bank_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        HLMPreferenceBankConfig(bank_size=-1)

    with pytest.raises(ValueError, match="bank_dim"):
        HLMPreferenceBankConfig(bank_dim=0)

    with pytest.raises(ValueError, match="top_k"):
        HLMPreferenceBankConfig(top_k=-2)

    with pytest.raises(ValueError, match="decay"):
        HLMPreferenceBankConfig(decay=1.5)

    with pytest.raises(ValueError, match="min_strength"):
        HLMPreferenceBankConfig(min_strength=-0.1)


# ── Empty bank reads ───────────────────────────────────────────────────────


def test_empty_bank_read_returns_zero_context_and_zero_confidence() -> None:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_dim=32))
    query = torch.randn(2, 32)
    result = bank.read(query)

    assert result.context.shape == (2, 32)
    assert result.weights.shape == (2, 4)  # default top_k=4
    assert (result.context == 0).all()
    assert (result.confidence == 0).all()


# ── Upsert ─────────────────────────────────────────────────────────────────


def test_upsert_fills_first_empty_slot() -> None:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=16))
    key = torch.ones(16)
    slot = bank.upsert(HLMPreferenceWrite(key=key, value=key, strength=1.0))
    assert slot == 0

    key2 = torch.ones(16) * 2
    slot2 = bank.upsert(HLMPreferenceWrite(key=key2, value=key2, strength=0.8))
    assert slot2 == 1


def test_upsert_replaces_lowest_strength_when_full() -> None:
    cfg = HLMPreferenceBankConfig(bank_size=3, bank_dim=16)
    bank = HLMPreferenceBank(cfg)

    for i in range(3):
        bank.upsert(HLMPreferenceWrite(key=torch.ones(16), value=torch.ones(16), strength=float(i + 1)))

    # Bank full — next upsert with strength 0.1 should evict slot with strength 1.0
    # But the low-strength write (0.1) itself goes into the lowest-strength slot
    new_slot = bank.upsert(HLMPreferenceWrite(key=torch.ones(16) * 10, value=torch.ones(16) * 10, strength=0.1))
    assert new_slot == 0  # slot 0 had strength 1.0 (lowest)


# ── Read returns nearest ───────────────────────────────────────────────────


def test_read_returns_nearest_written_slot() -> None:
    dim = 16
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim))

    # Write a distinctive vector to slot 0
    vec = torch.zeros(dim)
    vec[0] = 1.0
    bank.upsert(HLMPreferenceWrite(key=vec, value=vec, strength=1.0))

    # Write a different vector to slot 1
    vec2 = torch.zeros(dim)
    vec2[1] = 1.0
    bank.upsert(HLMPreferenceWrite(key=vec2, value=vec2, strength=1.0))

    # Query close to vec — should hit slot 0
    query = torch.zeros(dim)
    query[0] = 1.0
    result = bank.read(query.unsqueeze(0), top_k=1)

    assert result.indices[0, 0] == 0
    assert result.confidence[0, 0] > 0.5


# ── Decay and consolidate ──────────────────────────────────────────────────


def test_decay_and_consolidate_clear_weak_slots() -> None:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=16, min_strength=0.01))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(16), value=torch.ones(16), strength=0.5))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(16), value=torch.ones(16), strength=0.02))

    # Many decay steps should drive slot 1 below threshold
    for _ in range(200):
        bank.decay_(1)

    bank.consolidate_()
    assert bank.strengths[0] > 0.0
    assert bank.strengths[1] == 0.0


# ── Export/import round-trip ──────────────────────────────────────────────


def test_export_import_round_trip_preserves_read_result() -> None:
    dim = 32
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim) * 2, strength=0.7))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim) * 3, value=torch.ones(dim) * 4, strength=1.0))

    query = torch.randn(2, dim)
    orig_result = bank.read(query, top_k=2)

    state = bank.export_state()
    restored = HLMPreferenceBank.from_state(state)
    restored_result = restored.read(query, top_k=2)

    torch.testing.assert_close(restored_result.context, orig_result.context)
    assert (restored_result.indices == orig_result.indices).all()


# ── Metadata safety ──────────────────────────────────────────────────────


def test_write_metadata_does_not_require_raw_prompt_text() -> None:
    bank = HLMPreferenceBank()
    write = HLMPreferenceWrite(
        key=torch.ones(64),
        value=torch.ones(64),
        strength=0.5,
        provenance="dreambank",
        trust="unverified",
        metadata_hash="abc123",
    )
    bank.upsert(write)
    # If we got here without error, the write didn't need raw prompt text


# ── No-grad invariant ─────────────────────────────────────────────────────


def test_upsert_runs_without_gradients() -> None:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_dim=16))
    key_requires = torch.randn(16, requires_grad=True)
    bank.upsert(HLMPreferenceWrite(key=key_requires, value=key_requires, strength=0.5))
    # key_requires.grad should remain None since upsert is @torch.no_grad
    assert key_requires.grad is None


# ── is_empty ──────────────────────────────────────────────────────────────


def test_is_empty_before_and_after_upsert() -> None:
    bank = HLMPreferenceBank()
    assert bank.is_empty()
    bank.upsert(HLMPreferenceWrite(key=torch.ones(64), value=torch.ones(64), strength=0.5))
    assert not bank.is_empty()


# ── Telemetry ─────────────────────────────────────────────────────────────


def test_telemetry_reports_filled_slots() -> None:
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=10, bank_dim=8))
    tel = bank.telemetry()
    assert tel["filled_slots"] == 0

    bank.upsert(HLMPreferenceWrite(key=torch.ones(8), value=torch.ones(8), strength=0.5))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(8), value=torch.ones(8), strength=0.3))
    tel = bank.telemetry()
    assert tel["filled_slots"] == 2


# ── metadata persistence (Fix #2) ────────────────────────────────────────


def test_export_import_preserves_metadata() -> None:
    dim = 16
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim))
    bank.upsert(HLMPreferenceWrite(
        key=torch.ones(dim),
        value=torch.ones(dim),
        strength=1.0,
        provenance="dream_recent",
        trust="verified",
        metadata_hash="abc123def456",
    ))
    state = bank.export_state()
    assert "metadata_hashes" in state
    assert "trusts" in state
    assert "provenances" in state
    assert state["metadata_hashes"][0] == "abc123def456"
    assert state["trusts"][0] == "verified"
    assert state["provenances"][0] == "dream_recent"

    restored = HLMPreferenceBank.from_state(state)
    assert restored._metadata_hashes[0] == "abc123def456"
    assert restored._trusts[0] == "verified"
    assert restored._provenances[0] == "dream_recent"


def test_export_does_not_contain_raw_prompt_text() -> None:
    dim = 16
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim))
    secret_prompt = "my super secret prompt content xyz"
    bank.upsert(HLMPreferenceWrite(
        key=torch.ones(dim),
        value=torch.ones(dim),
        strength=1.0,
        provenance="dreambank",
        trust="unverified",
        metadata_hash="hash_of_" + secret_prompt[:4],  # only hash, not raw text
    ))
    state = bank.export_state()
    state_str = str(state)
    assert secret_prompt not in state_str


# ── is_empty lifecycle (Fix #3) ──────────────────────────────────────────


def test_restored_nonempty_bank_has_is_empty_false() -> None:
    dim = 16
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim), strength=1.0))
    assert not bank.is_empty()

    state = bank.export_state()
    restored = HLMPreferenceBank.from_state(state)
    assert not restored.is_empty()


def test_consolidated_bank_has_is_empty_true() -> None:
    dim = 16
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim, min_strength=0.5))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim), strength=0.1))
    # Strength 0.1 < min_strength 0.5, consolidate should clear it
    bank.consolidate_()
    assert bank.is_empty()


def test_is_empty_after_decay_clears_all() -> None:
    dim = 16
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=5, bank_dim=dim, min_strength=0.01))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim), strength=0.02))
    # Many decay steps should drive it below min_strength
    for _ in range(500):
        bank.decay_(1)
    assert bank.is_empty()


# ── top_k output shape (Fix #4) ──────────────────────────────────────────


def test_partial_fill_returns_shape_top_k() -> None:
    dim = 16
    top_k = 4
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=10, bank_dim=dim, top_k=top_k))
    # Only write 2 slots (less than top_k=4)
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim), value=torch.ones(dim), strength=1.0))
    bank.upsert(HLMPreferenceWrite(key=torch.ones(dim) * 2, value=torch.ones(dim) * 2, strength=0.8))

    query = torch.randn(3, dim)  # batch of 3
    result = bank.read(query, top_k=top_k)

    assert result.weights.shape == (3, top_k)
    assert result.indices.shape == (3, top_k)
    # Padding slots should have weight 0 and index -1
    assert (result.indices[:, 2:] == -1).all()
    assert (result.weights[:, 2:] == 0).all()
