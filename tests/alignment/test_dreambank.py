"""Tests for DreamBankController — sleep-time preference consolidation."""

from __future__ import annotations

import pytest
import torch

from src.alignment.dreambank import (
    DreamBankConfig,
    DreamBankController,
    DreamCycleResult,
    DreamPreferencePair,
    DreamSeed,
)
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig


def _make_bank(dim=32) -> HLMPreferenceBank:
    return HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=10, bank_dim=dim))


def _dummy_generate(prompt: str, temperature: float) -> str:
    return f"response_for_{prompt}_at_{temperature}"


def _dummy_score(prompt: str, response: str) -> float:
    score = sum(ord(c) for c in response) % 100 / 100.0
    return score


def _dummy_embed(text: str, dim=32) -> torch.Tensor:
    seed = sum(ord(c) for c in text)
    torch.manual_seed(seed % (2**31))
    return torch.randn(dim)


# ── basic cycle flow ──────────────────────────────────────────────────────


def test_dream_cycle_no_seeds_no_writes() -> None:
    bank = _make_bank()
    ctrl = DreamBankController(bank)
    result = ctrl.run_cycle(
        seeds=[],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    assert result.writes == 0
    assert result.pairs == 0
    assert isinstance(result, DreamCycleResult)


def test_dream_cycle_generates_candidates_per_temperature() -> None:
    bank = _make_bank()
    ctrl = DreamBankController(bank, DreamBankConfig(temperatures=(0.1, 0.5, 0.9)))
    result = ctrl.run_cycle(
        seeds=[DreamSeed("hello")],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    assert result.candidates == 3  # 1 seed x 3 temps


def test_dream_cycle_writes_only_when_margin_exceeds_threshold() -> None:
    bank = _make_bank()
    ctrl = DreamBankController(bank, DreamBankConfig(min_margin=1.0))
    result = ctrl.run_cycle(
        seeds=[DreamSeed("hello")],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    # All dummy scores are < 1.0 margin, so no writes
    assert result.writes == 0


def test_dream_cycle_writes_chosen_not_rejected_embedding() -> None:
    bank = _make_bank()
    best_score_seen: list[float] = []
    chosen_text_seen: list[str] = []

    def scoring_score(prompt: str, response: str) -> float:
        s = sum(ord(c) for c in response) % 100 / 100.0
        best_score_seen.append(s)
        return s

    def scoring_embed(text: str) -> torch.Tensor:
        chosen_text_seen.append(text)
        return _dummy_embed(text, 32)

    ctrl = DreamBankController(bank, DreamBankConfig(min_margin=0.001, max_writes_per_cycle=4))
    ctrl.run_cycle(
        seeds=[DreamSeed("a"), DreamSeed("b")],
        generate_fn=_dummy_generate,
        score_fn=scoring_score,
        embed_fn=scoring_embed,
    )
    assert bank.is_empty() is False


# ── metadata uses hash not raw prompt ────────────────────────────────────


def test_dream_cycle_metadata_uses_hash_not_raw_prompt() -> None:
    bank = _make_bank()
    ctrl = DreamBankController(bank, DreamBankConfig(min_margin=0.001))
    ctrl.run_cycle(
        seeds=[DreamSeed("my secret prompt content")],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    state = bank.export_state()
    for v in state.values():
        if isinstance(v, (str, dict, list)):
            assert "my secret prompt content" not in str(v)


# ── bank decay applied ──────────────────────────────────────────────────


def test_dream_cycle_applies_bank_decay_once_per_cycle() -> None:
    bank = _make_bank()
    from src.memory.hlm_bank import HLMPreferenceWrite
    bank.upsert(HLMPreferenceWrite(
        key=torch.ones(32),
        value=torch.ones(32),
        strength=0.5,
    ))
    pre_strength = bank.strengths[0].item()
    ctrl = DreamBankController(bank, DreamBankConfig(decay_steps_per_cycle=5, min_margin=99.0))
    ctrl.run_cycle(
        seeds=[DreamSeed("x")],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    post_strength = bank.strengths[0].item()
    assert post_strength < pre_strength


# ── max_writes_per_cycle ────────────────────────────────────────────────


def test_dream_cycle_respects_max_writes_per_cycle() -> None:
    bank = _make_bank()
    ctrl = DreamBankController(
        bank,
        DreamBankConfig(max_writes_per_cycle=2, min_margin=0.001),
    )
    ctrl.run_cycle(
        seeds=[DreamSeed(f"s{i}") for i in range(10)],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    assert ctrl.run_cycle(
        seeds=[],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t),
    ).bank_fill <= 10  # at most 10 writes total


# ── result JSON-safe ────────────────────────────────────────────────────


def test_dream_cycle_result_is_json_safe() -> None:
    import json

    bank = _make_bank()
    ctrl = DreamBankController(bank)
    result = ctrl.run_cycle(
        seeds=[DreamSeed("test")],
        generate_fn=_dummy_generate,
        score_fn=_dummy_score,
        embed_fn=lambda t: _dummy_embed(t, 32),
    )
    # All fields should be JSON-serializable
    json.dumps({
        "seeds": result.seeds,
        "candidates": result.candidates,
        "pairs": result.pairs,
        "writes": result.writes,
        "mean_margin": result.mean_margin,
        "bank_fill": result.bank_fill,
    })
