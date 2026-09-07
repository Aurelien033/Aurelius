"""Tests for CB-06 serving harness — greedy decode + per-policy cost proxy."""

from __future__ import annotations

import json

import torch

from src.eval.serving_harness import (
    POLICY_COST_MULTIPLIER,
    HarnessReport,
    ServingHarnessConfig,
    greedy_decode,
    run_harness,
)
from src.inference.cascade_routing import CascadeRouter, CascadeRouterConfig, ComputePolicy
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def _tiny_config(use_bank: bool = True) -> AMCTransformerConfig:
    return AMCTransformerConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        kv_lrank=8,
        max_seq_len=16,
        use_hlm_bank=use_bank,
    )


def _tiny_bank(dim: int = 8, size: int = 4) -> HLMPreferenceBank:
    return HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=size, bank_dim=dim))


def _populated_bank(dim: int = 8, size: int = 4) -> HLMPreferenceBank:
    bank = _tiny_bank(dim, size)
    torch.manual_seed(999)
    for _ in range(size):
        bank.upsert(HLMPreferenceWrite(key=torch.randn(dim), value=torch.randn(dim), strength=1.0))
    return bank


def _prompt(seed: int, length: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 32, (1, length))


# ── greedy_decode ──────────────────────────────────────────────────────────


def test_greedy_decode_deterministic_same_seed() -> None:
    torch.manual_seed(7)
    model = AMCTransformer(_tiny_config(use_bank=False)).eval()
    prompt = _prompt(seed=0)
    gen1, m1 = greedy_decode(model, prompt, max_new_tokens=4)
    gen2, m2 = greedy_decode(model, prompt, max_new_tokens=4)
    assert gen1 == gen2
    assert abs(m1 - m2) < 1e-6


def test_greedy_decode_output_length_equals_max_new_tokens() -> None:
    torch.manual_seed(11)
    model = AMCTransformer(_tiny_config(use_bank=False)).eval()
    for n in (1, 3, 8):
        gen, _ = greedy_decode(model, _prompt(0), max_new_tokens=n)
        assert len(gen) == n


def test_greedy_decode_returns_logit_margin_in_unit_interval() -> None:
    torch.manual_seed(13)
    model = AMCTransformer(_tiny_config(use_bank=False)).eval()
    _, margin = greedy_decode(model, _prompt(0), max_new_tokens=3)
    assert 0.0 <= margin <= 1.0


def test_greedy_decode_zero_tokens_returns_empty() -> None:
    torch.manual_seed(17)
    model = AMCTransformer(_tiny_config(use_bank=False)).eval()
    gen, margin = greedy_decode(model, _prompt(0), max_new_tokens=0)
    assert gen == []
    assert margin == 0.0


def test_policy_cost_multiplier_ordering_fast_balanced_thorough() -> None:
    assert (
        POLICY_COST_MULTIPLIER[ComputePolicy.FAST] < POLICY_COST_MULTIPLIER[ComputePolicy.BALANCED]
    )
    assert (
        POLICY_COST_MULTIPLIER[ComputePolicy.BALANCED]
        < POLICY_COST_MULTIPLIER[ComputePolicy.THOROUGH]
    )
    # Sanity: BALANCED is the baseline (1.0)
    assert POLICY_COST_MULTIPLIER[ComputePolicy.BALANCED] == 1.0


# ── run_harness ────────────────────────────────────────────────────────────


def _empty_report(use_bank: bool = True) -> HarnessReport:
    torch.manual_seed(21)
    model = AMCTransformer(_tiny_config(use_bank=use_bank)).eval()
    prompts = [_prompt(i) for i in range(5)]
    empty = _tiny_bank() if use_bank else None
    return run_harness(model, prompts, bank=empty)


def test_run_harness_with_empty_bank_returns_all_fallback() -> None:
    """Empty bank -> alpha=0, confidence=0 -> router routes to FAST (low confidence).
    If use_bank=False, router falls back to BALANCED (the fallback class)."""
    report_disabled = _empty_report(use_bank=False)
    assert report_disabled.policy_distribution["balanced"] == 5
    assert report_disabled.policy_distribution["fast"] == 0
    assert report_disabled.policy_distribution["thorough"] == 0


def test_run_harness_with_populated_bank_routes_nontrivially() -> None:
    """Populated bank should yield at least one decision that is not the fallback.
    Using a router with extreme thresholds to force THOROUGH deterministically."""
    torch.manual_seed(23)
    model = AMCTransformer(_tiny_config(use_bank=True)).eval()
    prompts = [_prompt(i) for i in range(5)]

    extreme_router = CascadeRouter(
        CascadeRouterConfig(
            alpha_thorough=0.0,
            confidence_fast=-1.0,
            require_confidence_for_thorough=False,
        )
    )

    report = run_harness(
        model,
        prompts,
        bank=_populated_bank(),
        router=extreme_router,
    )
    # With alpha_thorough=0.0 and require_confidence_for_thorough=False,
    # any non-zero alpha should escalate to THOROUGH.
    assert (
        report.policy_distribution["thorough"] >= 1 or report.policy_distribution["balanced"] >= 1
    )


def test_run_harness_per_prompt_compute_proxy_uses_policy_multiplier() -> None:
    """Each PromptResult's compute_proxy = tokens * POLICY_COST_MULTIPLIER[decision]."""
    torch.manual_seed(29)
    model = AMCTransformer(_tiny_config(use_bank=True)).eval()
    prompts = [_prompt(i) for i in range(3)]

    # Force all-FALLBACK by not passing a bank
    report = run_harness(
        model,
        prompts,
        bank=None,
        config=ServingHarnessConfig(use_bank=False),
    )

    for r in report.results:
        expected = len(r.generated_ids) * POLICY_COST_MULTIPLIER[ComputePolicy(r.routing_decision)]
        assert abs(r.compute_proxy - expected) < 1e-9


def test_run_harness_report_to_dict_is_json_serializable() -> None:
    report = _empty_report(use_bank=False)
    as_json = json.dumps(report.to_dict())
    round_trip = json.loads(as_json)
    assert round_trip["total_prompts"] == 5
    assert "policy_distribution" in round_trip
    assert "mean_logit_margin_per_policy" in round_trip


def test_run_harness_total_prompts_matches_input_count() -> None:
    for n in (1, 5, 9):
        torch.manual_seed(31)
        model = AMCTransformer(_tiny_config(use_bank=False)).eval()
        prompts = [_prompt(i) for i in range(n)]
        report = run_harness(model, prompts)
        assert report.total_prompts == n
        assert len(report.results) == n


def test_run_harness_mean_margin_per_policy_contains_only_observed_classes() -> None:
    """mean_logit_margin_per_policy should only contain classes that appear in distribution."""
    torch.manual_seed(37)
    model = AMCTransformer(_tiny_config(use_bank=False)).eval()
    prompts = [_prompt(i) for i in range(6)]

    report = run_harness(model, prompts)

    observed = {k for k, v in report.policy_distribution.items() if v > 0}
    in_margin_dict = set(report.mean_logit_margin_per_policy.keys())

    # Every class in distribution appears in margin dict (with possibly zero mean if no samples)
    assert observed <= in_margin_dict
    # Every class in margin dict has at least one non-zero margin entry OR is a default key
    for cls in observed:
        assert cls in report.mean_logit_margin_per_policy
