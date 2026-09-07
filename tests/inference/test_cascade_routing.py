"""CascadeRouter TDD tests — CB-01.

Strict TDD: these tests must exist and FAIL before the implementation.
"""

from __future__ import annotations

import torch

from src.inference.cascade_routing import (
    CascadeRouter,
    CascadeRouterConfig,
    ComputePolicy,
)


def test_router_default_config_exists_with_sane_thresholds():
    cfg = CascadeRouterConfig()
    assert cfg.alpha_thorough == 0.70
    assert cfg.confidence_fast == 0.40
    assert cfg.require_confidence_for_thorough is True
    assert cfg.balanced_confidence_floor == 0.30
    assert cfg.fallback == ComputePolicy.BALANCED


def test_router_missing_alpha_falls_back():
    router = CascadeRouter()
    assert router.decision(None, torch.tensor([[[0.5]]])) == ComputePolicy.BALANCED


def test_router_missing_confidence_falls_back():
    router = CascadeRouter()
    assert router.decision(torch.tensor([[[0.8]]]), None) == ComputePolicy.BALANCED


def test_router_both_none_falls_back():
    router = CascadeRouter()
    assert router.decision(None, None) == ComputePolicy.BALANCED


def test_router_custom_fallback():
    router = CascadeRouter(CascadeRouterConfig(fallback=ComputePolicy.FAST))
    assert router.decision(None, None) == ComputePolicy.FAST


def test_router_low_confidence_returns_fast():
    router = CascadeRouter()
    alpha = torch.tensor([[[0.8]]])  # high alpha
    conf = torch.tensor([[[0.1]]])  # low confidence (< 0.40)
    assert router.decision(alpha, conf) == ComputePolicy.FAST


def test_router_high_alpha_with_sufficient_confidence_returns_thorough():
    router = CascadeRouter()
    alpha = torch.tensor([[[0.8]]])  # > 0.70
    conf = torch.tensor([[[0.5]]])  # > 0.40 and > 0.30
    assert router.decision(alpha, conf) == ComputePolicy.THOROUGH


def test_router_high_alpha_with_low_confidence_returns_balanced_when_required():
    router = CascadeRouter()
    alpha = torch.tensor([[[0.8]]])  # > 0.70
    conf = torch.tensor([[[0.25]]])  # > 0.40 is false, so FAST
    assert router.decision(alpha, conf) == ComputePolicy.FAST


def test_router_high_alpha_balanced_confidence_returns_balanced():
    """High alpha but confidence between confidence_fast and balanced_confidence_floor."""
    router = CascadeRouter(
        CascadeRouterConfig(
            alpha_thorough=0.70,
            confidence_fast=0.20,
            balanced_confidence_floor=0.40,
        )
    )
    alpha = torch.tensor([[[0.8]]])  # > 0.70
    conf = torch.tensor([[[0.30]]])  # > 0.20 (not fast), but < 0.40 (fails floor)
    assert router.decision(alpha, conf) == ComputePolicy.BALANCED


def test_router_ambiguous_input_returns_balanced():
    """Alpha below thorough threshold, confidence above fast threshold."""
    router = CascadeRouter()
    alpha = torch.tensor([[[0.5]]])  # < 0.70
    conf = torch.tensor([[[0.6]]])  # > 0.40
    assert router.decision(alpha, conf) == ComputePolicy.BALANCED


def test_router_zero_tensors_returns_fast():
    router = CascadeRouter()
    alpha = torch.zeros(1, 5, 1)
    conf = torch.zeros(1, 5, 1)
    assert router.decision(alpha, conf) == ComputePolicy.FAST


def test_router_batched_input_reduces_correctly():
    """Multi-token batch: max(alpha) is used for thorough, mean(conf) for fast."""
    router = CascadeRouter()
    # 3 tokens, max alpha = 0.9 (>= 0.70), mean conf = 0.6 (> 0.40)
    alpha = torch.tensor([[[0.3], [0.5], [0.9]]])
    conf = torch.tensor([[[0.5], [0.6], [0.7]]])
    assert router.decision(alpha, conf) == ComputePolicy.THOROUGH


def test_router_batched_low_mean_conf_returns_fast():
    """Even with high max alpha, if mean confidence is low → FAST."""
    router = CascadeRouter()
    alpha = torch.tensor([[[0.9], [0.9], [0.9]]])
    conf = torch.tensor([[[0.1], [0.1], [0.1]]])  # mean 0.1 < 0.40
    assert router.decision(alpha, conf) == ComputePolicy.FAST


def test_telemetry_counts_distribution():
    router = CascadeRouter()
    history = [
        ComputePolicy.FAST,
        ComputePolicy.FAST,
        ComputePolicy.BALANCED,
        ComputePolicy.THOROUGH,
        ComputePolicy.THOROUGH,
        ComputePolicy.THOROUGH,
    ]
    result = router.telemetry(history)
    assert result["fast"] == 2
    assert result["balanced"] == 1
    assert result["thorough"] == 3
    assert result["total"] == 6


def test_telemetry_empty_history():
    router = CascadeRouter()
    result = router.telemetry([])
    assert result["fast"] == 0
    assert result["balanced"] == 0
    assert result["thorough"] == 0
    assert result["total"] == 0


def test_telemetry_none_history():
    router = CascadeRouter()
    result = router.telemetry(None)
    assert result["total"] == 0


def test_router_never_raises():
    """Router should never raise regardless of input shape."""
    router = CascadeRouter()
    # Various shapes
    for alpha, conf in [
        (torch.tensor([[[0.5]]]), torch.tensor([[[0.5]]])),
        (torch.randn(2, 10, 1), torch.randn(2, 10, 1)),
        (torch.tensor(0.5), torch.tensor(0.5)),
    ]:
        result = router.decision(alpha, conf)
        assert isinstance(result, ComputePolicy)


def test_compute_policy_is_str_enum():
    assert isinstance(ComputePolicy.FAST, str)
    assert ComputePolicy.FAST == "fast"
    assert ComputePolicy.BALANCED == "balanced"
    assert ComputePolicy.THOROUGH == "thorough"
