"""Tests for differential privacy bounds on federated bank deltas."""

from __future__ import annotations

import math

import pytest

from src.privacy.dp_bounds import (
    GaussianMechanismConfig,
    bank_tensor_element_count,
    compute_epsilon,
    compute_sigma_for_target,
    dp_parameterized_table,
)

# ── Config validation ─────────────────────────────────────────────────────


def test_gaussian_mechanism_config_defaults() -> None:
    cfg = GaussianMechanismConfig()
    assert cfg.clip_norm == 1.0
    assert cfg.sigma == 1.0
    assert cfg.delta == 1e-5
    assert cfg.num_devices == 8


def test_gaussian_mechanism_config_rejects_invalid() -> None:
    for bad in [
        dict(clip_norm=-1.0),
        dict(sigma=0.0),
        dict(delta=0.0),
        dict(delta=1.0),
        dict(delta=-0.1),
        dict(num_devices=0),
    ]:
        with pytest.raises(ValueError):
            GaussianMechanismConfig(**bad)  # type: ignore[arg-type]


# ── Epsilon formula ────────────────────────────────────────────────────────


def test_gaussian_epsilon_formula() -> None:
    """Manual verification: C=1, σ=1, δ=1e-5 → ε = √(2 * ln(1.25e5))"""
    C = 1.0
    sigma = 1.0
    delta = 1e-5
    expected = (C) * math.sqrt(2 * math.log(1.25 / delta)) / sigma
    actual = compute_epsilon(C, sigma, delta)
    assert abs(actual - expected) < 1e-10
    # Sanity: should be finite and positive
    assert 0 < actual < 100


def test_epsilon_decreases_as_sigma_increases() -> None:
    eps_1 = compute_epsilon(1.0, 1.0, 1e-5)
    eps_2 = compute_epsilon(1.0, 2.0, 1e-5)
    eps_5 = compute_epsilon(1.0, 5.0, 1e-5)
    assert eps_1 > eps_2 > eps_5


def test_epsilon_increases_as_delta_decreases() -> None:
    """Smaller δ → stricter privacy → higher ε for fixed σ."""
    eps_large_delta = compute_epsilon(1.0, 1.0, 1e-3)
    eps_small_delta = compute_epsilon(1.0, 1.0, 1e-6)
    assert eps_small_delta > eps_large_delta


def test_epsilon_scales_linearly_with_clip_norm() -> None:
    """ε ∝ clip_norm when σ, δ are fixed."""
    eps_c1 = compute_epsilon(1.0, 1.0, 1e-5)
    eps_c2 = compute_epsilon(2.0, 1.0, 1e-5)
    assert abs(eps_c2 - 2 * eps_c1) < 1e-10


def test_default_config_epsilon_at_delta_1e5() -> None:
    """Our default config: C=1, σ=1, δ=1e-5, M=8.
    ε = (1/8) * √(2 * ln(1.25e5)) ≈ (1/8) * 7.18 ≈ 0.897"""
    cfg = GaussianMechanismConfig()
    eps = compute_epsilon(cfg.clip_norm, cfg.sigma, cfg.delta, cfg.num_devices)
    expected_approx = (1.0 / 8.0) * math.sqrt(2 * math.log(1.25 / 1e-5))
    assert abs(eps - expected_approx) < 1e-10
    assert eps < 1.0  # Should be sub-ε=1 at default settings


def test_epsilon_is_finite_for_valid_params() -> None:
    for C in [0.1, 1.0, 10.0]:
        for sigma in [0.1, 1.0, 10.0]:
            for delta in [1e-3, 1e-5, 1e-9]:
                for M in [1, 8, 64]:
                    eps = compute_epsilon(C, sigma, delta, M)
                    assert math.isfinite(eps)
                    assert eps > 0


def test_sigma_for_target() -> None:
    """Compute σ needed for ε≤1.0 at δ=1e-5, C=1.0, M=8."""
    sigma = compute_sigma_for_target(clip_norm=1.0, target_epsilon=1.0, delta=1e-5, num_devices=8)
    # Verify: this σ should give ε ≈ 1.0
    eps = compute_epsilon(1.0, sigma, 1e-5, 8)
    assert abs(eps - 1.0) < 1e-6


# ── Bank tensor element count ──────────────────────────────────────────────


def test_bank_tensor_element_count() -> None:
    """2 * bank_size * bank_dim + bank_size"""
    assert bank_tensor_element_count(8, 8) == 2 * 8 * 8 + 8  # 136
    assert bank_tensor_element_count(1, 1) == 3
    assert bank_tensor_element_count(14, 64) == 2 * 14 * 64 + 14  # 1806


# ── Parameterized table ────────────────────────────────────────────────────


def test_dp_parameterized_table_structure() -> None:
    table = dp_parameterized_table(clip_norm=1.0, num_devices=8, deltas=(1e-5,), sigmas=(1.0, 2.0))
    assert len(table) == 2
    assert table[0]["sigma"] == 1.0
    assert table[1]["sigma"] == 2.0
    # Each row should have epsilon for each delta
    assert "eps_d1e-05" in table[0]


def test_dp_parameterized_table_monotone_in_sigma() -> None:
    """In each row, higher sigma should give lower epsilon."""
    table = dp_parameterized_table(sigmas=(0.1, 0.5, 1.0, 5.0, 10.0))
    for key in table[0]:
        if key.startswith("eps_"):
            values = [row[key] for row in table]
            # Should be strictly decreasing
            for i in range(len(values) - 1):
                assert values[i] > values[i + 1], f"{key}: {values[i]} should be > {values[i + 1]}"
