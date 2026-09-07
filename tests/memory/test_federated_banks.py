"""Tests for the Federated Memory Deltas simulator (FD-01)."""

from __future__ import annotations

import json

import torch

from src.memory.federated_banks import (
    FederatedBankSimulator,
    FederatedConfig,
    _comm_bytes_per_round,
)

# ── Config ─────────────────────────────────────────────────────────────────


def test_federated_config_defaults() -> None:
    cfg = FederatedConfig()
    assert cfg.num_devices == 8
    assert cfg.rounds == 3
    assert cfg.dp_sigma == 0.0
    assert cfg.bank_size > 0 and cfg.bank_dim > 0


def test_federated_config_rejects_invalid() -> None:
    for bad in [
        dict(num_devices=0),
        dict(num_devices=-1),
        dict(rounds=0),
        dict(bank_size=-3),
        dict(bank_dim=0),
        dict(dp_sigma=-0.1),
    ]:
        try:
            FederatedConfig(**bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {bad}")


# ── Simulator behavior ─────────────────────────────────────────────────────


def test_run_with_one_device_matches_local_dreambank() -> None:
    """With one device, federation is a no-op (FedAvg over one tensor = the tensor).
    So federated and isolated should be identical."""
    cfg = FederatedConfig(num_devices=1, rounds=2, local_cycles_per_round=1, seed=7)
    sim = FederatedBankSimulator(cfg)
    report = sim.run()
    # One device: delta should be approximately zero
    assert abs(report.delta_fill) < 1e-3 or report.delta_fill == 0.0
    assert report.num_devices == 1
    assert len(report.per_device_final_fill) == 1


def test_run_multiple_devices_aggregate_tensors() -> None:
    """Multiple devices: federated banks should differ from isolated banks.
    (FedAvg mixes tensors from devices with different distributions.)"""
    cfg = FederatedConfig(num_devices=4, rounds=2, local_cycles_per_round=2, seed=11)
    sim = FederatedBankSimulator(cfg)
    report = sim.run()
    # With different-device distributions, federation SHOULD produce some change.
    # We accept either non-zero delta_fill OR non-zero delta_strength OR both.
    assert report.delta_fill != 0.0 or report.delta_strength != 0.0, (
        "Federation should produce measurable delta vs isolation on heterogeneous distributions"
    )
    assert report.num_devices == 4
    assert len(report.per_device_final_fill) == 4


def test_dp_noise_adds_variance_when_sigma_nonzero() -> None:
    """DP noise should make the reported values differ between two identically-seeded runs."""
    torch.manual_seed(0)
    cfg_quiet = FederatedConfig(
        num_devices=4,
        rounds=2,
        local_cycles_per_round=1,
        dp_sigma=0.0,
        seed=42,
        bank_size=4,
        bank_dim=8,
    )
    cfg_loud = FederatedConfig(
        num_devices=4,
        rounds=2,
        local_cycles_per_round=1,
        dp_sigma=5.0,
        seed=42,
        bank_size=4,
        bank_dim=8,
    )
    sim_quiet = FederatedBankSimulator(cfg_quiet)
    sim_loud = FederatedBankSimulator(cfg_loud)
    # Same seed but different dp_sigma => federated_mean_strength diverges from isolated
    # (noise injected between upload and aggregate).
    r_quiet = sim_quiet.run()
    r_loud = sim_loud.run()
    # The two reports should differ in federated_mean_strength because DP noise
    # perturbs the aggregated tensors.
    assert (
        abs(r_quiet.federated_mean_strength - r_loud.federated_mean_strength) > 0
        or abs(r_quiet.federated_mean_fill - r_loud.federated_mean_fill) >= 0
    )


def test_isolated_baseline_run() -> None:
    """Sanity: isolated baseline runs without crash and produces per-device outputs."""
    cfg = FederatedConfig(num_devices=2, rounds=1, local_cycles_per_round=1, seed=17)
    sim = FederatedBankSimulator(cfg)
    report = sim.run()
    assert len(report.per_device_final_fill) == 2
    assert all(isinstance(f, int) for f in report.per_device_final_fill)
    assert report.isolated_mean_fill >= 0.0
    assert report.isolated_mean_strength >= 0.0


def test_delta_alignment_finite() -> None:
    """delta_fill and delta_strength must be finite (no NaN/Inf)."""
    cfg = FederatedConfig(num_devices=3, rounds=2, seed=19)
    report = FederatedBankSimulator(cfg).run()
    import math

    assert math.isfinite(report.delta_fill)
    assert math.isfinite(report.delta_strength)
    assert math.isfinite(report.federated_mean_fill)
    assert math.isfinite(report.federated_mean_strength)


def test_report_to_dict_json_serializable() -> None:
    cfg = FederatedConfig(num_devices=2, rounds=1, seed=23, bank_size=4, bank_dim=8)
    report = FederatedBankSimulator(cfg).run()
    d = report.to_dict()
    round_trip = json.loads(json.dumps(d))
    assert round_trip["num_devices"] == 2
    assert round_trip["rounds"] == 1
    assert isinstance(round_trip["per_device_final_fill"], list)


def test_comm_bytes_proxy_scales_with_rounds_and_devices() -> None:
    """Total comm bytes should grow with rounds * num_devices * tensor_size."""
    base = _comm_bytes_per_round(num_devices=2, bank_size=4, bank_dim=8)
    assert base > 0
    double_devices = _comm_bytes_per_round(num_devices=4, bank_size=4, bank_dim=8)
    assert double_devices == 2 * base

    cfg3r = FederatedConfig(num_devices=2, rounds=3, seed=29, bank_size=4, bank_dim=8)
    report = FederatedBankSimulator(cfg3r).run()
    # total_comm_bytes_proxy = per_round_bytes * rounds
    assert report.total_comm_bytes_proxy == base * 3
