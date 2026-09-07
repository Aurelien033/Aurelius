"""Differential privacy bounds for the Gaussian mechanism used in
Federated Memory Deltas.

The Gaussian mechanism M(x) = x + N(0, σ²I) provides (ε,δ)-DP when
σ ≥ (Δ₂ * √(2 * log(1.25/δ))) / ε, where Δ₂ is the L2 sensitivity
of the function being privatized.

For our federated bank aggregation, each device uploads:
  - keys:   (bank_size, bank_dim) tensor
  - values: (bank_size, bank_dim) tensor
  - strengths: (bank_size,) tensor

After L2 clipping to norm C per device, the L2 sensitivity of the
per-device contribution to the mean over M devices is:
  Δ₂ = C / M

The ε for a given (C, σ, δ, M) is:
  ε = (C / M) * √(2 * log(1.25/δ)) / σ
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianMechanismConfig:
    """Parameters for the Gaussian mechanism applied to federated bank deltas."""

    clip_norm: float = 1.0  # L2 clipping norm per device (C)
    sigma: float = 1.0  # Noise scale (σ)
    delta: float = 1e-5  # Failure probability
    num_devices: int = 8  # M in the mean aggregation

    def __post_init__(self) -> None:
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {self.clip_norm}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not 0 < self.delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.num_devices < 1:
            raise ValueError(f"num_devices must be >= 1, got {self.num_devices}")


def compute_epsilon(
    clip_norm: float,
    sigma: float,
    delta: float,
    num_devices: int = 1,
) -> float:
    """Compute ε for the Gaussian mechanism at the given parameters.

    Formula: ε = (clip_norm / num_devices) * sqrt(2 * ln(1.25 / delta)) / sigma

    This is the single-round bound. For k rounds with composition, the
    advanced composition bound is tighter but more complex — see Dwork-Roth
    Algorithm 3.22 for the Gaussian mechanism composition theorem.
    """
    sensitivity = clip_norm / num_devices
    log_term = math.log(1.25 / delta)
    epsilon = sensitivity * math.sqrt(2 * log_term) / sigma
    return epsilon


def compute_sigma_for_target(
    clip_norm: float,
    target_epsilon: float,
    delta: float,
    num_devices: int = 1,
) -> float:
    """Compute the required σ to achieve ε ≤ target_epsilon."""
    sensitivity = clip_norm / num_devices
    log_term = math.log(1.25 / delta)
    sigma = sensitivity * math.sqrt(2 * log_term) / target_epsilon
    return sigma


def bank_tensor_element_count(bank_size: int, bank_dim: int) -> int:
    """Number of scalar elements in a full bank upload (keys + values + strengths)."""
    return 2 * bank_size * bank_dim + bank_size


def dp_parameterized_table(
    clip_norm: float = 1.0,
    num_devices: int = 8,
    deltas: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6),
    sigmas: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
) -> list[dict[str, float]]:
    """Return a matrix of ε values for all (σ, δ) combinations."""
    rows: list[dict[str, float]] = []
    for sigma in sigmas:
        row: dict[str, float] = {"sigma": sigma}
        for delta in deltas:
            row[f"eps_d{delta:.0e}"] = compute_epsilon(clip_norm, sigma, delta, num_devices)
        rows.append(row)
    return rows
