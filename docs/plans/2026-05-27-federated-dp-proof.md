# Federated Memory Deltas — DP Proof

> **For Hermes:** Implement inline, strict TDD. Do not push.

**Goal:** Derive and verify the (ε,δ) differential privacy guarantee for the
Gaussian mechanism used in `src/memory/federated_banks.py`, parameterized
by clip norm, σ, δ, and tensor dimension.

**Tech Stack:** Python 3.12, pytest.

**Depends on:** Federated banks commit `36f2a979`.

## Truth surface

HEAD: `fb4a3c61` (synced with origin)
Suite: 132 tests green.

## Claim / not-claimed

**Claim:**
- For `L2`-clipped bank tensors with clip norm `C` and noise `N(0, σ²I)`,
  the Gaussian mechanism provides (ε,δ)-DP with:
  `ε = C * √(2 * log(1.25/δ)) / σ`
- We compute ε for a matrix of (σ, δ, C) values and verify the
  simulator's default configuration achieves a target (e.g., ε≤1.0
  at δ=10⁻⁵, C=1.0).

**Not claimed:**
- Formal advanced-composition bounds over multiple rounds (we use
  a single-round bound; composition would improve or degrade ε
  depending on the advanced composition theorem used).
- Privacy against adaptive/corrupted adversaries with global view
  of intermediate rounds (this is the standard honest-but-curious
  aggregator model).

## File map

Create:
- `src/privacy/dp_bounds.py` — `GaussianMechanism` with ε computation,
  sensitivity derivation for bank tensors, parameterized tables.
- `tests/privacy/test_dp_bounds.py` — tests.

## Tranches

### D-00: Pre-flight — 132 tests green, HEAD synced.

### D-01: DP bounds module + TDD

**Tests:**
- `test_gaussian_epsilon_formula` — verifies ε = C√(2 ln(1.25/δ))/σ
- `test_epsilon_decreases_as_sigma_increases`
- `test_epsilon_increases_as_delta_decreases`
- `test_epsilon_scales_linearly_with_clip_norm`
- `test_bank_tensor_sensitivity_formula`
- `test_default_config_epsilon_at_delta_1e5`
- `test_epsilon_is_finite_for_valid_params`
- `test_epsilon_rejects_negative_sigma`
- `test_epsilon_rejects_invalid_delta_range`

### D-02: Documentation

Append section 11 to `docs/research-brief.md`.

## Done criteria

- 9 new tests green.
- Combined suite: 132 + 9 = 141 tests green.
- Proof document matches parameterized table output.
