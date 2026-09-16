# Federated Memory Deltas — Simulation Harness

> **For Hermes:** Implement inline, strict TDD. Do not push.

**Goal:** Demonstrate that federating DreamBank deltas (rather than weights) produces measurable alignment benefit at much lower communication cost than federated LLM fine-tuning.

**Architecture:** Pure-Python simulator of N device-local `HLMPreferenceBank` instances, each running DreamBank cycles on locally sampled preference distributions. Periodic FedAvg aggregation of bank tensors, with optional Gaussian DP noise on uploaded deltas.

**Tech Stack:** Python 3.12, PyTorch (CPU), pytest.

**Depends on:** DreamBank (`3a759891`), serving harness (`72a99589`).

---

## Truth surface

HEAD: `72a99589 eval(cb06): greedy decode + per-policy cost-proxy serving harness`
DreamBank suite: 100 tests green (DreamBank + CascadeBank + CB-06).

## Claim / not-claimed

**Claim:**
- Federated bank deltas at the DreamBank level are ~O(bank_size × bank_dim) tensors (~7 KB at 14 × 64 × fp16) — orders of magnitude smaller than federated weights.
- Aggregated banks outperform isolated banks and global-only banks on a proxy alignment metric (logit-margin mean over synthetic prompts).

**Not claimed:**
- Real device/communication simulation (we aggregate tensors in-process).
- Real DP privacy proofs (we add Gaussian noise with configurable sigma and check signal/noise tradeoff — not a formal proof).
- Real personalization quality on user data — synthetic preference distributions only.

## File map

Create:
- `src/memory/federated_banks.py` — `FederatedBankSimulator`, `FederatedConfig`, `FederationReport`.
- `tests/memory/test_federated_banks.py` — tests.

Modify:
- `docs/research-brief.md` — append section 8.

## API contract

```python
@dataclass(frozen=True)
class FederatedConfig:
    num_devices: int = 8
    rounds: int = 3
    local_cycles_per_round: int = 2
    dp_sigma: float = 0.0           # 0 = no DP noise
    aggregation: str = "fedavg"     # only fedavg supported in MVP
    seed: int = 0

@dataclass
class FederationReport:
    num_devices: int
    rounds: int
    per_device_final_fill: list[int]
    per_device_final_mean_strength: list[float]
    isolated_mean_logit_margin: float     # baseline: each device alone, no federation
    federated_mean_logit_margin: float    # after federation
    isolated_mean_compute_proxy: float
    federated_mean_compute_proxy: float
    delta_alignment: float                # federated - isolated
    total_comm_bytes_proxy: int           # bytes transmitted per round * rounds

    def to_dict(self) -> dict: ...


class FederatedBankSimulator:
    def __init__(self, config: FederatedConfig | None = None) -> None: ...

    def run(self) -> FederationReport:
        """
        1. Initialize num_devices local banks + preference distributions.
        2. For each round:
             a. Each device runs local DreamBank cycles on its own distribution.
             b. Each device uploads bank tensors (keys, values, strengths).
             c. Server aggregates via FedAvg.
             d. Each device downloads merged tensors (overwrites local).
        3. Measure per-device logit margin over shared synthetic eval prompts
           with harness.
        4. Compare against isolated baseline (no federation).
        """
```

## Tranches

### FD-00: Pre-flight

Verify 100 tests still green at HEAD `72a99589`.

### FD-01: Core simulator + TDD

**Tests:**
- `test_federated_config_defaults`
- `test_federated_config_rejects_invalid`
- `test_run_with_one_device_matches_local_dreambank`
- `test_run_multiple_devices_aggregate_tensors`
- `test_dp_noise_adds_variance_when_sigma_nonzero`
- `test_isolated_baseline_run`
- `test_delta_alignment_finite`
- `test_report_to_dict_json_serializable`
- `test_comm_bytes_proxy_scales_with_rounds_and_devices`

### FD-02: Documentation

Append section 8 to `docs/research-brief.md`.

## Done criteria

- 9 new tests green.
- Combined suite = 100 + 9 = 109 tests green.
- `FederationReport.delta_alignment` is finite.
- No changes to DreamBank / CascadeBank / CB-06 files.
