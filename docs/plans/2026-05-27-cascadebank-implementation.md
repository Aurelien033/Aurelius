# CascadeBank Implementation Plan

> **For Hermes:** Use `subagent-driven-development` to implement this plan task-by-task. Use strict TDD: write the failing test first, run it, implement the minimum code, then re-run focused tests. Do not push.

**Goal:** Add CascadeBank to Aurelius: a routing policy that reads the DreamBank adapter's per-token `alpha_t` and `confidence_t` (already emitted by `AMCTransformer` when `use_hlm_bank=True` and a bank is passed) and returns a `ComputePolicy` decision — `fast`, `balanced`, or `thorough` — without touching model weights or introducing new parameters.

**Architecture:** Pure-function routing module. Input: per-token gate mean + confidence (from `AMCModelOutput.bank_alpha`, `bank_confidence`). Output: `ComputePolicy` enum. Thresholds are configurable via a frozen dataclass. No model surgery, no new modules in the forward pass, no latency overhead.

**Tech Stack:** Python 3.12, pytest, existing `src/model/amc_transformer.py` (read-only consumer), existing `src/memory/hlm_bank.py` (read-only consumer).

**Depends on:** DreamBank MVP commit `3a759891` — the fixed adapter/transformer surfaces (use_hlm_bank gate, alpha/confidence emission, top_k shape padding) must be intact.

---

## Current Verified Truth Surface

Verified in `/Users/christienantonio/aurelius` on 2026-05-27:

```bash
git -C /Users/christienantonio/aurelius log -1 --oneline
```

Observed HEAD:

```text
3a759891 (HEAD -> clean/amc-curation-20260521-101220) fix: harden DreamBank MVP contracts (gate, metadata, gradients, shape, restore)
```

Non-negotiable safety rules:
- Do not stage or commit any pre-existing dirty/untracked files unrelated to CascadeBank.
- Do not push.
- Stage only CascadeBank files listed in each task.
- Do not modify `src/memory/hlm_bank.py`, `src/model/hlm_bank_adapter.py`, `src/model/amc_transformer.py`, or any DreamBank test — those are upstream and were just fixed in the previous tranche.
- If the working tree changes unexpectedly, stop and report.

Existing integration points (READ-ONLY):
- `AMCModelOutput.bank_alpha: torch.Tensor | None` — shape `(B, T, 1)` when `use_hlm_bank=True`
- `AMCModelOutput.bank_confidence: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_telemetry: dict[str, float | int] | None`
- `AMCTransformerConfig.use_hlm_bank: bool`
- `AMCTransformerConfig.hlm_bank_inject_scale: float`

---

## Research Claim Being Built

CascadeBank is not "another mixture-of-depths router." Mixture-of-Depths trains a separate per-token router with its own parameters. CascadeBank REUSES the existing DreamBank gate signal.

The defensible claim is narrower and cleaner:

> A routing policy built on top of an already-existing alignment gate (DreamBank `alpha_t`) can produce measurable quality/compute tradeoffs without adding any trainable parameters or any new forward-pass cost.

Minimum viable evidence:
1. Router maps `(alpha_mean, confidence)` → `ComputePolicy` deterministically.
2. Router is threshold-configurable via frozen dataclass; defaults produce all three policy classes on synthetic input.
3. Router handles missing DreamBank signals (bank disabled / empty bank) by defaulting to `balanced` (fail-safe).
4. Router runs in O(1) per-token, no heap allocations in hot path.
5. Ablation shows: with a non-trivial bank, the router distributes decisions across all three classes; with an empty bank, every decision falls back to `balanced`.

---

## Proposed File Map

Create:
- `src/inference/cascade_routing.py` — pure-function router + `ComputePolicy` enum + `CascadeRouterConfig`.
- `tests/inference/test_cascade_routing.py`

Modify:
- None. CascadeBank is fully additive.

Defer until after MVP:
- Wiring the router into the actual serving layer (that requires choosing an inference harness, which is bigger scope).
- Training the bank to produce router-friendly signal distributions.
- Per-layer router (requires per-layer DreamBank wiring, which is post-MVP).
- Real-world FLOPs/latency measurement (requires GPU serving).

---

## Core API Contract

### `src/inference/cascade_routing.py`

Required public surface:

```python
from enum import Enum
from dataclasses import dataclass

import torch


class ComputePolicy(str, Enum):
    """Coarse inference-compute budget decision."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass(frozen=True)
class CascadeRouterConfig:
    """Routing thresholds. Defaults chosen so balanced is the fail-safe class."""
    # If max(alpha_t) over the prompt >= alpha_thorough, escalate.
    alpha_thorough: float = 0.70
    # If mean(confidence_t) over the prompt < confidence_fast, downgrade.
    confidence_fast: float = 0.40
    # Require both signals present to upgrade to THOROUGH; otherwise stay at current level.
    require_confidence_for_thorough: bool = True
    # Minimum mean confidence to stay BALANCED when alpha is ambiguous.
    balanced_confidence_floor: float = 0.30
    # If DreamBank is disabled/empty, always return this policy (fail-safe).
    fallback: ComputePolicy = ComputePolicy.BALANCED


class CascadeRouter:
    def __init__(self, config: CascadeRouterConfig | None = None) -> None: ...

    def decision(
        self,
        bank_alpha: torch.Tensor | None,
        bank_confidence: torch.Tensor | None,
    ) -> ComputePolicy:
        """Pure-function routing. O(1) per-token reduction. Never raises.

        Args:
            bank_alpha: shape (B, T, 1) — per-token alignment gate from DreamBank.
            bank_confidence: shape (B, T, 1) — per-token confidence from DreamBank.

        Returns:
            ComputePolicy. Falls back to `fallback` when either input is None.
        """
        ...

    def telemetry(self, histories: "list[ComputePolicy] | None" = None) -> dict[str, float | int]:
        """Class distribution over a history of decisions. Pure function."""
        ...
```

Mechanism:

```text
if bank_alpha is None or bank_confidence is None:
    return fallback

alpha_max = bank_alpha.max().item()
conf_mean = bank_confidence.mean().item()

# Step 1: fast gate — low confidence is strong signal to downgrade
if conf_mean < config.confidence_fast:                # default: 0.40
    return FAST

# Step 2: thorough gate — high alpha AND strong confidence
if alpha_max >= config.alpha_thorough:                # default: 0.70
    if config.require_confidence_for_thorough:
        if conf_mean >= config.balanced_confidence_floor:
            return THOROUGH
    else:
        return THOROUGH

# Default: balanced
return BALANCED
```

Invariants:
- `decision(None, *)` and `decision(*, None)` always return `fallback`.
- `decision(zeros_like, zeros_like)` → `FAST` if `conf_mean < confidence_fast`.
- Result is a `ComputePolicy` enum, never raises, never allocates per-call state.
- No trainable parameters. No gradients. No side effects.

---

## Tranches

### Task CB-00: Pre-flight and baseline

**Objective:** Verify DreamBank MVP is still green before writing new code.

**Files:** None.

**Commands:**

```bash
cd /Users/christienantonio/aurelius && git log -1 --oneline
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py tests/scripts/test_run_dreambank_cycle.py tests/eval/test_dreambank_ablation.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "from src.memory.hlm_bank import HLMPreferenceBank; from src.model.hlm_bank_adapter import HLMPreferenceAdapter; from src.alignment.dreambank import DreamBankController; print('dreambank imports OK')"
```

**Acceptance criteria:**
- HEAD is still `3a759891`.
- DreamBank focused suite green.
- DreamBank import smoke green.
- No files changed.

**Commit:** None.

---

### Task CB-01: CascadeRouter core + deterministic routing

**Objective:** Implement the pure-function router with full TDD coverage of the threshold table.

**Files:**
- Create: `src/inference/cascade_routing.py`
- Create: `tests/inference/test_cascade_routing.py`

**TDD tests to write first:**
- `test_router_default_config_exists_with_sane_thresholds`
- `test_router_missing_alpha_falls_back`
- `test_router_missing_confidence_falls_back`
- `test_router_low_confidence_returns_fast`
- `test_router_high_alpha_with_sufficient_confidence_returns_thorough`
- `test_router_high_alpha_with_low_confidence_returns_balanced_when_required`
- `test_router_ambiguous_input_returns_balanced`
- `test_router_zero_tensors_returns_fast`
- `test_router_batched_input_reduces_correctly`
- `test_telemetry_counts_distribution`

**Implementation steps:**
1. Write the tests above as failing tests.
2. Run and verify RED:
   ```bash
   cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/inference/test_cascade_routing.py -q
   ```
3. Implement `CascadeRouter` per the API contract above.
4. Re-run focused tests.
5. Refactor only if needed.

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/inference/test_cascade_routing.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/inference/cascade_routing.py
```

**Acceptance criteria:**
- All 10 TDD tests pass.
- Router is parameter-free, gradient-free, fail-safe.
- `telemetry()` handles empty history without raising.

**Commit message:** `feat(inference): add CascadeBank router on DreamBank gate signals`

**Stage only:**
```bash
git add src/inference/cascade_routing.py tests/inference/test_cascade_routing.py
```

---

### Task CB-02: Integration probe with AMCModelOutput shape contract

**Objective:** Prove the router correctly consumes live `bank_alpha`/`bank_confidence` shapes emitted by `AMCTransformer` (B, T, 1).

**Files:**
- Create: `tests/inference/test_cascade_routing_integration.py`

Note: this is an INTEGRATION test, separate from the unit tests in CB-01. It uses the real `HLMPreferenceBank` + `AMCTransformer` + `CascadeRouter` together.

**TDD tests to write first:**
- `test_router_consumes_real_bank_alpha_confidence_shape`
- `test_router_empty_bank_falls_back_to_balanced`
- `test_router_nonempty_bank_produces_nontrivial_decision`

**Test skeleton:**
```python
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig
from src.inference.cascade_routing import CascadeRouter, ComputePolicy

def test_router_consumes_real_bank_alpha_confidence_shape():
    cfg = AMCTransformerConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4,
                                kv_lrank=8, max_seq_len=16, use_hlm_bank=True)
    model = AMCTransformer(cfg).eval()
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=8))
    # populate bank with a few random slots
    import torch
    for _ in range(3):
        bank.upsert(HLMPreferenceWrite(key=torch.randn(8), value=torch.randn(8), strength=1.0))
    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out = model(ids, preference_bank=bank)
    assert out.bank_alpha is not None and out.bank_confidence is not None
    decision = CascadeRouter().decision(out.bank_alpha, out.bank_confidence)
    assert isinstance(decision, ComputePolicy)
```

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/inference/test_cascade_routing_integration.py -q
```

**Acceptance criteria:**
- Integration test consumes real `AMCModelOutput` shape.
- Empty-bank case falls back to `BALANCED`.
- Populated-bank case produces any of `{FAST, BALANCED, THOROUGH}` (decision is `isinstance(ComputePolicy)`).

**Commit message:** `test(inference): add CascadeBank × DreamBank integration probe`

**Stage only:**
```bash
git add tests/inference/test_cascade_routing_integration.py
```

---

### Task CB-03: Ablation harness

**Objective:** Demonstrate the router distributes decisions across all three classes on a non-trivial bank, and collapses to `balanced` on an empty bank.

**Files:**
- Create: `src/eval/cascadebank_ablation.py`
- Create: `tests/eval/test_cascadebank_ablation.py`

**Ablation contract:**
```python
@dataclass
class CascadeAblationResult:
    decision_distribution: dict[str, int]   # {"fast": N, "balanced": N, "thorough": N}
    total_decisions: int
    alpha_mean: float
    confidence_mean: float
    bank_fill: int

def run_cascade_ablation(
    model: AMCTransformer,
    prompts: Sequence[torch.Tensor],
    bank: HLMPreferenceBank | None = None,
    router: CascadeRouter | None = None,
) -> CascadeAblationResult: ...
```

**TDD tests to write first:**
- `test_empty_bank_collapse_to_balanced_only`
- `test_filled_bank_spreads_across_classes` (requires seeded bank with hand-tuned signal distributions so this is deterministic — acceptable to use a custom router config with extreme thresholds in the test)
- `test_ablation_result_is_json_serializable`

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/eval/test_cascadebank_ablation.py tests/inference/test_cascade_routing.py -q
```

**Acceptance criteria:**
- Empty bank → `decision_distribution == {"fast": 0, "balanced": N, "thorough": 0}`.
- Filled bank with a custom router config produces at least 2 distinct classes on the seeded input.
- Result dataclass serializes to JSON-safe dict.

**Commit message:** `feat(eval): add CascadeBank ablation harness`

**Stage only:**
```bash
git add src/eval/cascadebank_ablation.py tests/eval/test_cascadebank_ablation.py
```

---

### Task CB-04: Documentation and research traceability

**Objective:** Record the claim, evidence, and next steps. Do not re-hash DreamBank traceability — link to it.

**Files:**
- Modify: `docs/research-brief.md`
- Create: `docs/reports/CASCADEBANK_MVP_TRACEABILITY.md`

**Required content in traceability doc:**
- Claim: routing on existing alignment gate signals produces measurable compute decisions without new parameters.
- Not claimed: a mixture-of-depths router, a trained router, a full inference harness.
- Prior art boundary: Mixture-of-Depths (Raposo et al. 2024), LLMCascade, AdaLLM — all of those train new routers. CascadeBank reuses the DreamBank gate.
- Evidence produced by MVP: deterministic threshold tests, shape integration test, ablation.
- Next evidence needed: real serving-layer wiring, FLOPs/latency measurement on GPU, quality measurement (win rate vs iso-compute baseline).

**Commit message:** `docs: add CascadeBank MVP traceability`

**Stage only:**
```bash
git add docs/research-brief.md docs/reports/CASCADEBANK_MVP_TRACEABILITY.md
```

---

### Task CB-05: Integration proof sweep

**Objective:** Make sure the new module does not fracture imports or the DreamBank suite.

**Files:** None unless fixing regressions caused by CascadeBank.

**Commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/inference/test_cascade_routing.py tests/inference/test_cascade_routing_integration.py tests/eval/test_cascadebank_ablation.py tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py tests/scripts/test_run_dreambank_cycle.py tests/eval/test_dreambank_ablation.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "from src.inference.cascade_routing import CascadeRouter, ComputePolicy, CascadeRouterConfig; print('cascade imports OK')"
cd /Users/christienantonio/aurelius && .venv/bin/python -m compileall -q src/inference/cascade_routing.py src/eval/cascadebank_ablation.py
cd /Users/christienantonio/aurelius && git diff --stat HEAD -- src/inference src/eval/cascadebank_ablation.py tests/inference/test_cascade_routing.py tests/inference/test_cascade_routing_integration.py tests/eval/test_cascadebank_ablation.py
```

**Acceptance criteria:**
- Focused CB + DB suite green.
- Imports green.
- Compileall green.
- Diff contains only CascadeBank files.

**Commit:** If CB-01 through CB-04 were committed individually, no new commit. If done as a single branch commit, use `feat: add CascadeBank MVP`.

---

## Paper-Grade Evidence Target

After CB-05 is green, the next tranche (CB-06, not yet planned in this file) should add:

1. A real serving-layer harness that runs:
   - baseline greedy decode (no routing, no DreamBank)
   - DreamBank-enabled decode with balanced routing
   - DreamBank-enabled decode with CascadeBank FAST policy (skip expensive post-processing)
   - DreamBank-enabled decode with CascadeBank THOROUGH policy (e.g., chain-of-thought or self-consistency)
2. Metrics:
   - preference win rate (MT-Bench, AlpacaEval)
   - p50/p95 latency per-decision-class
   - estimated FLOPs savings at decision-distribution observed on real prompts
3. Paper-grade threshold:
   - CascadeBank THOROUGH improves preference win rate vs BALANCED by ≥ +3pp.
   - CascadeBank FAST degrades preference win rate by ≤ 1pp but saves ≥ 10% p95 latency.
4. If thresholds are not met, the failure mode is usually the gate signal — in which case the next tranche becomes "train the bank to produce router-friendly signal distributions" instead of wiring more serving code.

---

## Invariants to Paste Into Any Implementation Agent Prompt

```text
CascadeBank MVP invariants:
1. Zero new trainable parameters. The router is a pure function over DreamBank gate signals.
2. Missing DreamBank signals always produce the configured fallback (BALANCED by default).
3. No changes to src/memory/hlm_bank.py, src/model/hlm_bank_adapter.py, or src/model/amc_transformer.py.
4. Stage only CascadeBank files; do not bundle pre-existing dirty/untracked repo drift.
5. DreamBank-focused tests must still pass unchanged after CB-05.
```

---

## Done Criteria for MVP

CascadeBank MVP is complete when all are true:

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/inference/test_cascade_routing.py tests/inference/test_cascade_routing_integration.py tests/eval/test_cascadebank_ablation.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "from src.inference.cascade_routing import CascadeRouter, ComputePolicy, CascadeRouterConfig; print('cascade imports OK')"
cd /Users/christienantonio/aurelius && git diff --stat HEAD -- src/inference src/eval/cascadebank_ablation.py tests/inference/test_cascade_routing.py tests/inference/test_cascade_routing_integration.py tests/eval/test_cascadebank_ablation.py docs/reports/CASCADEBANK_MVP_TRACEABILITY.md
```

Expected:
- Focused CB suite green (aim for ≥ 15 tests).
- Import smoke prints `cascade imports OK`.
- Diff contains only CascadeBank files + DreamBank traceability doc update.
- Pre-existing DreamBank focused suite still green (CB-05 checks).

---

## Next Phase After MVP

Recommended order:
1. Real serving-layer harness (described above — CB-06).
2. Federated Preference Banks (simulate N local banks + DP/FedAvg merge — from the DreamBank plan's Phase E).
3. Per-layer MLA Bank (move final-only bank read into selected MLA layers after correctness is proven).

Recommended step: the serving-layer harness, because CascadeBank without an actual consumer has no measurable compute impact — it just becomes well-tested dead code.
