# AMC v2: GARB + H-MoE + CLX Integration Plan

> **For Hermes:** Use `subagent-driven-development` to implement tranche-by-tranche. Strict TDD: write the failing test first, run it, implement minimum passing code, re-run focused tests. Do not push.

**Goal:** Extend `AMCTransformer` with three coupled architectural upgrades — GARB (fast-weight associative bank), H-MoE (tier-aware expert routing conditioned on bank signal), and CLX (cross-layer modulation augmented by bank read) — without breaking any existing DreamBank, CascadeBank, or TrustRAG contracts.

**Design decision (locked):** CLX reads from GARB. CLX does not write to GARB. GARB is written only via the DreamBank debate/TrustRAG protocol at HOPE layers. The bank is preference-curated at all times; it shapes compute at inference, it does not receive inference-time writes.

**Architecture:** Three new modules slot into AMCTransformer behind config flags (`use_garb`, `use_hmoe`, `use_clx`). When all flags are `False`, behavior is byte-for-byte identical to AMC v1. Feature flags compose independently — each ablation row in the paper maps to a specific flag combination.

**Tech Stack:** Python 3.12, PyTorch (CPU for tests), pytest, existing `AMCTransformer` + `HLMPreferenceBank` + `CascadeRouter` + `SparseMoELayer`.

**Depends on:**
- DreamBank hardened: `3a759891`
- CascadeBank: `97ccc1dd`
- CB-06 harness: `72a99589`

---

## Current Verified Truth Surface

```bash
git -C /Users/christienantonio/aurelius log -1 --oneline
```

Expected HEAD:
```
46ee2f13 memory(debate): LLM voices for proposer/skeptic/judge via configurable API
```

Non-negotiable safety rules:
- Do not modify `src/memory/hlm_bank.py`, `src/model/hlm_bank_adapter.py`, `src/model/amc_transformer.py` (except the single integration tranche G-04), or any DreamBank/CascadeBank test.
- Stage only files listed in each tranche's "Stage only" section.
- Do not commit pre-existing dirty/untracked files unrelated to AMC v2.
- Do not push.

Frozen contracts (read-only from all tranches except G-04):
- `AMCModelOutput.bank_alpha: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_confidence: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_telemetry: dict[str, float | int] | None`
- `AMCTransformerConfig.use_hlm_bank: bool`
- `AMCTransformerConfig.hlm_bank_inject_scale: float`
- `AMCTransformerConfig.hlm_bank_read_layers: list[int] | None`
- `CascadeRouter.decision(bank_alpha, bank_confidence) -> ComputePolicy`

---

## Research Claim

AMC v2 is not three independent features. It is one system with three dimensions:

> **Preference-conditioned compute allocation:** A bank of alignment-curated memories (GARB) conditions both expert tier selection (H-MoE, via CascadePolicy) and cross-layer context propagation (CLX, via bank-augmented modulation vectors). The bank shapes compute; compute does not reshape the bank at inference time.

Prior art boundary:
- **GARB vs Titans/Larimar/Memorizing Transformers:** Prior work uses separate trained write networks. GARB's write gate *is* the DreamBank alignment signal — zero new write parameters.
- **H-MoE vs Switch/Mixtral:** Prior MoE systems route on token embedding similarity to expert centroids. H-MoE routes on memory alignment signal (`bank_alpha`, `bank_confidence`). First MoE design where routing is conditioned on an external preference bank.
- **CLX vs Feedback Transformer/CLA:** Prior cross-layer work uses backward attention (breaks causal decoding). CLX is forward-only (causal-safe), with a compact projection (d_clx ≪ d_model), bank-augmented at extraction points.

Minimum viable evidence from this plan:
1. Each module is independently tested and ablatable.
2. `use_garb=True, use_hmoe=False, use_clx=False` produces measurably different retrieval quality vs `HLMPreferenceBank`.
3. `use_hmoe=True` with `ComputePolicy.THOROUGH` activates a distinct expert pool vs `FAST`.
4. `use_clx=True` with bank augmentation (`b_l` nonzero) changes hidden states differently than raw CLX (`b_l = 0`).
5. All three enabled together: ablation harness produces all six rows of the paper table.
6. AMC v1 behavior preserved when all flags are `False`.

---

## Proposed File Map

Create:
- `src/memory/garb_memory.py` — fast-weight associative bank
- `src/model/clx_modulator.py` — CLX extraction + GARB-augmented broadcast
- `src/model/hmoe_layer.py` — tier-aware MoE routing
- `src/eval/amc_v2_ablation.py` — ablation harness (six rows)
- `tests/memory/test_garb_memory.py`
- `tests/model/test_clx_modulator.py`
- `tests/model/test_hmoe_layer.py`
- `tests/model/test_amc_v2_integration.py`
- `tests/eval/test_amc_v2_ablation.py`
- `docs/reports/AMC_V2_TRACEABILITY.md`

Modify:
- `src/model/amc_transformer.py` — add CLX extraction points, HMoE wiring, GARB swap, `clx_contexts` output (tranche G-04 only)
- `src/model/moe.py` — delegate to `HMoELayer` when `use_hmoe=True` (tranche G-04 only)
- `docs/MASTER-IMPLEMENTATION-PLAN.md` — mark AMC v2 as in-progress

Defer until after MVP:
- GARB persistence across sessions (serialization/deserialization of fast-weight matrix M).
- Per-layer H-MoE with different tier sizes per depth.
- CLX context bank (separate store for cross-turn context persistence — explicitly deferred, see design notes).
- Training loop changes for GARB/CLX/HMoE (this plan covers inference-time behavior only; training integration is a post-MVP tranche).

---

## Core API Contracts

### `src/memory/garb_memory.py`

```python
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GARBConfig:
    d_bank: int = 1024           # fast-weight matrix dimension
    d_query: int = 256           # query/key projection dim (matches d_clx)
    eta: float = 0.01            # Hebbian learning rate
    init_scale: float = 0.01     # M initialization scale (near-zero)
    normalize_keys: bool = True  # L2-normalize keys before Hebbian update


class GARBMemory(nn.Module):
    """Fast-weight associative bank. Drop-in upgrade for HLMPreferenceBank.

    Read path (inference-safe, called at every extraction point):
        output = softmax(W_q(query) @ M.T) @ M     Hopfield-style

    Write path (DreamBank protocol only — never called at inference):
        M ← M + eta * gate * (v - M@k) @ k.T       Hebbian delta, gated by alpha_t

    M is NOT an nn.Parameter. It is updated via write() only.
    """

    def __init__(self, config: GARBConfig | None = None) -> None: ...

    def retrieve(
        self,
        query: torch.Tensor,          # (B, T, d_query) or (B, d_query)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hopfield-style read.
        Returns:
            output: (B, T, d_bank) — retrieved memory
            alpha:  (B, T, 1)      — attention weight (becomes bank_alpha)
        Never raises. Returns zeros if M is near-zero initialized.
        """
        ...

    def write(
        self,
        key: torch.Tensor,             # (d_query,) or (B, d_query)
        value: torch.Tensor,           # (d_bank,) or (B, d_bank)
        gate: float | torch.Tensor = 1.0,  # alpha_t from DreamBank
    ) -> None:
        """Hebbian delta write. gate=0.0 is a no-op (safe to call unconditionally).
        Called only from DreamBank write path, never from forward pass.
        """
        ...

    def reset(self) -> None:
        """Re-initialize M to near-zero (init_scale). Called between sessions."""
        ...

    @classmethod
    def from_preference_bank(
        cls,
        bank: "HLMPreferenceBank",
        config: GARBConfig | None = None,
    ) -> "GARBMemory":
        """Upgrade path: seed M from existing HLMPreferenceBank slot contents."""
        ...
```

Invariants:
- `retrieve(zeros_like(q))` returns `(zeros, zeros)` — never raises on empty bank.
- `write(k, v, gate=0.0)` is a no-op — safe to call with gate from disabled DreamBank.
- `M` is not registered as a parameter — `model.parameters()` does not include it.
- `retrieve` is side-effect free — calling it never modifies M.

---

### `src/model/clx_modulator.py`

```python
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.memory.garb_memory import GARBMemory


@dataclass
class CLXConfig:
    d_model: int = 2048
    d_clx: int = 256             # compressed context dim (d_clx ≪ d_model)
    d_bank: int = 1024           # must match GARBConfig.d_bank
    extraction_interval: int = 4 # extract every N layers (matches agent_interval)
    enabled: bool = False


class CLXModulator(nn.Module):
    """Cross-layer modulation with GARB-augmented context vectors.

    extract() compresses h_l → c_l, then augments with GARB read → c_l'.
    modulate() applies Hadamard gate: h_i += W_out @ (c_l' ⊗ h_i).

    When garb is None or enabled=False, behaves as raw CLX (b_l = zeros).
    This enables the ablation: raw CLX vs GARB-augmented CLX.
    """

    def __init__(self, config: CLXConfig) -> None: ...

    def extract(
        self,
        hidden: torch.Tensor,            # (B, T, d_model) — layer l output
        garb: GARBMemory | None = None,
    ) -> torch.Tensor:
        """Extract and augment context vector.

        c_l  = LayerNorm(W_c @ h_l)          raw compression
        b_l  = garb.retrieve(c_l)[0]         bank read (zeros if garb is None)
        c_l' = c_l + W_b @ b_l               augment
        Returns c_l': (B, T, d_clx)
        """
        ...

    def modulate(
        self,
        hidden: torch.Tensor,            # (B, T, d_model) — layer i hidden state
        context: torch.Tensor,           # (B, T, d_clx) — c_l' from extract()
    ) -> torch.Tensor:
        """Apply Hadamard gating modulation.
        h_i' = h_i + W_out @ (c_l' ⊗ h_i)
        Returns h_i': (B, T, d_model)
        """
        ...

    def should_extract(self, layer_idx: int) -> bool:
        """True if layer_idx is an extraction point (layer_idx % extraction_interval == 3)."""
        ...
```

Invariants:
- `extract(h, garb=None)` is equivalent to `extract(h, garb)` when `garb.M` is all-zeros — ablation-safe.
- `modulate(h, zeros_like(c))` returns `h` unchanged (identity when context is zero).
- `should_extract(3)` → True, `should_extract(7)` → True, `should_extract(4)` → False (for interval=4).

---

### `src/model/hmoe_layer.py`

```python
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.inference.cascade_routing import ComputePolicy


@dataclass
class HMoEConfig:
    d_model: int = 2048
    d_ff_fast: int = 2816        # Tier 1 expert FFN dim  (0.5× standard)
    d_ff_balanced: int = 5632    # Tier 2 expert FFN dim  (1.0× standard)
    d_ff_thorough: int = 8448    # Tier 3 expert FFN dim  (1.5× standard)
    n_experts_fast: int = 2      # experts [0:2]
    n_experts_balanced: int = 3  # experts [2:5]
    n_experts_thorough: int = 3  # experts [5:8]
    top_k_fast: int = 1
    top_k_balanced: int = 2
    top_k_thorough: int = 2
    enabled: bool = False        # False → delegates to standard SparseMoELayer


class HMoELayer(nn.Module):
    """Tier-aware MoE. CascadePolicy selects expert pool; top-k within pool.

    FAST     → experts [0:2],   top-1, cheap d_ff
    BALANCED → experts [2:5],   top-2, standard d_ff
    THOROUGH → experts [5:8],   top-2, large d_ff

    policy=None → BALANCED (fail-safe, identical to standard routing).
    enabled=False → delegates entirely to wrapped SparseMoELayer (AMC v1 behavior).
    """

    def __init__(self, config: HMoEConfig) -> None: ...

    def forward(
        self,
        hidden: torch.Tensor,            # (B, T, d_model)
        policy: ComputePolicy | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output: (B, T, d_model), aux_loss: scalar)."""
        ...

    def get_active_tier(self, policy: ComputePolicy | None) -> str:
        """Returns 'fast' | 'balanced' | 'thorough'. Pure function."""
        ...
```

Invariants:
- `forward(h, policy=None)` produces output equivalent to `forward(h, ComputePolicy.BALANCED)`.
- `forward(h, policy=ComputePolicy.FAST)` only activates experts [0:2] — never touches experts [2:8].
- `enabled=False` → `forward` delegates to wrapped `SparseMoELayer`, policy arg ignored.
- `get_active_tier` never raises, always returns one of the three string literals.

---

### AMCTransformer changes (tranche G-04 only)

New config fields added to `AMCTransformerConfig`:
```python
use_garb: bool = False
use_hmoe: bool = False
use_clx: bool = False
clx_d_clx: int = 256
clx_extraction_interval: int = 4
hmoe_d_ff_fast: int = 2816
hmoe_d_ff_thorough: int = 8448
```

New `AMCModelOutput` field:
```python
clx_contexts: list[torch.Tensor] | None = None  # one per extraction point
```

No existing fields removed or type-changed.

---

### `src/eval/amc_v2_ablation.py`

```python
@dataclass
class AMCV2AblationConfig:
    use_garb: bool
    use_hmoe: bool
    use_clx_raw: bool        # CLX without GARB augment (b_l forced to zero)
    use_clx_aug: bool        # CLX with GARB augment
    label: str               # e.g. "baseline", "garb_only", "full_v2"

@dataclass
class AMCV2AblationResult:
    label: str
    retrieval_cosine_sim: float    # GARB vs HLMPreferenceBank on held-out queries
    expert_tier_distribution: dict[str, int]   # {"fast": N, "balanced": N, "thorough": N}
    clx_hidden_delta_norm: float   # ||h_with_clx - h_without_clx||_F
    clx_bank_aug_delta_norm: float # ||h_clx_aug - h_clx_raw||_F  (0 if no bank)
    total_decisions: int

def run_amc_v2_ablation(
    model_config: AMCTransformerConfig,
    prompts: Sequence[torch.Tensor],
    bank: GARBMemory | HLMPreferenceBank | None = None,
    ablation_config: AMCV2AblationConfig | None = None,
) -> AMCV2AblationResult: ...
```

The six rows of the paper table map to six `AMCV2AblationConfig` instances:
```python
ABLATION_ROWS = [
    AMCV2AblationConfig(False, False, False, False, "baseline"),
    AMCV2AblationConfig(True,  False, False, False, "garb_only"),
    AMCV2AblationConfig(True,  True,  False, False, "garb_hmoe"),
    AMCV2AblationConfig(True,  False, True,  False, "garb_clx_raw"),
    AMCV2AblationConfig(True,  False, False, True,  "garb_clx_aug"),
    AMCV2AblationConfig(True,  True,  False, True,  "full_v2"),
]
```

---

## Tranches

### Tranche G-00: Pre-flight

**Objective:** Verify full existing suite is green before writing any new code.

**Files:** None.

**Commands:**
```bash
cd /Users/christienantonio/aurelius && git log -1 --oneline
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/memory/test_hlm_bank.py \
  tests/model/test_hlm_bank_adapter.py \
  tests/model/test_amc_transformer.py \
  tests/alignment/test_dreambank.py \
  tests/inference/test_cascade_routing.py \
  tests/inference/test_cascade_routing_integration.py \
  -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "
from src.memory.hlm_bank import HLMPreferenceBank
from src.model.amc_transformer import AMCTransformer, AMCModelOutput
from src.inference.cascade_routing import CascadeRouter, ComputePolicy
print('pre-flight imports OK')
"
```

**Acceptance criteria:**
- HEAD is `46ee2f13`.
- All focused tests pass.
- Import smoke prints `pre-flight imports OK`.
- No files changed.

**Commit:** None.

---

### Tranche G-01: GARBMemory core

**Objective:** Implement fast-weight associative bank with full TDD coverage of read/write/reset/upgrade-path contracts.

**Files:**
- Create: `src/memory/garb_memory.py`
- Create: `tests/memory/test_garb_memory.py`

**TDD tests to write first:**
- `test_garb_config_defaults_are_sane`
- `test_retrieve_on_empty_bank_returns_zeros_never_raises`
- `test_retrieve_output_shape_batched`
- `test_retrieve_output_shape_unbatched`
- `test_retrieve_returns_alpha_in_zero_one`
- `test_write_gate_zero_is_noop`
- `test_write_then_retrieve_increases_cosine_sim`
- `test_write_multiple_then_retrieve_nearest`
- `test_reset_clears_bank`
- `test_m_not_in_model_parameters`
- `test_from_preference_bank_upgrade_path_runs`
- `test_retrieve_is_side_effect_free`

**Validation commands:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_garb_memory.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/memory/garb_memory.py
```

**Acceptance criteria:**
- All 12 TDD tests pass.
- `M` not in `model.parameters()`.
- `retrieve` side-effect free.
- `write(gate=0.0)` is no-op.

**Commit message:** `feat(memory): add GARBMemory fast-weight associative bank`

**Stage only:**
```bash
git add src/memory/garb_memory.py tests/memory/test_garb_memory.py
```

---

### Tranche G-02: CLXModulator

**Objective:** Implement cross-layer modulation with GARB-augmented and raw modes, with full ablation-safe test coverage.

**Files:**
- Create: `src/model/clx_modulator.py`
- Create: `tests/model/test_clx_modulator.py`

**TDD tests to write first:**
- `test_clx_config_defaults_are_sane`
- `test_should_extract_correct_layers`
- `test_extract_without_garb_is_raw_clx`
- `test_extract_with_garb_differs_from_raw`
- `test_extract_output_shape`
- `test_modulate_output_shape`
- `test_modulate_zero_context_is_identity`
- `test_extract_garb_none_equals_extract_empty_garb`
- `test_modulate_nonzero_context_changes_hidden`
- `test_clx_disabled_config_passes_through`

**Validation commands:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_clx_modulator.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/model/clx_modulator.py
```

**Acceptance criteria:**
- All 10 TDD tests pass.
- `extract(h, garb=None)` == `extract(h, garb)` when garb.M is zero-initialized.
- `modulate(h, zeros)` returns `h` unchanged.

**Commit message:** `feat(model): add CLXModulator with GARB-augmented cross-layer context`

**Stage only:**
```bash
git add src/model/clx_modulator.py tests/model/test_clx_modulator.py
```

---

### Tranche G-03: HMoELayer

**Objective:** Implement tier-aware MoE routing conditioned on CascadePolicy, with full coverage of expert isolation and fail-safe behavior.

**Files:**
- Create: `src/model/hmoe_layer.py`
- Create: `tests/model/test_hmoe_layer.py`

**TDD tests to write first:**
- `test_hmoe_config_defaults_are_sane`
- `test_get_active_tier_fast`
- `test_get_active_tier_balanced`
- `test_get_active_tier_thorough`
- `test_get_active_tier_none_returns_balanced`
- `test_fast_policy_activates_only_fast_experts`
- `test_thorough_policy_activates_only_thorough_experts`
- `test_balanced_policy_activates_only_balanced_experts`
- `test_output_shape_all_policies`
- `test_aux_loss_is_scalar`
- `test_disabled_delegates_to_moe`
- `test_fast_policy_different_output_from_thorough`

**Validation commands:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_hmoe_layer.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/model/hmoe_layer.py
```

**Acceptance criteria:**
- All 12 TDD tests pass.
- FAST policy never touches experts [2:8] (verified via activation hook in test).
- `enabled=False` delegates to `SparseMoELayer` unchanged.
- `policy=None` is identical to `policy=BALANCED`.

**Commit message:** `feat(model): add HMoELayer with CascadePolicy tier-aware expert routing`

**Stage only:**
```bash
git add src/model/hmoe_layer.py tests/model/test_hmoe_layer.py
```

---

### Tranche G-04: AMCTransformer integration

**Objective:** Wire GARB, CLX, and H-MoE into AMCTransformer behind config flags. Add `clx_contexts` to `AMCModelOutput`. Verify AMC v1 behavior preserved when all flags are `False`.

**Files:**
- Modify: `src/model/amc_transformer.py`
- Modify: `src/model/moe.py`
- Create: `tests/model/test_amc_v2_integration.py`

**TDD tests to write first:**
- `test_all_flags_false_output_matches_v1_exactly`
- `test_use_garb_true_bank_alpha_from_garb_retrieve`
- `test_use_clx_true_clx_contexts_populated`
- `test_use_clx_false_clx_contexts_is_none`
- `test_use_hmoe_true_expert_tier_matches_policy`
- `test_use_hmoe_false_delegates_to_sparse_moe`
- `test_all_flags_true_full_forward_pass_runs`
- `test_all_flags_true_bank_alpha_confidence_shapes_unchanged`
- `test_clx_contexts_count_matches_extraction_points`
- `test_frozen_contracts_bank_alpha_confidence_telemetry_present`

**Implementation notes:**
- Add new config fields to `AMCTransformerConfig` (see API contract above).
- In `AMCTransformer.forward`: after each layer, check `clx.should_extract(layer_idx)` — if True, call `clx.extract(h, garb if use_garb else None)` and cache context. Apply `clx.modulate(h, context)` on subsequent layers until next extraction.
- In `AMCTransformer.forward`: pass `policy` to each `HMoELayer.forward` when `use_hmoe=True`. `policy` is derived from `CascadeRouter.decision(bank_alpha, bank_confidence)` — already computed during the bank read step.
- `AMCModelOutput`: add `clx_contexts: list[torch.Tensor] | None = None`. Default `None` preserves backward compat.

**Validation commands:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_amc_v2_integration.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/memory/test_hlm_bank.py \
  tests/model/test_hlm_bank_adapter.py \
  tests/model/test_amc_transformer.py \
  tests/alignment/test_dreambank.py \
  tests/inference/test_cascade_routing.py \
  -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig, AMCModelOutput
cfg = AMCTransformerConfig(vocab_size=32, d_model=16, n_layers=4, n_heads=4, kv_lrank=8, max_seq_len=16)
assert not cfg.use_garb and not cfg.use_hmoe and not cfg.use_clx
print('v1 defaults preserved')
"
```

**Acceptance criteria:**
- All 10 integration tests pass.
- Full DreamBank + CascadeBank focused suite still green.
- v1 defaults smoke prints `v1 defaults preserved`.
- `AMCModelOutput` has `clx_contexts` field but defaults to `None`.

**Commit message:** `feat(model): wire GARB + CLX + H-MoE into AMCTransformer behind config flags`

**Stage only:**
```bash
git add src/model/amc_transformer.py src/model/moe.py tests/model/test_amc_v2_integration.py
```

---

### Tranche G-05: Ablation harness

**Objective:** Implement the six-row ablation table. Prove each flag combination produces measurably distinct outputs on synthetic prompts.

**Files:**
- Create: `src/eval/amc_v2_ablation.py`
- Create: `tests/eval/test_amc_v2_ablation.py`

**TDD tests to write first:**
- `test_ablation_config_baseline_all_false`
- `test_all_six_rows_run_without_error`
- `test_baseline_and_full_v2_produce_different_outputs`
- `test_garb_only_retrieval_sim_greater_than_baseline`
- `test_fast_thorough_expert_distributions_differ`
- `test_clx_aug_delta_greater_than_clx_raw_delta`
- `test_ablation_result_is_json_serializable`
- `test_full_v2_all_metrics_populated`

**Validation commands:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/eval/test_amc_v2_ablation.py -q
```

**Acceptance criteria:**
- All 8 tests pass.
- `clx_bank_aug_delta_norm > 0` for `garb_clx_aug` row (proves GARB augments CLX).
- `retrieval_cosine_sim` is higher for `garb_only` than `baseline` on seeded bank.
- All results JSON-serializable.

**Commit message:** `feat(eval): add AMC v2 six-row ablation harness`

**Stage only:**
```bash
git add src/eval/amc_v2_ablation.py tests/eval/test_amc_v2_ablation.py
```

---

### Tranche G-06: Documentation and traceability

**Objective:** Record the claim, prior art boundary, evidence produced, and next steps. Update MASTER-IMPLEMENTATION-PLAN.

**Files:**
- Create: `docs/reports/AMC_V2_TRACEABILITY.md`
- Modify: `docs/MASTER-IMPLEMENTATION-PLAN.md` (mark Track A Phase in-progress, add CB-07 as unblocked)

Required content in traceability doc:
- Unified claim (preference-conditioned compute allocation).
- Prior art boundary table (GARB/H-MoE/CLX vs their respective prior works).
- Evidence produced: unit tests, integration tests, ablation rows.
- Not claimed: training loop integration (post-MVP), session persistence (deferred), real MT-Bench scores (CB-07).
- Next evidence needed: training with GARB write path active, GPU FLOPs per tier, quality measurement on real prompts.

**Commit message:** `docs: add AMC v2 traceability and update master plan`

**Stage only:**
```bash
git add docs/reports/AMC_V2_TRACEABILITY.md docs/MASTER-IMPLEMENTATION-PLAN.md
```

---

### Tranche G-07: Integration proof sweep

**Objective:** Full suite green after all AMC v2 files are added. No regressions in any existing test.

**Files:** None unless fixing regressions.

**Commands:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/memory/test_garb_memory.py \
  tests/model/test_clx_modulator.py \
  tests/model/test_hmoe_layer.py \
  tests/model/test_amc_v2_integration.py \
  tests/eval/test_amc_v2_ablation.py \
  tests/memory/test_hlm_bank.py \
  tests/model/test_hlm_bank_adapter.py \
  tests/model/test_amc_transformer.py \
  tests/alignment/test_dreambank.py \
  tests/inference/test_cascade_routing.py \
  tests/inference/test_cascade_routing_integration.py \
  tests/eval/test_cascadebank_ablation.py \
  -q

cd /Users/christienantonio/aurelius && .venv/bin/python -c "
from src.memory.garb_memory import GARBMemory, GARBConfig
from src.model.clx_modulator import CLXModulator, CLXConfig
from src.model.hmoe_layer import HMoELayer, HMoEConfig
from src.eval.amc_v2_ablation import run_amc_v2_ablation, ABLATION_ROWS
print('amc v2 imports OK')
"

cd /Users/christienantonio/aurelius && .venv/bin/python -m compileall -q \
  src/memory/garb_memory.py \
  src/model/clx_modulator.py \
  src/model/hmoe_layer.py \
  src/eval/amc_v2_ablation.py

cd /Users/christienantonio/aurelius && git diff --stat HEAD -- \
  src/memory/garb_memory.py \
  src/model/clx_modulator.py \
  src/model/hmoe_layer.py \
  src/model/amc_transformer.py \
  src/model/moe.py \
  src/eval/amc_v2_ablation.py \
  tests/
```

**Acceptance criteria:**
- Full AMC v2 + DreamBank + CascadeBank suite green (aim ≥ 60 tests).
- Import smoke prints `amc v2 imports OK`.
- Compileall clean.
- Diff contains only AMC v2 files + modified `amc_transformer.py`, `moe.py`.

**Commit:** If G-01 through G-06 were committed individually, no new commit. If single branch, use `feat: add AMC v2 (GARB + H-MoE + CLX)`.

---

## Done Criteria for MVP

AMC v2 MVP is complete when all are true:

```bash
# Full focused suite green
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/memory/test_garb_memory.py \
  tests/model/test_clx_modulator.py \
  tests/model/test_hmoe_layer.py \
  tests/model/test_amc_v2_integration.py \
  tests/eval/test_amc_v2_ablation.py \
  -q

# AMC v1 behavior preserved
cd /Users/christienantonio/aurelius && .venv/bin/python -c "
from src.model.amc_transformer import AMCTransformerConfig
cfg = AMCTransformerConfig(vocab_size=32, d_model=16, n_layers=4, n_heads=4, kv_lrank=8, max_seq_len=16)
assert not cfg.use_garb and not cfg.use_hmoe and not cfg.use_clx
print('v1 defaults preserved OK')
"

# Imports clean
cd /Users/christienantonio/aurelius && .venv/bin/python -c "
from src.memory.garb_memory import GARBMemory
from src.model.clx_modulator import CLXModulator
from src.model.hmoe_layer import HMoELayer
print('amc v2 imports OK')
"

# Pre-existing DreamBank + CascadeBank suite still green
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/alignment/test_dreambank.py \
  tests/inference/test_cascade_routing.py \
  -q
```

---

## Next Phase After MVP

Recommended order:

1. **Training integration** — wire GARB write path into the training loop so the fast-weight matrix actually learns from preference data (not just near-zero initialized at inference). This is the step that makes GARB more than a wrapper around HLMPreferenceBank.

2. **CB-07 GPU serving** — slot AMC v2 into the CB-06 harness, measure per-tier FLOPs and latency with real GPU. H-MoE FAST policy should show measurable FLOPs reduction vs THOROUGH.

3. **APEX integration** — APEX credit engine (Tier 2) produces hierarchical A(t) per token. A(t) is a natural write gate for GARB (stronger signal than simple alpha_t). Wire APEX Tier 2 output as GARB write gate once APEX trainer exists.

4. **Session persistence** — serialize/deserialize GARB fast-weight matrix M across sessions. Not needed for the paper claim; needed for production agent use.

---

## Invariants to Paste Into Any Implementation Agent Prompt

```text
AMC v2 invariants:
1. CLX reads from GARB. CLX never writes to GARB. Write path is DreamBank-only.
2. All three components (use_garb, use_hmoe, use_clx) default to False. AMC v1 behavior
   is preserved exactly when all flags are False.
3. GARBMemory.M is not an nn.Parameter. model.parameters() must not include it.
4. HMoELayer with policy=None behaves identically to policy=BALANCED.
5. Do not modify src/memory/hlm_bank.py, src/model/hlm_bank_adapter.py, or any
   DreamBank/CascadeBank source file. Those are upstream read-only contracts.
6. DreamBank + CascadeBank focused suite must pass unchanged after G-07.
7. Stage only AMC v2 files; do not bundle pre-existing dirty/untracked drift.
```

---

**Last updated:** 2026-05-28  
**Author:** Christien Antonio  
**Branch baseline:** `46ee2f13`  
**Design doc:** `docs/plans/2026-05-28-amc-v2-garb-hmoe-clx.md` (this file)
