# Unified Implementation Plan: AMC v2 × Unified Provenance DAG

> **For Hermes:** AMC is the spine. Execute Track A first and do not start
> UPD implementation work until Gate G-A is green. UPD may be schema-reviewed
> on paper during Track A, but Track B/C/D/E code is downstream audit/provenance
> work, not a competing memory or control substrate. Use strict TDD on every
> tranche. Do not push.

**Spec IDs covered:** AMC-V2-2026-05-28, UPD-2026-05-28
**Supersedes:** `docs/plans/2026-05-28-amc-v2-garb-hmoe-clx.md` (replaced by this doc)
**UPD spec source:** `~/Desktop/AI:ML Research/SPEC-Unified-Provenance-DAG-2026-05-28.md`
**Date:** 2026-05-28
**Branch baseline:** `46ee2f13`
**Current verified truth surface (2026-05-29):**
- Target plan resolved to `/Users/christienantonio/aurelius/docs/plans/2026-05-28-amc-v2-upd-unified.md`; the similarly named `~/Desktop/Aurelius/...` path was not present.
- Git branch is `clean/amc-curation-20260521-101220`; this plan file is currently untracked. The working tree already contains unrelated modified/untracked files, so any future commit must stage only files listed by the active tranche.
- Live AMC contracts checked: `src/model/amc_transformer.py`, `src/memory/hlm_bank.py`, `src/alignment/dreambank.py`, and `src/inference/cascade_routing.py` exist.
- Current `AMCTransformerConfig.hlm_bank_read_layers` is `tuple[int, ...] | None`, not `list[int] | None`.
- Current `DreamBankController` exposes `run_cycle(...)` and writes via `HLMPreferenceBank.upsert(...)`; there is no `admit()` method in the live tree. Any plan text below that speaks of admission must normalize that existing write path first.

---

## 0. The Unified Thesis

A malware verdict, an AMC memory admission, and an agent decision are
**structurally the same object**: a typed, signed, directed acyclic graph of
evidence nodes feeding decision nodes, terminating in a verdict with a confidence
and a policy-bound action.

AMC v2 (GARB + H-MoE + CLX) and the Unified Provenance DAG (UPD) are not equal
peers in execution order. They are **two views of one system**, but only one is
the research spine:

- **AMC v2 is first.** It is the computational substrate: memory-conditioned
  compute allocation through AMC-native memory, routing, and context modulation.
- **UPD is second.** It is the audit layer: every consequential AMC decision can
  later become a replayable, signed, queryable DAG.
- **No UPD surface may redefine memory ownership.** UPD records AMC events; it
  does not create a second admission gate, second memory store, or second
  controller for promotion/quarantine.

The combination produces something neither plan achieves alone:

> **Every consequential decision Aurelius makes — "why remembered", "why this expert
> tier", "why this action" — is a replayable signed DAG that shares one explanation
> surface with the security verdict pipeline.**

---

## 1. The Five Coupling Points

These are the exact code locations where AMC v2 and UPD converge. Everything else
in both plans either leads toward or derives from these five points.

### CP-1: AMC memory write = MemoryAdmission DAG emission

AMC-first correction: the live tree currently has `DreamBankController.run_cycle(...)`
writing `HLMPreferenceWrite` records through `HLMPreferenceBank.upsert(...)`; it
does **not** have `DreamBankController.admit()`. Therefore CP-1 starts by
normalizing the existing write/upsert into a `MemoryAdmissionEvent`. If a later
`admit()` facade is introduced, it must delegate to this same event path.

When an AMC-native memory write occurs (`HLMPreferenceBank.upsert(...)` today,
`GARBMemory.write(key, value, gate=alpha_t)` after Track A), the following
structured values map directly to UPD nodes:

```
surprise_score (src/memory/amc_tier2.py)        → signal node
retrieval_hit (src/memory/trust_rag.py)          → detector node  (supports)
nli_contradiction (src/memory/trust_rag.py)      → detector node  (refutes)
debate_transcript (src/alignment/dreambank.py)   → deliberation node
trust_score_fusion                               → aggregator node
tier_promotion (Tier1|Tier2|Tier3|quarantine)   → policy node
ADMIT / QUARANTINE / REJECT verdict              → verdict node
```

Minimal instrumentation only: add an opt-in event capture at the existing
memory-write boundary, default `emit_upd=False`. If proposer/skeptic/judge
debate output is present in a future admission controller, the UPD deliberation
node stores the captured output hash, not raw transcript text (privacy-safe,
UPD §6.3).

`AMCMemoryCacheKey` provides `subject.id`. Memory revocation epoch maps to a
`revokes` edge in a new DAG (not an amendment of the old one — append-only).

### CP-2: CLX context = signal node in AgentDecision DAG

`AMCModelOutput.clx_contexts: list[Tensor]` (added in G-04) maps to:
- One `signal` node per extraction point (layer 3, 7, 11, …)
- The GARB read within each extraction maps to a `detector` node — "retrieval hit
  via GARB bank"
- `content_hash` = sha256(JCS(producer, layer_idx, c_l' tensor hash)) — volatile
  timing fields excluded

### CP-3: CascadePolicy = detector+policy in AgentDecision DAG

```
CascadeRouter.decision(bank_alpha, bank_confidence) → detector node
  output.label = "FAST" | "BALANCED" | "THOROUGH"
  output.score  = bank_alpha.max().item()
  output.uncertainty = 1 - bank_confidence.mean().item()
→ policy node  (triggers edge from detector)
  output.label = selected_tier
```

### CP-4: H-MoE tier selection = policy verdict in AgentDecision DAG

```
HMoELayer.get_active_tier(policy) → policy node
  output.label  = "tier_fast" | "tier_balanced" | "tier_thorough"
  output.payload = {n_experts: 2|3|3, d_ff: 2816|5632|8448}
→ verdict node  (decides edge from policy)
```

### CP-5: AMCTransformer.forward() = full AgentDecision DAG

The complete forward pass produces the AgentDecision DAG:

```
source   (input token ids)
  → signal  × N   (CLX contexts per extraction point)       CP-2
  → detector       (CascadePolicy from bank_alpha/conf)     CP-3
  → policy         (H-MoE tier selection)                   CP-4
  → verdict        (output action / chosen token sequence)  CP-4
  [→ oversight     (conformal abstain if CI crosses threshold — UPD §6.3)]
```

`AMCModelOutput` gets one new optional field: `upd_dag: UpdDag | None`. When
`emit_upd=False` (default), field is `None` — zero overhead.

---

## 2. Repository Layout (all in Aurelius)

```
aurelius/
├── crates/
│   ├── upd-core/               NEW  schema, JCS canon, hash, Ed25519, Merkle, store trait
│   ├── upd-cli/                NEW  verify / explain / replay / anchor
│   └── upd-store-sqlite/       NEW  append-only SQLite + node dedup
├── schema/
│   └── upd-v1.json             NEW  canonical JSON Schema (from UPD spec §3.4)
├── src/
│   ├── memory/
│   │   └── garb_memory.py      NEW  fast-weight associative bank      (Track A)
│   ├── model/
│   │   ├── clx_modulator.py    NEW  CLX extraction + GARB-augmented   (Track A)
│   │   └── hmoe_layer.py       NEW  tier-aware MoE routing            (Track A)
│   ├── upd/
│   │   ├── __init__.py         NEW  Python UPD SDK re-exports
│   │   ├── types.py            NEW  UpdDag, UpdNode, UpdEdge, UpdVerdict
│   │   ├── builder.py          NEW  DAGBuilder — construct + validate
│   │   ├── signer.py           NEW  Ed25519 sign/verify (wraps cryptography)
│   │   ├── explain.py          NEW  greedy backward sufficiency
│   │   ├── replay.py           NEW  determinism verify + LLM captured-output check
│   │   └── adapters/
│   │       ├── memory.py       NEW  MemoryAdmissionAdapter             (Track C)
│   │       ├── agent.py        NEW  AgentDecisionAdapter               (Track C)
│   │       └── security.py     NEW  SecurityVerdictAdapter             (Track D)
├── conformance/
│   ├── runner.py               NEW  conformance test runner
│   ├── cases/                  NEW  ≥30 test cases across 3 domains
│   └── golden/                 NEW  golden fixtures (3/domain)
├── middle/src/upd/             NEW  TypeScript SDK (dashboard + connectors)
└── schema/examples/            NEW  golden UPD fixture files
```

---

## 3. Frozen Contracts (do not touch in any tranche)

- `AMCModelOutput.bank_alpha: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_confidence: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_telemetry: dict[str, float | int] | None`
- `AMCTransformerConfig.use_hlm_bank: bool`
- `AMCTransformerConfig.hlm_bank_inject_scale: float`
- `AMCTransformerConfig.hlm_bank_read_layers: tuple[int, ...] | None`
- `CascadeRouter.decision(bank_alpha, bank_confidence) -> ComputePolicy`
- `src/memory/hlm_bank.py` — read-only
- `src/model/hlm_bank_adapter.py` — read-only
- `src/alignment/dreambank.py` — read-only except opt-in event capture / `emit_upd` wiring added in C-01

---

## 3.5 AMC-First Design Guardrails

This plan is improved by turning the coupling into a strict dependency graph, not
a parallel novelty buffet. The audit/provenance layer can only observe or verify
AMC decisions after the AMC substrate has proved it is useful on its own.

| Temptation | Verdict | AMC-first replacement |
|---|---|---|
| Build UPD foundation in parallel with AMC v2 | Reject for execution | Read/review the UPD schema on paper only; start UPD code after G-A |
| Let UPD own memory admission semantics | Reject | AMC owns admission/promotion/quarantine; UPD stores signed evidence after the decision |
| Add a separate GARB memory stack | Reject | GARB must wrap/upgrade existing AMC/HLM bank surfaces and preserve `AMCMemoryBlock`/`PromotionGate` lineage |
| Ship security connectors before memory evidence | Defer | First prove `Why remembered` and `Why this compute tier`; security verdicts are downstream reuse |
| Build dashboard/standards before conformance | Defer | A signed DAG without deterministic replay is just decorative JSON confetti |

**AMC-first acceptance rule:** Before any Track B/C/D/E implementation commit,
Track A must produce an AMC-only result row table showing that memory-conditioned
compute allocation has signal independent of provenance instrumentation. UPD can
then make that signal auditable; it cannot be the reason the signal exists.

---

## 4. Workstream Overview — AMC-first Critical Path

```text
Phase 0  Pre-flight truth check
   |
   v
Track A  AMC v2 substrate (A-00..A-05)
   |      GARB/HLM compatibility -> CLX -> H-MoE -> AMCTransformer -> AMC-only ablation
   |
   v
Gate G-A: AMC-only evidence exists and v1 defaults are preserved
   |
   v
Track B  UPD foundation (schema, Rust core, Python SDK)
   |      Audit engine only; no memory ownership
   |
   v
Track C  AMC x UPD adapters
   |      MemoryAdmission first, AgentDecision second, e2e third
   |
   v
Track D  Evidence reuse + connectors
   |      Combined ablation extends A-05; security connector is downstream
   |
   v
Track E  Standards + UI
          CLI, conformance, dashboard after replay/signature gates
```

**Execution rules:**
- Track A is serial and primary. Do not dispatch Track B implementation agents
  until Gate G-A is green.
- During Track A, UPD work is limited to paper review, schema comments, and
  non-committed interface notes. This prevents the audit layer from shaping the
  memory substrate prematurely.
- Track C requires all of A-01/A-02/A-03/A-04/A-05 green and B-02/B-03 green.
- Track D requires C-01/C-02/C-03 green.
- Track E requires D-01 green for CLI/conformance and D-03 green before dashboard
  integration.
- If an executor wants parallelism, parallelize tests/spec reviews inside Track A;
  do not parallelize a second architecture stack against it.

---

## 5. Track A — AMC v2 Core Modules (Primary Spine)

*This track runs before UPD implementation. Touches only new files plus the
explicit AMC integration files in A-04/A-05. It must produce AMC-only evidence
before any provenance layer is allowed to claim success.*

### Tranche A-00: Pre-flight

**Objective:** Verify all existing contracts green before any new code.

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

**Acceptance:** HEAD `46ee2f13`, all focused tests green, imports OK. No files changed.

**Commit:** None.

---

### Tranche A-00.5: AMC-Native Surface Audit

**Objective:** Make the implementation substrate explicit before adding new
modules. This is a documentation/truth-surface tranche, not runtime code.

**Files:** Create `docs/reports/amc-first-surface-audit.md`

**Audit checklist:**
- Record live constructor/API facts for `AMCTransformerConfig`,
  `AMCModelOutput`, `HLMPreferenceBank`, `DreamBankController`,
  `CascadeRouter`, `AMCMemoryBlock`, and `PromotionGate`.
- Decide the canonical memory-write event shape used by A-01 and C-01.
- State which existing surface owns each decision: memory admission, tier
  promotion, routing, quarantine, revocation, and provenance emission.
- Explicitly reject any second memory owner or UPD-first admission path.

**Acceptance:** Report exists, every new Track A module maps to an existing AMC
surface, and no source files are modified.

**Commit:** `docs(research): add AMC-first surface audit for AMC v2`

**Stage only:** `docs/reports/amc-first-surface-audit.md`

---

### Tranche A-01: GARBMemory

**Files:** `src/memory/garb_memory.py`, `tests/memory/test_garb_memory.py`

**API contract:**
```python
@dataclass
class GARBConfig:
    d_bank: int = 1024
    d_query: int = 256           # must match CLXConfig.d_clx
    eta: float = 0.01
    init_scale: float = 0.01
    normalize_keys: bool = True

class GARBMemory(nn.Module):
    def retrieve(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (output: (B,T,d_bank), alpha: (B,T,1)). Never raises."""
    def write(self, key: Tensor, value: Tensor, gate: float | Tensor = 1.0) -> None:
        """Hebbian delta update, gated by alpha_t. gate=0 is no-op."""
    def reset(self) -> None: ...
    @classmethod
    def from_preference_bank(cls, bank, config=None) -> "GARBMemory": ...
```

**TDD tests (write failing first):**
- `test_garb_config_defaults_are_sane`
- `test_retrieve_empty_bank_returns_zeros_never_raises`
- `test_retrieve_output_shape_batched`
- `test_retrieve_returns_alpha_in_zero_one`
- `test_write_gate_zero_is_noop`
- `test_write_then_retrieve_increases_cosine_sim`
- `test_write_multiple_retrieve_nearest`
- `test_reset_clears_bank`
- `test_m_not_in_model_parameters`
- `test_from_preference_bank_upgrade_path_runs`
- `test_retrieve_is_side_effect_free`

**Critical invariants:**
- `M` is NOT `nn.Parameter`. `model.parameters()` must not include it.
- GARB is an AMC-native bank view/upgrade path, not a competing memory
  subsystem. It must preserve HLM/DreamBank provenance fields (`provenance`,
  `trust`, `metadata_hash`) and expose deterministic conversion from existing
  `HLMPreferenceBank` state.

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_garb_memory.py -q
```

**Commit:** `feat(memory): add GARBMemory fast-weight associative bank`

**Stage only:** `src/memory/garb_memory.py tests/memory/test_garb_memory.py`

---

### Tranche A-02: CLXModulator

**Files:** `src/model/clx_modulator.py`, `tests/model/test_clx_modulator.py`

**API contract:**
```python
@dataclass
class CLXConfig:
    d_model: int = 2048
    d_clx: int = 256
    d_bank: int = 1024
    extraction_interval: int = 4
    enabled: bool = False

class CLXModulator(nn.Module):
    def extract(self, hidden: Tensor, garb: GARBMemory | None = None) -> Tensor:
        """c_l = LN(W_c @ h_l);  b_l = garb.retrieve(c_l)[0] or zeros;
           returns c_l' = c_l + W_b @ b_l  shape (B,T,d_clx)"""
    def modulate(self, hidden: Tensor, context: Tensor) -> Tensor:
        """h_i' = h_i + W_out @ (c_l' ⊗ h_i).  Returns (B,T,d_model)."""
    def should_extract(self, layer_idx: int) -> bool:
        """True at layer_idx % extraction_interval == (extraction_interval - 1)."""
```

**TDD tests:**
- `test_clx_config_defaults_are_sane`
- `test_should_extract_at_correct_layers`
- `test_extract_without_garb_is_raw_clx`
- `test_extract_with_garb_differs_from_raw`
- `test_extract_output_shape`
- `test_modulate_output_shape_preserved`
- `test_modulate_zero_context_is_identity`
- `test_extract_garb_none_equals_empty_garb`
- `test_modulate_nonzero_context_changes_hidden`

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_clx_modulator.py -q
```

**Commit:** `feat(model): add CLXModulator with GARB-augmented cross-layer context`

**Stage only:** `src/model/clx_modulator.py tests/model/test_clx_modulator.py`

---

### Tranche A-03: HMoELayer

**Files:** `src/model/hmoe_layer.py`, `tests/model/test_hmoe_layer.py`

**API contract:**
```python
@dataclass
class HMoEConfig:
    d_model: int = 2048
    d_ff_fast: int = 2816
    d_ff_balanced: int = 5632
    d_ff_thorough: int = 8448
    n_experts_fast: int = 2       # pool [0:2]
    n_experts_balanced: int = 3   # pool [2:5]
    n_experts_thorough: int = 3   # pool [5:8]
    top_k_fast: int = 1
    top_k_balanced: int = 2
    top_k_thorough: int = 2
    enabled: bool = False

class HMoELayer(nn.Module):
    def forward(self, hidden: Tensor, policy: ComputePolicy | None = None
                ) -> tuple[Tensor, Tensor]:  # (output, aux_loss)
    def get_active_tier(self, policy: ComputePolicy | None) -> str:
        """Pure function. Returns 'fast'|'balanced'|'thorough'. Never raises."""
```

**TDD tests:**
- `test_get_active_tier_all_three_policies`
- `test_get_active_tier_none_returns_balanced`
- `test_fast_activates_only_pool_0_2` (activation hook in test)
- `test_thorough_activates_only_pool_5_8`
- `test_disabled_delegates_to_sparse_moe`
- `test_output_shape_all_policies`
- `test_aux_loss_is_scalar`
- `test_fast_and_thorough_produce_different_outputs`
- `test_policy_none_equals_balanced`

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_hmoe_layer.py -q
```

**Commit:** `feat(model): add HMoELayer with CascadePolicy tier-aware expert routing`

**Stage only:** `src/model/hmoe_layer.py tests/model/test_hmoe_layer.py`

---

### Tranche A-04: AMCTransformer Integration

*This is the only tranche that touches frozen contract files. Requires A-01/02/03 green.*

**Files:** Modify `src/model/amc_transformer.py`, `src/model/moe.py`;
create `tests/model/test_amc_v2_integration.py`

**New config fields (additive only):**
```python
# AMCTransformerConfig — new fields, all default False/None
use_garb: bool = False
use_hmoe: bool = False
use_clx: bool = False
emit_upd: bool = False          # UPD DAG emission — wired in Track C
clx_d_clx: int = 256
clx_extraction_interval: int = 4
hmoe_d_ff_fast: int = 2816
hmoe_d_ff_thorough: int = 8448
```

**New AMCModelOutput field (additive only):**
```python
clx_contexts: list[torch.Tensor] | None = None
upd_dag: "UpdDag | None" = None   # populated when emit_upd=True; else None always
```

**TDD tests:**
- `test_all_flags_false_output_matches_v1_exactly`
- `test_use_garb_true_bank_alpha_comes_from_garb_retrieve`
- `test_use_clx_true_clx_contexts_populated`
- `test_use_clx_false_clx_contexts_is_none`
- `test_use_hmoe_true_tier_matches_policy`
- `test_use_hmoe_false_delegates_to_sparse_moe`
- `test_all_flags_true_forward_runs`
- `test_all_flags_true_bank_alpha_confidence_shapes_unchanged`
- `test_clx_context_count_matches_extraction_interval`
- `test_emit_upd_false_upd_dag_is_none`
- `test_frozen_output_fields_still_present`

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_amc_v2_integration.py -q
# + full DreamBank + CascadeBank suite must still pass
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/alignment/test_dreambank.py tests/inference/test_cascade_routing.py -q
```

**Commit:** `feat(model): wire GARB + CLX + H-MoE into AMCTransformer behind config flags`

**Stage only:** `src/model/amc_transformer.py src/model/moe.py tests/model/test_amc_v2_integration.py`

---

### Tranche A-05: AMC-Only Evidence Harness

**Objective:** Prove the AMC v2 mechanism has measurable signal before UPD is
implemented. This is the gate that makes the plan AMC-first instead of
provenance-first.

**Files:**
- Create: `src/eval/amc_v2_ablation.py`
- Create: `tests/eval/test_amc_v2_ablation.py`

**Six AMC-only ablation rows:**
```python
AMC_V2_ROWS = [
    AblationConfig(use_garb=False, use_hmoe=False, use_clx=False,
                   use_clx_aug=False, label="baseline"),
    AblationConfig(use_garb=True,  use_hmoe=False, use_clx=False,
                   use_clx_aug=False, label="garb_only"),
    AblationConfig(use_garb=True,  use_hmoe=True,  use_clx=False,
                   use_clx_aug=False, label="garb_hmoe"),
    AblationConfig(use_garb=True,  use_hmoe=False, use_clx=True,
                   use_clx_aug=False, label="garb_clx_raw"),
    AblationConfig(use_garb=True,  use_hmoe=False, use_clx=True,
                   use_clx_aug=True,  label="garb_clx_aug"),
    AblationConfig(use_garb=True,  use_hmoe=True,  use_clx=True,
                   use_clx_aug=True,  label="full_v2"),
]
```

**Result schema:**
```python
@dataclass
class AMCV2AblationResult:
    label: str
    retrieval_cosine_sim: float
    expert_tier_distribution: dict[str, int]
    clx_hidden_delta_norm: float
    clx_bank_aug_delta_norm: float
    total_decisions: int
    v1_defaults_preserved: bool
```

**TDD tests:**
- `test_all_six_amc_rows_run_without_error`
- `test_baseline_has_no_garb_alpha`
- `test_garb_only_improves_retrieval_metric_over_baseline`
- `test_garb_hmoe_changes_expert_tier_distribution`
- `test_clx_aug_delta_greater_than_raw_delta`
- `test_full_v2_preserves_output_shape_and_no_nan`
- `test_results_json_serializable`

**Acceptance:**
- `garb_clx_aug.clx_bank_aug_delta_norm > garb_clx_raw.clx_bank_aug_delta_norm`
- `full_v2` output is finite, shape-preserving, and non-identical to baseline under
  a deterministic probe fixture.
- This tranche produces the first paper-table cell for AMC v2 without using UPD.

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/eval/test_amc_v2_ablation.py -q
```

**Commit:** `feat(eval): add AMC v2-only ablation harness`

**Stage only:** `src/eval/amc_v2_ablation.py tests/eval/test_amc_v2_ablation.py`

---

## 6. Track B — UPD Foundation

*Starts only after Gate G-A. UPD is an audit/provenance foundation and must
not introduce memory, routing, or admission semantics that compete with AMC.*

### Tranche B-01: Schema + Golden Fixtures

**Files:**
- Create: `schema/upd-v1.json`
- Create: `schema/examples/memory_admission_golden.json`
- Create: `schema/examples/agent_decision_golden.json`
- Create: `schema/examples/security_verdict_golden.json`
- Create: `schema/examples/tamper_mutated_node.json`
- Create: `schema/examples/memory_revocation.json`

**Schema source:** UPD spec §3.4 (verbatim). Required fields: `upd_version`, `dag_id`,
`domain`, `created_at`, `nodes`, `edges`, `verdict_node_id`.

**TDD tests:** `tests/upd/test_schema_validation.py`
- `test_schema_validates_memory_golden_fixture`
- `test_schema_validates_agent_golden_fixture`
- `test_schema_validates_security_golden_fixture`
- `test_schema_rejects_missing_verdict_node_id`
- `test_schema_rejects_unknown_node_kind`
- `test_schema_rejects_unknown_edge_rel`
- `test_tamper_fixture_has_correct_structure` (used in later tamper tests)
- `test_revocation_fixture_has_revokes_edge`

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/upd/test_schema_validation.py -q
```

**Commit:** `feat(schema): add UPD v1 schema + 5 golden fixtures`

**Stage only:** `schema/ tests/upd/test_schema_validation.py`

---

### Tranche B-02: UPD Rust Core

**Files:** `crates/upd-core/` (new crate), `crates/upd-store-sqlite/` (new crate)

**Crate: `upd-core`** — pure library, no I/O

Key types and functions:
```rust
pub struct UpdNode { pub id: String, pub kind: NodeKind, pub producer: Producer,
                     pub content_hash: ContentHash, pub output: NodeOutput, ... }
pub struct UpdDag  { pub dag_id: DagId, pub domain: Domain, pub nodes: Vec<UpdNode>,
                     pub edges: Vec<UpdEdge>, pub verdict_node_id: String, ... }

pub fn canonicalize(dag: &UpdDag) -> Result<Vec<u8>>;       // RFC 8785 JCS
pub fn content_hash(node: &UpdNode) -> ContentHash;          // sha256 over JCS, volatile excluded
pub fn merkle_root(dag: &UpdDag) -> MerkleRoot;              // topological order
pub fn validate(dag: &UpdDag) -> Result<(), Vec<ValidationError>>;
  // checks: acyclic, single verdict, all edges reference valid node ids,
  //         content_hash consistency
pub fn sign(dag: &UpdDag, key: &Ed25519SigningKey) -> Signature;
pub fn verify(dag: &UpdDag, sig: &Signature, key: &Ed25519VerifyingKey) -> bool;
```

**Dependencies (add to workspace `Cargo.toml`):**
```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
ed25519-dalek = "2"
serde-jcs = "0.1"   # RFC 8785 JCS
```

**Crate: `upd-store-sqlite`** — append-only DAG store

```rust
pub struct SqliteStore { ... }
impl SqliteStore {
    pub fn new(path: &str) -> Result<Self>;
    pub fn insert(&self, dag: &UpdDag) -> Result<()>;    // append-only
    pub fn get(&self, dag_id: &str) -> Result<Option<UpdDag>>;
    pub fn by_subject(&self, kind: &str, id: &str) -> Result<Vec<UpdDag>>;
}
```

**Tests:** `crates/upd-core/tests/`
- `test_canonicalize_is_deterministic_1000_iterations`
- `test_content_hash_excludes_volatile_fields`
- `test_merkle_root_stable_under_node_reorder`
- `test_validate_rejects_cycle`
- `test_validate_rejects_missing_verdict`
- `test_sign_verify_roundtrip`
- `test_tamper_any_field_breaks_content_hash`
- `test_store_insert_then_get_roundtrip`
- `test_store_is_append_only` (second insert same dag_id returns error)

**Validation:**
```bash
cd /Users/christienantonio/aurelius && cargo test -p upd-core -p upd-store-sqlite 2>&1 | tail -5
```

**SC2 determinism test:**
```bash
cd /Users/christienantonio/aurelius && cargo test -p upd-core test_canonicalize_is_deterministic_1000_iterations -- --nocapture
```

**Commit:** `feat(crates): add upd-core and upd-store-sqlite`

**Stage only:** `crates/upd-core/ crates/upd-store-sqlite/ Cargo.toml Cargo.lock`

---

### Tranche B-03: Python UPD SDK

*Second reference engine (satisfies SC6 alongside Rust core).*

**Files:** `src/upd/__init__.py`, `src/upd/types.py`, `src/upd/builder.py`,
`src/upd/signer.py`, `src/upd/explain.py`, `src/upd/replay.py`,
`tests/upd/test_upd_python_sdk.py`

**Key classes:**

```python
# src/upd/types.py
@dataclass
class UpdNode:
    id: str                          # pattern: "^n[0-9]+$"
    kind: Literal["source","signal","detector","deliberation",
                  "aggregator","policy","verdict","oversight"]
    producer: Producer
    content_hash: str                # "sha256:<64hex>"
    output: NodeOutput
    input_refs: list[str] = field(default_factory=list)
    redaction: Literal["none","hash_only","encrypted"] = "none"

@dataclass
class UpdDag:
    upd_version: str = "1.0"
    dag_id: str                      # "upd_<32hex>"
    domain: Literal["security","memory","agent"]
    created_at: str
    nodes: list[UpdNode]
    edges: list[UpdEdge]
    verdict_node_id: str
    anchor: Anchor | None = None
    signatures: list[Signature] = field(default_factory=list)

# src/upd/builder.py
class DAGBuilder:
    def __init__(self, domain: str, subject: Subject) -> None: ...
    def add_node(self, kind: str, producer: Producer, output: NodeOutput,
                 input_refs: list[str] | None = None,
                 redaction: str = "none") -> str:
        """Adds node, computes content_hash, returns node_id."""
    def add_edge(self, from_id: str, to_id: str, rel: str,
                 weight: float | None = None) -> None: ...
    def build(self, verdict_node_id: str) -> UpdDag: ...

# src/upd/explain.py
def explain(dag: UpdDag) -> MinimalSubgraph:
    """Greedy backward sufficiency. Returns minimal subgraph + ranked 'because' list."""
```

**Python dependencies (add to `pyproject.toml`):**
```toml
cryptography = ">=42"   # Ed25519
jsonschema = ">=4.23"
```

**TDD tests:**
- `test_dag_builder_node_ids_sequential`
- `test_dag_builder_content_hash_deterministic`
- `test_dag_builder_volatile_fields_excluded_from_hash`
- `test_dag_builder_build_validates_against_schema`
- `test_signer_sign_verify_roundtrip`
- `test_explain_minimal_subgraph_size_leq_40pct_full`
- `test_explain_includes_all_refutes_edges`
- `test_explain_always_includes_oversight_node`
- `test_replay_deterministic_nodes_match`
- `test_replay_llm_nodes_use_captured_output_hash`
- `test_python_sdk_golden_fixture_matches_rust_root_hash`
  (loads `schema/examples/memory_admission_golden.json`, recomputes root hash,
   asserts == stored root hash — this is the SC6 cross-engine parity test)

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/upd/test_upd_python_sdk.py -q
```

**Commit:** `feat(upd): add Python UPD SDK (builder, signer, explain, replay)`

**Stage only:** `src/upd/ tests/upd/test_upd_python_sdk.py pyproject.toml`

---

## 7. Track C — Integration (AMC v2 × UPD)

*Requires Track A complete (A-01 through A-05) AND Track B-02/B-03 green.*

### Tranche C-01: MemoryAdmissionAdapter

**The central integration.** Wraps the AMC memory-write event stream so every
eligible AMC admission/write can emit a signed MemoryAdmission UPD DAG. CP-1.
The live source path is `DreamBankController.run_cycle(...)` ->
`HLMPreferenceBank.upsert(...)`; do not assume an `admit()` method exists unless
a preceding tranche adds it as a thin facade over the same event.

**Files:**
- Create: `src/upd/adapters/memory.py`
- Modify: `src/alignment/dreambank.py` (add `emit_upd: bool = False` to
  `DreamBankController` — the only modification to that file)
- Create: `tests/upd/test_memory_admission_adapter.py`

**Adapter contract:**
```python
class MemoryAdmissionAdapter:
    domain: Literal["memory"] = "memory"

    def to_dag(self, event: MemoryAdmissionEvent) -> UpdDag:
        """Pure, deterministic. Maps:
          event.surprise_score      → signal node
          event.retrieval_hits      → detector node (supports/refutes per TrustRAG)
          event.nli_contradiction   → detector node (refutes)
          event.debate_transcript   → deliberation node (LLM-kind, captured output hash)
          event.trust_score         → aggregator node
          event.tier_decision       → policy node
          event.verdict             → verdict node
        """

@dataclass
class MemoryAdmissionEvent:
    memory_key: str
    surprise_score: float
    retrieval_hits: list[dict]        # from TrustRAG
    nli_contradiction: bool
    debate_transcript_hash: str       # sha256 of captured debate output
    trust_score: float
    tier_decision: str                # "Tier1"|"Tier2"|"Tier3"|"quarantine"|"reject"
    verdict_label: str                # "ADMIT"|"QUARANTINE"|"REJECT"
    verdict_score: float
    model_version: str
```

**DreamBankController change (minimal):**
```python
# At the existing run_cycle()/upsert boundary, after a write is accepted.
# If an admit() facade is later added, it must call this same helper.
if self.emit_upd:
    event = MemoryAdmissionEvent(...)   # populate from HLMPreferenceWrite + AMC signals
    dag = MemoryAdmissionAdapter().to_dag(event)
    signed_dag = self._upd_signer.sign(dag)
    self._upd_store.insert(signed_dag)
```

**Privacy:** memory `source` nodes use `redaction="hash_only"` by default.
Raw memory text never enters the DAG. `subject.id = memory_key`.

**Revocation:** when a Tier-3 memory is revoked, a new DAG is created with a
`revokes` edge pointing at the original `verdict_node_id`. The original DAG is
never modified (append-only).

**TDD tests:**
- `test_adapter_memory_golden_fixture_validates_schema`
- `test_adapter_mapping_surprise_score_to_signal_node`
- `test_adapter_retrieval_hit_produces_supports_edge`
- `test_adapter_nli_contradiction_produces_refutes_edge`
- `test_adapter_debate_transcript_stored_as_captured_hash`
- `test_adapter_verdict_label_matches_tier_decision`
- `test_memory_write_with_emit_upd_false_produces_no_dag`
- `test_memory_write_with_emit_upd_true_produces_signed_dag`
- `test_revocation_creates_new_dag_with_revokes_edge`
- `test_source_node_redaction_is_hash_only`
- `test_determinism_same_event_same_dag_root_hash`

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/upd/test_memory_admission_adapter.py \
  tests/alignment/test_dreambank.py \
  -q
# DreamBank suite must still pass.
cd /Users/christienantonio/aurelius && .venv/bin/python -c "
from src.upd.adapters.memory import MemoryAdmissionAdapter
from src.alignment.dreambank import DreamBankController
print('memory adapter imports OK')
"
```

**Commit:** `feat(upd): add MemoryAdmissionAdapter + DreamBankController emit_upd flag`

**Stage only:**
```bash
git add src/upd/adapters/memory.py src/alignment/dreambank.py \
        tests/upd/test_memory_admission_adapter.py
```

---

### Tranche C-02: AgentDecisionAdapter

**Wires the full AMCTransformer forward pass into a UPD AgentDecision DAG. CP-2/3/4/5.**

**Files:**
- Create: `src/upd/adapters/agent.py`
- Create: `tests/upd/test_agent_decision_adapter.py`

**Adapter contract:**
```python
class AgentDecisionAdapter:
    domain: Literal["agent"] = "agent"

    def to_dag(self, event: AgentDecisionEvent) -> UpdDag:
        """Maps AMCModelOutput + agent step context → UPD DAG.

        source     (input_ids, redaction=hash_only)
        signal × N (clx_contexts[i], one per extraction layer)     CP-2
        detector   (CascadePolicy from bank_alpha/conf)            CP-3
        policy     (H-MoE tier from hmoe_tier)                     CP-4
        verdict    (chosen action / output label)
        [oversight if conformal_abstain triggered]
        """

@dataclass
class AgentDecisionEvent:
    step_id: str
    input_ids_hash: str               # sha256 of input token ids
    clx_contexts: list[torch.Tensor] | None
    bank_alpha: torch.Tensor | None
    bank_confidence: torch.Tensor | None
    cascade_policy: ComputePolicy | None
    hmoe_tier: str | None             # "fast"|"balanced"|"thorough"|None
    action_label: str                 # tool call name or "ABSTAIN"
    action_score: float
    conformal_abstain: bool = False
    model_version: str = ""
```

**AMCTransformer usage (emit_upd path):**
```python
# When emit_upd=True, in AMCTransformer.forward():
if self.config.emit_upd:
    event = AgentDecisionEvent(
        step_id=...,
        input_ids_hash=sha256_hex(input_ids),
        clx_contexts=output.clx_contexts,
        bank_alpha=output.bank_alpha,
        bank_confidence=output.bank_confidence,
        cascade_policy=cascade_policy,
        hmoe_tier=hmoe_tier_str,
        action_label=...,
        action_score=...,
    )
    output.upd_dag = AgentDecisionAdapter().to_dag(event)
```

**TDD tests:**
- `test_adapter_agent_golden_fixture_validates_schema`
- `test_clx_contexts_produce_signal_nodes`
- `test_clx_context_count_matches_signal_node_count`
- `test_cascade_policy_fast_maps_to_detector_node`
- `test_hmoe_tier_maps_to_policy_node`
- `test_policy_decides_verdict`
- `test_conformal_abstain_adds_oversight_node`
- `test_emit_upd_true_amc_output_has_upd_dag`
- `test_emit_upd_false_amc_output_upd_dag_is_none`
- `test_input_ids_stored_as_hash_not_raw`
- `test_determinism_same_step_same_dag_root_hash`

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/upd/test_agent_decision_adapter.py \
  tests/model/test_amc_v2_integration.py \
  tests/eval/test_amc_v2_ablation.py \
  -q
```

**Commit:** `feat(upd): add AgentDecisionAdapter wired into AMCTransformer forward`

**Stage only:**
```bash
git add src/upd/adapters/agent.py src/model/amc_transformer.py \
        tests/upd/test_agent_decision_adapter.py
```

---

### Tranche C-03: End-to-End Integration Probe

**Objective:** One real Aurelius memory admission produces a verifiable,
signed, explainable DAG end-to-end. This is the M2 milestone from the UPD spec.

**Files:**
- Create: `tests/upd/test_upd_e2e_integration.py`
- Create: `src/eval/upd_live_probe.py`

**Test skeleton:**
```python
def test_memory_admission_produces_verifiable_dag():
    # 1. Build AMCTransformer with emit_upd=True
    cfg = AMCTransformerConfig(..., use_garb=True, use_clx=True, use_hmoe=True,
                                emit_upd=True)
    model = AMCTransformer(cfg)
    bank = GARBMemory(GARBConfig())

    # 2. Run a debate cycle that produces ADMIT
    controller = DreamBankController(bank=bank, emit_upd=True,
                                     upd_store=InMemoryStore(),
                                     upd_signer=DevSigner())
    result = controller.admit(candidate=..., bank=bank)
    assert result.verdict == "ADMIT"
    assert result.upd_dag is not None

    # 3. Verify the DAG
    from src.upd.replay import replay
    report = replay(result.upd_dag)
    assert report.deterministic
    assert report.signature_ok

    # 4. Explain the DAG
    from src.upd.explain import explain
    subgraph = explain(result.upd_dag)
    assert len(subgraph.nodes) <= len(result.upd_dag.nodes) * 0.4  # SC4

    # 5. Agent step produces its own DAG
    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out = model(ids, preference_bank=bank)
    assert out.upd_dag is not None
    assert out.upd_dag.domain == "agent"
```

**Validation:**
```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest \
  tests/upd/test_upd_e2e_integration.py -q -v
```

**Commit:** `test(upd): add end-to-end integration probe (M2 milestone)`

**Stage only:**
```bash
git add tests/upd/test_upd_e2e_integration.py src/eval/upd_live_probe.py
```

---

## 8. Track D — Evidence + Connectors

*Requires Track C complete.*

### Tranche D-01: Combined AMC v2 × UPD Ablation Harness

**Extends A-05; does not replace it.** A-05 proves the AMC v2 compute-allocation
claim without provenance. D-01 reruns the same rows with UPD enabled to measure
auditability, replay determinism, explanation compactness, and overhead.

**Files:**
- Create: `src/eval/amc_v2_upd_ablation.py`
- Create: `tests/eval/test_amc_v2_upd_ablation.py`

**Six ablation rows:**
```python
ABLATION_ROWS = [
    AblationConfig(use_garb=False, use_hmoe=False, use_clx=False,
                   use_clx_aug=False, emit_upd=False, label="baseline"),
    AblationConfig(use_garb=True,  use_hmoe=False, use_clx=False,
                   use_clx_aug=False, emit_upd=True,  label="garb_only"),
    AblationConfig(use_garb=True,  use_hmoe=True,  use_clx=False,
                   use_clx_aug=False, emit_upd=True,  label="garb_hmoe"),
    AblationConfig(use_garb=True,  use_hmoe=False, use_clx=True,
                   use_clx_aug=False, emit_upd=True,  label="garb_clx_raw"),
    AblationConfig(use_garb=True,  use_hmoe=False, use_clx=True,
                   use_clx_aug=True,  emit_upd=True,  label="garb_clx_aug"),
    AblationConfig(use_garb=True,  use_hmoe=True,  use_clx=True,
                   use_clx_aug=True,  emit_upd=True,  label="full_v2"),
]
```

**Extended result:**
```python
@dataclass
class AblationResult:
    label: str
    retrieval_cosine_sim: float
    expert_tier_distribution: dict[str, int]
    clx_hidden_delta_norm: float
    clx_bank_aug_delta_norm: float       # key: proves CP-2 coupling
    upd_dag_valid: bool                  # DAG passes schema validation
    upd_dag_replay_deterministic: bool   # replay() returns deterministic=True
    upd_explain_subgraph_ratio: float    # explain nodes / total nodes
    total_decisions: int
```

**TDD tests:**
- `test_all_six_rows_run_without_error`
- `test_baseline_no_upd_dag`
- `test_garb_only_emits_valid_dag`
- `test_full_v2_dag_explains_under_40pct`
- `test_clx_aug_delta_greater_than_raw`
- `test_fast_thorough_expert_distributions_differ`
- `test_all_dags_are_replay_deterministic`
- `test_results_json_serializable`

**Commit:** `feat(eval): add combined AMC v2 × UPD six-row ablation harness`

**Stage only:**
```bash
git add src/eval/amc_v2_upd_ablation.py tests/eval/test_amc_v2_upd_ablation.py
```

---

### Tranche D-02: SecurityVerdictAdapter

**Files:** `src/upd/adapters/security.py`, `tests/upd/test_security_verdict_adapter.py`

Maps AMW scanner pipeline → UPD SecurityVerdict DAG:
```
source(file_hash, redaction=hash_only)
  → signal(pe_features) → detector(yara: rule, hit)           supports
  → signal(entropy)     → detector(rf_model: score)           supports/refutes
  → detector(reputation: prevalence)                          supports
  → aggregator(verdict_fusion)
  → policy(quarantine_vs_allow)
  → verdict(malicious|suspicious|clean, score)
  [→ oversight(analyst_override)]
```

**TDD tests:**
- `test_security_golden_fixture_validates_schema`
- `test_yara_hit_produces_supports_detector`
- `test_rf_below_threshold_produces_refutes_edge`
- `test_analyst_override_adds_oversight_node`
- `test_source_redaction_is_hash_only`
- `test_determinism`

**Commit:** `feat(upd): add SecurityVerdictAdapter`

**Stage only:** `src/upd/adapters/security.py tests/upd/test_security_verdict_adapter.py`

---

### Tranche D-03: Connectors

**Files:** `src/upd/connectors/` (elastic_ecs.py, splunk_soar.py, misp.py, github_action.py)

Each connector: `emit(dag: UpdDag) -> dict` — pure serialization, no I/O.

**TDD tests** (one per connector):
- Each test loads the corresponding golden fixture, calls `emit()`, and validates
  the output against the target schema from `integrations/`.
- `test_ecs_connector_emits_valid_ecs_doc`
- `test_soar_connector_emits_verdict_and_subgraph`
- `test_misp_connector_source_becomes_attribute`
- `test_github_action_connector_explain_in_annotation`

**Commit:** `feat(upd): add SOAR/ECS/MISP/GHA connectors`

**Stage only:** `src/upd/connectors/ tests/upd/test_connectors.py`

---

## 9. Track E — Standards

*E-01/E-02 can start after D-01. E-03 can start after D-03.*

### Tranche E-01: Epoch Anchoring + CLI

**Files:** `crates/upd-cli/` (new crate: `upd verify|replay|explain|anchor`)

CLI targets:
```bash
upd verify  <dag_file.json>       # validate schema + signatures
upd replay  <dag_id> --store <db> # re-derive hashes; compare
upd explain <dag_id> --store <db> # print minimal subgraph "because" list
upd anchor  --epoch               # seal Merkle root over unsealed DAGs
```

**Acceptance (SC3, SC7):**
```bash
cargo test -p upd-cli
upd verify schema/examples/memory_admission_golden.json   # exits 0
upd verify schema/examples/tamper_mutated_node.json       # exits 1
```

**Commit:** `feat(crates): add upd-cli with verify/replay/explain/anchor`

---

### Tranche E-02: Conformance Suite ≥ 30 Cases

**Files:** `conformance/cases/`, `conformance/runner.py`

30 cases split across:
- Valid DAGs × 3 domains (9 cases)
- Cycle detection (2)
- Missing verdict (1)
- Tamper: mutated node output, edge weight, producer version (3)
- Signature mismatch (2)
- Replay determinism × 3 domains (3)
- Explain minimality × 3 domains (3)
- Redaction: hash_only + encrypted (2)
- Revocation + revokes edge (2)
- Oversight present (1)
- LLM captured-output exemption (1)
- Cross-engine parity: Python SDK produces same root hash as Rust (1)

**Runner:**
```bash
python conformance/runner.py --engine python  # runs all 30 cases against Python SDK
cargo run -p upd-cli -- conformance           # same 30 cases against Rust
```

Both must pass all 30. Badge emitted as `conformance/results.json`.

**Commit:** `feat(conformance): add 30-case conformance suite + runner`

---

### Tranche E-03: TypeScript SDK + Dashboard "Why" Surface

**Files:** `middle/src/upd/` (TypeScript), `frontend/src/components/ProvDAGViewer/`

TS SDK: `build()`, `validate()`, `explain()`, `emit()` (wraps Python API via BFF).

Dashboard component: renders the minimal subgraph returned by `explain()` as a
collapsible DAG tree. Three tabs: "Why blocked", "Why remembered", "Why acted".

**Commit:** `feat(dashboard): add Provenance DAG viewer + TS UPD SDK`

---

## 10. Timeline

AMC-first ordering trades away early parallelism for a cleaner paper spine. The
critical path is slightly less flashy, but the dependency graph stops lying. Good.

| Week | Track A: AMC substrate | Track B: UPD foundation | Track C: Integration | Track D: Evidence/connectors | Track E: Standards/UI |
|------|------------------------|-------------------------|----------------------|------------------------------|-----------------------|
| 1 | A-00, A-00.5, A-01 | paper/schema review only | - | - | - |
| 2 | A-02, A-03 | paper/schema review only | - | - | - |
| 3 | A-04 | - | - | - | - |
| 4 | A-05, Gate G-A | - | - | - | - |
| 5 | - | B-01, B-02 start | - | - | - |
| 6 | - | B-02 complete | - | - | - |
| 7 | - | B-03 | - | - | - |
| 8 | - | - | C-01 | - | - |
| 9 | - | - | C-02, C-03 | - | - |
| 10 | - | - | - | D-01 | E-01 draft |
| 11 | - | - | - | D-02, D-03 | E-01 |
| 12 | - | - | - | - | E-02 |
| 13 | - | - | - | - | E-02, E-03 |
| 14 | buffer/replication | buffer/replication | buffer/replication | connector hardening | publish package |

**Total: ~14 weeks**, but now the first publishable evidence checkpoint is AMC-only
at Week 4 instead of being entangled with provenance machinery at Week 7+.

---

## 11. Done Criteria

### Gate G-A (Track A complete)
```bash
.venv/bin/python -m pytest \
  tests/memory/test_garb_memory.py \
  tests/model/test_clx_modulator.py \
  tests/model/test_hmoe_layer.py \
  tests/model/test_amc_v2_integration.py \
  tests/eval/test_amc_v2_ablation.py \
  -q
# v1 defaults preserved
.venv/bin/python -c "
from src.model.amc_transformer import AMCTransformerConfig
cfg = AMCTransformerConfig(vocab_size=32, d_model=16, n_layers=4, n_heads=4,
                            kv_lrank=8, max_seq_len=16)
assert not cfg.use_garb and not cfg.use_hmoe and not cfg.use_clx
print('v1 defaults preserved')
# A-05 must also show garb_clx_aug delta > garb_clx_raw delta under the deterministic fixture.
"
```

### Gate G-B (Track B complete)
```bash
cargo test -p upd-core -p upd-store-sqlite
.venv/bin/python -m pytest tests/upd/ -q
# SC2 determinism
cargo test -p upd-core test_canonicalize_is_deterministic_1000_iterations
# SC3 sign/verify < 3ms p99 — verified in Rust test
```

### Gate G-C (Track C complete — M2 milestone)
```bash
.venv/bin/python -m pytest tests/upd/test_upd_e2e_integration.py -q
# One real memory admission produces a signed, verifiable, explainable DAG
```

### Gate G-D (Track D complete)
```bash
.venv/bin/python -m pytest \
  tests/eval/test_amc_v2_upd_ablation.py \
  tests/upd/test_security_verdict_adapter.py \
  tests/upd/test_connectors.py \
  -q
# SC5: all connectors validate against target schemas
```

### Gate G-E (Track E complete — GA milestone)
```bash
python conformance/runner.py --engine python   # all 30 pass
cargo run -p upd-cli -- conformance            # all 30 pass
# SC6: 2 independent engines pass conformance
# SC7: all tamper cases flagged
```

---

## 11.5 Abort / Revision Criteria

Stop and revise the plan rather than pushing forward if any of these fire:

| Gate | Abort condition | Why it matters |
|---|---|---|
| A-01 | GARB cannot round-trip from existing HLM/DreamBank bank state | Indicates a competing memory stack, not AMC-first evolution |
| A-03 | H-MoE tier policy does not change active expert pool under deterministic probes | Routing is decorative; no compute-allocation claim |
| A-04 | All flags false is not v1-identical | Backward compatibility broken before novelty is measured |
| A-05 | `garb_clx_aug <= garb_clx_raw` on the deterministic fixture | The central GARB->CLX coupling is not doing work |
| B-02/B-03 | Rust and Python roots diverge on a golden fixture | UPD is not a conformance target yet |
| C-01 | Raw memory text appears in any DAG fixture | Privacy contract broken |
| C-03 | Replay requires live LLM sampling rather than captured-output hashes | Provenance is not deterministic |
| D-01 | UPD overhead changes AMC-only ablation conclusions | Audit layer is perturbing the substrate it claims to observe |

---

## 12. Paper Evidence Targets

### Paper 1: AMC v2 — Memory-Conditioned Compute Allocation

Every prior method is an ablation. The six-row AMC-only table (A-05) is the core evidence; D-01 repeats it with provenance enabled to measure audit overhead and explainability:

| Row | Claim tested |
|-----|-------------|
| baseline | No memory-conditioned compute (AMC v1) |
| garb_only | GARB improves retrieval quality vs HLMPreferenceBank |
| garb_hmoe | Expert routing on bank signal improves task allocation |
| garb_clx_raw | Forward modulation without bank augmentation |
| garb_clx_aug | GARB→CLX coupling independently improves modulation |
| full_v2 | Closed loop: memory × compute × context = best |

Threshold: `garb_clx_aug delta > garb_clx_raw delta` (proves CP-2 coupling matters).

### Paper 2: UPD — Unified Decision Provenance

Every decision type is a conformance row. SC1–SC7 are the core evidence. The
`explain()` minimality metric (SC4: ≤40% subgraph) is the key product claim.

### Joint claim (cross-paper)

> AMC v2 decisions are natively provenance-tracked. The same bank that conditions
> compute (GARB) is the same bank whose admission decisions are recorded as signed,
> replayable DAGs (UPD). "Why remembered" and "why this expert tier" share one
> explanation surface.

---

## 13. Agent Invariants

```text
Unified AMC v2 × UPD invariants:
1. CLX reads from GARB; never writes. AMC memory writes originate through DreamBankController/run_cycle or its thin admission facade; CLX/H-MoE only read/route.
2. Track A completes before any UPD implementation commit. Schema review is OK; Track B code waits for G-A.
3. emit_upd defaults to False everywhere. Zero overhead when disabled.
4. UPD DAGs are emitted asynchronously after the decision. Never on the hot path.
5. Memory source nodes default to redaction=hash_only. Raw text never in a DAG.
6. GARBMemory.M is not nn.Parameter. Not in model.parameters().
7. AMC v1 behavior is byte-identical when use_garb=use_hmoe=use_clx=False.
8. A-05 AMC-only evidence must pass before D-01 combined evidence exists.
9. DreamBank + CascadeBank focused test suite must pass after every tranche.
10. Stage only files listed per tranche. Do not bundle dirty-tree drift.
11. Do not push. PR only after Gate G-A is verified.
12. Compose existing alignment modules (ORPO, GRPO, SAPO, etc.) — do not rewrite.
```

---

## 14. Cross-Reference

| Topic | Document |
|-------|----------|
| UPD spec (full) | `~/Desktop/AI:ML Research/SPEC-Unified-Provenance-DAG-2026-05-28.md` |
| AMC v2 design (superseded by this) | `docs/plans/2026-05-28-amc-v2-garb-hmoe-clx.md` |
| DreamBank contracts | `docs/plans/2026-05-27-dreambank-implementation.md` |
| CascadeBank contracts | `docs/plans/2026-05-27-cascadebank-implementation.md` |
| CB-06 serving harness | `docs/plans/2026-05-27-cb06-serving-harness.md` |
| Master roadmap | `docs/MASTER-IMPLEMENTATION-PLAN.md` |
| Architecture | `docs/ARCHITECTURE.md` (v4.0) |

---

**Last updated:** 2026-05-29
**Author:** Christien Antonio
**Baseline commit:** `46ee2f13`
