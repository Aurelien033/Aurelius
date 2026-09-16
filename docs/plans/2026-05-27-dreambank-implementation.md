# DreamBank Implementation Plan

> **For Hermes:** Use `subagent-driven-development` to implement this plan task-by-task. Use strict TDD: write the failing test first, run it, implement the minimum code, then re-run focused tests. Do not push.

**Goal:** Add DreamBank to Aurelius: a runtime-writable, MLA-latent preference bank that can be read during forward passes and consolidated during idle/sleep-time self-play without updating model weights.

**Architecture:** Start with a small HLM-style preference bank in MLA latent space (`kv_lrank`, default 64). A differentiable adapter reads the bank from hidden states through a bounded alignment gate `alpha_t`. Dream-mode generates/scored preference pairs through injectable callables and performs no-grad bank upserts. The first implementation is deliberately small: in-process bank, no federated sync, no GPU-heavy training loop.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing Aurelius `src/model/amc_transformer.py`, `src/model/mla.py`, `src/memory/*`, `src/alignment/preference_optimization.py`.

---

## Current Verified Truth Surface

Verified in `/Users/christienantonio/aurelius` on 2026-05-27:

```bash
git -C /Users/christienantonio/aurelius status --short --branch
```

Observed:

```text
## clean/amc-curation-20260521-101220...origin/clean/amc-curation-20260521-101220 [ahead 51]
 M gateway/aurelius_api.py
?? llm-skills/
?? projects/
?? skills/mlops/
?? skills/skill-generative-agents/
?? tests/gateway/test_t3_gateway_fail_closed.py
```

Non-negotiable safety rules:
- Do not stage or commit existing dirty/untracked files above.
- Do not push.
- Stage only the DreamBank files listed in each task.
- If the working tree changes unexpectedly, stop and report.

Existing integration points:
- Model config/output/forward: `src/model/amc_transformer.py`
- MLA latent rank: `src/model/mla.py` (`MLAConfig.kv_lora_rank`, `MultiheadLatentAttention.kv_down`)
- Gate pattern: `src/model/amc_gates.py`
- Runtime memory/trust primitives: `src/memory/amc_runtime_cache.py`
- Tier-2 episodic hook pattern: `src/memory/amc_tier2.py`
- Preference losses: `src/alignment/preference_optimization.py`
- Existing focused tests: `tests/model/test_amc_transformer.py`, `tests/model/test_mla.py`, `tests/memory/test_amc_tier2_runtime_cache.py`, `tests/alignment/test_preference_optimization.py`

---

## Research Claim Being Built

DreamBank is not "generic memory bank" novelty. That is already crowded: Larimar, Titans, Memorizing Transformers, MSA, MoC.

The defensible claim is narrower and cleaner:

> A tiny runtime-writable preference bank in MLA latent space can be updated during idle-time self-play to improve alignment/persona behavior without weight updates, while preserving bounded compute and privacy-friendly storage.

Minimum viable evidence:
1. Bank read path changes hidden states/logits only when enabled and non-empty.
2. Bank writes are no-grad, bounded, trust-labeled, and metadata-safe.
3. Dream cycles create deterministic bank updates from preference pairs.
4. Ablation shows bank-on differs from bank-off under identical model weights.
5. Sleep-time consolidation improves preference-score proxy on a tiny deterministic benchmark.

---

## Proposed File Map

Create:
- `src/memory/hlm_bank.py` — core bank data structure, top-k read, upsert, decay, serialization payload.
- `src/model/hlm_bank_adapter.py` — differentiable hidden-state adapter: gate, latent query, bank read, bias projection.
- `src/alignment/dreambank.py` — dream-mode controller, pair schema, scoring/upsert loop.
- `scripts/run_dreambank_cycle.py` — dry-run CLI for one or N dream cycles.
- `tests/memory/test_hlm_bank.py`
- `tests/model/test_hlm_bank_adapter.py`
- `tests/alignment/test_dreambank.py`
- `tests/scripts/test_run_dreambank_cycle.py`

Modify:
- `src/model/amc_transformer.py` — config fields, output telemetry, optional adapter wiring.
- `src/memory/__init__.py` — export HLM bank symbols.
- `src/alignment/__init__.py` — export DreamBank symbols only if this package already exports alignment helpers; otherwise leave untouched.
- `docs/research-brief.md` — one short entry documenting the claim and files.

Defer until after MVP:
- Federated Preference Banks.
- BitNet/ternary expert integration.
- PASI/DSA indexer preference routing.
- Persistent on-disk bank format beyond a safe dict/safetensors-compatible payload.

---

## Core API Contract

### `src/memory/hlm_bank.py`

Required public surface:

```python
@dataclass(frozen=True)
class HLMPreferenceBankConfig:
    bank_size: int = 14
    bank_dim: int = 64
    top_k: int = 4
    read_temperature: float = 8.0
    decay: float = 0.995
    min_strength: float = 1e-4
    write_momentum: float = 0.2
    eps: float = 1e-6

@dataclass(frozen=True)
class HLMPreferenceWrite:
    key: torch.Tensor       # (..., bank_dim) or (bank_dim,)
    value: torch.Tensor     # same shape contract as key
    strength: float
    provenance: str = "dreambank"
    trust: str = "unverified"
    metadata_hash: str = ""

@dataclass(frozen=True)
class HLMPreferenceRead:
    context: torch.Tensor   # (..., bank_dim)
    weights: torch.Tensor   # (..., top_k)
    indices: torch.Tensor   # (..., top_k)
    confidence: torch.Tensor # (..., 1)

class HLMPreferenceBank(nn.Module):
    def is_empty(self) -> bool: ...
    def read(self, query: torch.Tensor, *, top_k: int | None = None) -> HLMPreferenceRead: ...
    @torch.no_grad()
    def upsert(self, write: HLMPreferenceWrite) -> int: ...
    @torch.no_grad()
    def decay_(self, steps: int = 1) -> None: ...
    @torch.no_grad()
    def consolidate_(self) -> None: ...
    def telemetry(self) -> dict[str, float | int]: ...
    def export_state(self) -> dict[str, torch.Tensor | list[str]]: ...
    @classmethod
    def from_state(cls, state: dict[str, object]) -> "HLMPreferenceBank": ...
```

Implementation notes:
- Store keys/values/strength as registered buffers, not parameters.
- Normalize keys before cosine similarity.
- Empty bank read returns zeros and confidence 0, not NaN.
- Upsert chooses first empty slot, otherwise low-strength/LRU slot.
- Never store raw prompts or raw responses in metadata; store only hashes/provenance tags.

### `src/model/hlm_bank_adapter.py`

Required public surface:

```python
@dataclass(frozen=True)
class HLMPreferenceAdapterConfig:
    d_model: int
    bank_dim: int = 64
    gate_hidden: int = 0
    inject_scale: float = 0.1
    top_k: int = 4

@dataclass(frozen=True)
class HLMPreferenceAdapterOutput:
    hidden: torch.Tensor
    alpha: torch.Tensor
    bank_context: torch.Tensor
    confidence: torch.Tensor

class HLMPreferenceAdapter(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        bank: HLMPreferenceBank | None,
        *,
        read_only: bool = True,
    ) -> HLMPreferenceAdapterOutput: ...
```

Mechanism:

```text
query_t = normalize(query_proj(LN(hidden_t)))              # d_model -> bank_dim
alpha_t = sigmoid(gate_mlp(LN(hidden_t)))                  # d_model -> 1
bank_context_t = bank.read(query_t).context                # bank_dim
bias_t = out_proj(bank_context_t)                          # bank_dim -> d_model
hidden'_t = hidden_t + inject_scale * alpha_t * bias_t
```

Invariants:
- If `bank is None` or empty, `hidden' == hidden` exactly or within float tolerance.
- `alpha_t` is in `[0, 1]` and shape `(B, T, 1)`.
- Gradients flow into adapter parameters, not into bank buffers.

### `src/alignment/dreambank.py`

Required public surface:

```python
@dataclass(frozen=True)
class DreamBankConfig:
    max_candidates_per_seed: int = 4
    temperatures: tuple[float, ...] = (0.2, 0.7, 1.0, 1.3)
    min_margin: float = 0.05
    max_writes_per_cycle: int = 8
    decay_steps_per_cycle: int = 1
    metadata_salt: str = "dreambank-v1"

@dataclass(frozen=True)
class DreamSeed:
    prompt: str
    source: str = "recent_query"

@dataclass(frozen=True)
class DreamPreferencePair:
    prompt_hash: str
    chosen: str
    rejected: str
    margin: float
    provenance: str

@dataclass(frozen=True)
class DreamCycleResult:
    seeds: int
    candidates: int
    pairs: int
    writes: int
    mean_margin: float
    bank_fill: int

class DreamBankController:
    def run_cycle(
        self,
        seeds: Sequence[DreamSeed],
        *,
        generate_fn: Callable[[str, float], str],
        score_fn: Callable[[str, str], float],
        embed_fn: Callable[[str], torch.Tensor],
    ) -> DreamCycleResult: ...
```

MVP scoring rule:
- For each seed, generate responses at configured temperatures.
- Score each response with injected `score_fn`.
- Chosen = highest score, rejected = lowest score.
- If `chosen_score - rejected_score >= min_margin`, write chosen embedding to bank.
- Use `prompt_hash`, not raw prompt, in the write metadata.

---

## Tranches

### Task DB-00: Pre-flight and baseline

**Objective:** Verify current repo state and capture baseline before writing code.

**Files:** None.

**Commands:**

```bash
cd /Users/christienantonio/aurelius && git status --short --branch
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest --collect-only tests -q --maxfail=1
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_amc_transformer.py tests/model/test_mla.py tests/alignment/test_preference_optimization.py -q
```

**Acceptance criteria:**
- Baseline status recorded.
- Collection either passes or any pre-existing failure is recorded before DreamBank work starts.
- No files changed.

**Commit:** None.

---

### Task DB-01: Core HLM preference bank

**Objective:** Implement the bank as an isolated memory module with deterministic top-k read and no-grad upsert.

**Files:**
- Create: `src/memory/hlm_bank.py`
- Create: `tests/memory/test_hlm_bank.py`
- Modify: `src/memory/__init__.py`

**TDD tests to write first:**
- `test_bank_config_rejects_invalid_values`
- `test_empty_bank_read_returns_zero_context_and_zero_confidence`
- `test_upsert_fills_first_empty_slot`
- `test_read_returns_nearest_written_slot`
- `test_upsert_replaces_lowest_strength_when_full`
- `test_decay_and_consolidate_clear_weak_slots`
- `test_export_import_round_trip_preserves_read_result`
- `test_write_metadata_does_not_require_raw_prompt_text`

**Implementation steps:**
1. Write `tests/memory/test_hlm_bank.py` with the tests above.
2. Run and verify RED:
   ```bash
   cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py -q
   ```
3. Implement `HLMPreferenceBankConfig`, `HLMPreferenceWrite`, `HLMPreferenceRead`, `HLMPreferenceBank`.
4. Export symbols in `src/memory/__init__.py` only after tests prove the module works directly.
5. Run focused tests.

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/memory/hlm_bank.py
```

**Acceptance criteria:**
- Empty reads are zero/finite.
- Top-k read returns expected nearest slot.
- Upserts are no-grad and mutate buffers only.
- No raw prompt text is required or stored.

**Commit message:** `feat(memory): add HLM preference bank`

**Stage only:**
```bash
git add src/memory/hlm_bank.py src/memory/__init__.py tests/memory/test_hlm_bank.py
```

---

### Task DB-02: Differentiable HLM bank adapter

**Objective:** Add a model adapter that turns hidden states into latent bank reads and gated residual bias.

**Files:**
- Create: `src/model/hlm_bank_adapter.py`
- Create: `tests/model/test_hlm_bank_adapter.py`

**TDD tests to write first:**
- `test_adapter_returns_identity_when_bank_is_none`
- `test_adapter_returns_identity_when_bank_empty`
- `test_adapter_alpha_shape_and_bounds`
- `test_adapter_changes_hidden_when_bank_has_matching_slot`
- `test_adapter_gradients_flow_to_adapter_not_bank_buffers`
- `test_adapter_rejects_invalid_inject_scale`

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_hlm_bank_adapter.py tests/memory/test_hlm_bank.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/model/hlm_bank_adapter.py
```

**Acceptance criteria:**
- Adapter is mathematically inert when bank absent/empty.
- Adapter changes hidden states when bank has a relevant slot.
- Gate is bounded and telemetry-ready.
- No bank gradients.

**Commit message:** `feat(model): add HLM preference bank adapter`

**Stage only:**
```bash
git add src/model/hlm_bank_adapter.py tests/model/test_hlm_bank_adapter.py
```

---

### Task DB-03: Wire optional bank reads into `AMCTransformer`

**Objective:** Make DreamBank optional in the forward pass without changing default behavior.

**Files:**
- Modify: `src/model/amc_transformer.py`
- Modify: `tests/model/test_amc_transformer.py`

**Config additions:**

```python
use_hlm_bank: bool = False
hlm_bank_size: int = 14
hlm_bank_dim: int | None = None       # default to kv_lrank in __post_init__
hlm_bank_top_k: int = 4
hlm_bank_inject_scale: float = 0.1
hlm_bank_read_layers: tuple[int, ...] | None = None  # default final-only for MVP
```

**Output additions:**

```python
bank_alpha: torch.Tensor | None = None
bank_confidence: torch.Tensor | None = None
bank_telemetry: dict[str, float | int] | None = None
```

**Forward additions:**
- Add optional keyword `preference_bank: HLMPreferenceBank | None = None`.
- If `config.use_hlm_bank` is false, behavior must be bit-for-bit equivalent to current path.
- MVP placement: apply adapter once after `self.norm(hidden)` and before `lm_head`.
- Later placement can be per-MLA layer; do not do that in MVP unless final-only fails.

**TDD tests to write first:**
- `test_hlm_bank_disabled_by_default_preserves_output_shape`
- `test_hlm_bank_enabled_with_empty_bank_preserves_logits_close`
- `test_hlm_bank_enabled_with_written_slot_returns_bank_telemetry`
- `test_hlm_bank_forward_changes_logits_when_nonempty`
- `test_hlm_bank_output_fields_absent_when_disabled`

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/model/test_amc_transformer.py tests/model/test_hlm_bank_adapter.py tests/memory/test_hlm_bank.py -q
```

**Acceptance criteria:**
- Existing AMCTransformer tests still pass.
- Default config unchanged.
- Bank-on path is explicit and opt-in.
- Output telemetry is present only when enabled.

**Commit message:** `feat(model): wire optional DreamBank reads into AMCTransformer`

**Stage only:**
```bash
git add src/model/amc_transformer.py tests/model/test_amc_transformer.py
```

---

### Task DB-04: DreamBank controller and deterministic dream cycles

**Objective:** Implement dream-mode as a deterministic, injectable loop independent of actual text generation.

**Files:**
- Create: `src/alignment/dreambank.py`
- Create: `tests/alignment/test_dreambank.py`
- Optionally modify: `src/alignment/__init__.py`

**TDD tests to write first:**
- `test_dream_cycle_no_seeds_no_writes`
- `test_dream_cycle_generates_candidates_per_temperature`
- `test_dream_cycle_writes_only_when_margin_exceeds_threshold`
- `test_dream_cycle_writes_chosen_not_rejected_embedding`
- `test_dream_cycle_metadata_uses_hash_not_raw_prompt`
- `test_dream_cycle_applies_bank_decay_once_per_cycle`
- `test_dream_cycle_respects_max_writes_per_cycle`

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/alignment/test_dreambank.py tests/memory/test_hlm_bank.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -m py_compile src/alignment/dreambank.py
```

**Acceptance criteria:**
- Controller works with dummy `generate_fn`, `score_fn`, `embed_fn`.
- No raw prompt text is stored in bank metadata.
- Writes are bounded by config.
- Result metrics are JSON-safe.

**Commit message:** `feat(alignment): add DreamBank sleep-time controller`

**Stage only:**
```bash
git add src/alignment/dreambank.py tests/alignment/test_dreambank.py src/alignment/__init__.py
```

If `src/alignment/__init__.py` is not modified, do not stage it.

---

### Task DB-05: CLI dry-run script

**Objective:** Provide a tiny script that runs DreamBank cycles with deterministic local functions so future agents can smoke-test without a real model/generator.

**Files:**
- Create: `scripts/run_dreambank_cycle.py`
- Create: `tests/scripts/test_run_dreambank_cycle.py`

**CLI contract:**

```bash
python scripts/run_dreambank_cycle.py --dry-run --cycles 2 --seed "help me write safer code"
```

Expected stdout JSON keys:
- `cycles`
- `total_writes`
- `bank_fill`
- `mean_margin`
- `dry_run: true`

**TDD tests to write first:**
- `test_dreambank_runner_dry_run_outputs_json`
- `test_dreambank_runner_respects_cycles`
- `test_dreambank_runner_rejects_negative_cycles`

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/scripts/test_run_dreambank_cycle.py tests/alignment/test_dreambank.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python scripts/run_dreambank_cycle.py --dry-run --cycles 1 --seed "test" | .venv/bin/python -m json.tool >/tmp/dreambank.json
```

**Acceptance criteria:**
- Script has no network calls.
- Script has no external model requirement.
- Output is valid JSON.

**Commit message:** `feat(scripts): add DreamBank dry-run cycle runner`

**Stage only:**
```bash
git add scripts/run_dreambank_cycle.py tests/scripts/test_run_dreambank_cycle.py
```

---

### Task DB-06: Tiny ablation harness

**Objective:** Prove bank-on vs bank-off can be measured under identical weights.

**Files:**
- Create: `src/eval/dreambank_ablation.py`
- Create: `tests/eval/test_dreambank_ablation.py`

**MVP metric:**
- Run same tiny `AMCTransformer` input twice: empty/no bank vs non-empty bank.
- Report mean absolute logit delta, alpha mean, confidence mean, bank fill.
- This is not a quality claim yet. It proves the intervention is live and measurable.

**TDD tests to write first:**
- `test_ablation_returns_zeroish_delta_for_empty_bank`
- `test_ablation_returns_positive_delta_for_nonempty_bank`
- `test_ablation_result_is_json_serializable`

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/eval/test_dreambank_ablation.py tests/model/test_amc_transformer.py tests/memory/test_hlm_bank.py -q
```

**Acceptance criteria:**
- Ablation runs on CPU with tiny config.
- Empty bank does not perturb logits materially.
- Non-empty bank produces measurable delta.

**Commit message:** `feat(eval): add DreamBank intervention ablation`

**Stage only:**
```bash
git add src/eval/dreambank_ablation.py tests/eval/test_dreambank_ablation.py
```

---

### Task DB-07: Documentation and research traceability

**Objective:** Record the exact claim, prior-art boundary, and validation commands.

**Files:**
- Modify: `docs/research-brief.md`
- Create: `docs/reports/DREAMBANK_MVP_TRACEABILITY.md`

**Required content:**
- Claim: sleep-time preference consolidation in runtime-writable MLA latent bank.
- Not claimed: generic memory bank invention, new preference loss, federated sync.
- Prior art boundary: Larimar/Titans/Memorizing Transformers/MSA/MoC own generic bank territory.
- Evidence produced by MVP: bank contract tests, adapter tests, dream-cycle tests, ablation harness.
- Next evidence needed: real preference benchmark, sleep-cycle improvement curve, privacy test.

**Validation commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/alignment/test_dreambank.py tests/eval/test_dreambank_ablation.py -q
```

**Commit message:** `docs: add DreamBank MVP traceability`

**Stage only:**
```bash
git add docs/research-brief.md docs/reports/DREAMBANK_MVP_TRACEABILITY.md
```

---

### Task DB-08: Integration proof sweep

**Objective:** Make sure the MVP did not fracture package imports or existing AMC paths.

**Files:** None unless fixing regressions caused by DreamBank.

**Commands:**

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py tests/scripts/test_run_dreambank_cycle.py tests/eval/test_dreambank_ablation.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "from src.memory.hlm_bank import HLMPreferenceBank; from src.model.hlm_bank_adapter import HLMPreferenceAdapter; from src.alignment.dreambank import DreamBankController; print('dreambank imports OK')"
cd /Users/christienantonio/aurelius && .venv/bin/python -m compileall -q src/memory/hlm_bank.py src/model/hlm_bank_adapter.py src/alignment/dreambank.py src/eval/dreambank_ablation.py scripts/run_dreambank_cycle.py
cd /Users/christienantonio/aurelius && git diff --stat
```

**Acceptance criteria:**
- Focused suite green.
- Imports green.
- Compileall green.
- Diff contains only intended DreamBank files.

**Commit:** If DB-01 through DB-07 were committed individually, no new commit. If implemented as one branch commit, use: `feat: add DreamBank MVP`.

---

## First Real Benchmark After MVP

Do not overbuild this in MVP. After DB-08 is green, run a tiny deterministic preference benchmark:

1. Create 50 synthetic prompts with two response styles: safe/helpful vs risky/unhelpful.
2. Score with a deterministic proxy first; later replace with a reward model/judge.
3. Run:
   - baseline model, no bank
   - model + empty bank
   - model + manually seeded bank
   - model + DreamBank 10 cycles
4. Report:
   - preference win rate
   - refusal over-trigger rate
   - mean logit delta
   - p95 latency delta
   - bank fill/turnover

Paper-grade threshold to continue:
- DreamBank 10 cycles improves proxy win rate by at least +5pp over empty-bank with p95 latency delta under 5% on tiny CPU test.
- If not, pivot to CascadeBank early-exit instead of spending GPU.

---

## Invariants to Paste Into Any Implementation Agent Prompt

```text
DreamBank MVP invariants:
1. Bank slots are buffers, not trainable parameters; writes run under torch.no_grad().
2. Empty or absent bank must be identity: logits/hidden close to bank-disabled path.
3. Store hashes/provenance only; never persist raw prompts or raw responses in bank metadata.
4. Default AMCTransformer behavior must remain unchanged unless use_hlm_bank=True and a bank is passed.
5. Stage only DreamBank files; do not touch existing dirty/untracked repo drift.
```

---

## Cursor CLI Execution Prompt

Save this plan, then run one tranche at a time. Example:

```bash
cd /Users/christienantonio/aurelius && cursor composer -f /Users/christienantonio/aurelius/docs/plans/2026-05-27-dreambank-implementation.md
```

But for best results, paste only one task section at a time plus the invariant block above. Full-plan context can cause agents to jump ahead and modify too many files. Bad agents love scope creep. Very enterprise. Keep them fenced.

---

## Done Criteria for MVP

DreamBank MVP is complete when all are true:

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py tests/scripts/test_run_dreambank_cycle.py tests/eval/test_dreambank_ablation.py -q
cd /Users/christienantonio/aurelius && .venv/bin/python -c "from src.memory.hlm_bank import HLMPreferenceBank; from src.model.hlm_bank_adapter import HLMPreferenceAdapter; from src.alignment.dreambank import DreamBankController; print('dreambank imports OK')"
cd /Users/christienantonio/aurelius && .venv/bin/python scripts/run_dreambank_cycle.py --dry-run --cycles 2 --seed "help me write safer code"
```

Expected:
- Focused tests pass.
- Import smoke prints `dreambank imports OK`.
- Dry-run emits JSON with `total_writes > 0` and `bank_fill > 0`.
- `git diff --stat` shows only intended DreamBank files.

---

## Next Phase After MVP

If MVP passes, the next implementation plan should target one of these:

1. **CascadeBank:** reuse `alpha_t` and `confidence` for early exit. This is the cheapest compute win.
2. **DreamBank Benchmark v1:** real preference benchmark with judge/reward model.
3. **Federated Preference Banks:** simulate N local banks and DP/FedAvg merge.
4. **Per-layer MLA Bank:** move final-only bank read into selected MLA layers after correctness is proven.

Recommended next step: CascadeBank, because the adapter telemetry from DB-03 gives the exact signals needed for early exit.
