# AURELIUS GRAND UNIFIED RESEARCH PLAN v4.0
## The 12-Month Path to 4 Papers, One Complete Cognitive Platform, and a Research Program That Outlasts Any Single Publication

**Generated**: 2026-05-29 | **Supersedes**: aurelius-master-plan-v3, aurelius-master-strategic-plan  
**Status**: Canonical, gated, and truth-surface-sensitive  
**Model**: 150M–7B parameter family with per-layer Aurelian Memory Core  
**Target venues**: NeurIPS 2027 (primary), ICLR 2027 Workshop, ICML 2027

> **For any future agent executing this plan**: this is a plan authority, not a license to blindly edit. First re-check the live repo truth surface, then execute one gated tranche at a time. Do not assume model-specific tooling or stale file paths.

---

## MERGE ADDENDUM — AMC v2 × UPD is now inside v4 (2026-05-29)

This section is the canonical merge layer between this roadmap and
`docs/plans/2026-05-28-amc-v2-upd-unified.md`. If a later section appears to
conflict with this addendum, prefer this addendum: the point is to prevent two
beautifully detailed plans from becoming two competing operating systems. That
way lies enterprise software and sadness.

### Source fingerprints before merge

| Source plan | SHA-256 | Imported role |
|-------------|---------|---------------|
| `docs/plans/2026-05-29-aurelius-grand-unified-v4.md` | `aef9b881e1769609bf81890533bdbe2fe6bc09c7304d1512e3d8b9499c150e46` | Master research roadmap and paper schedule |
| `docs/plans/2026-05-28-amc-v2-upd-unified.md` | `2de507459ab209b44d01e06c3470dd678decba17be988da542ce7569d262686e` | Detailed AMC v2 × UPD execution runbook |

### Current verified truth surface

Verified on 2026-05-29 at repo HEAD `46ee2f13 memory(debate): LLM voices for proposer/skeptic/judge via configurable API`.

- Both source plan files exist and are currently untracked in git.
- Existing core surfaces confirmed present: `paper/main.tex`, `paper/sections/03_method.tex`, `paper/tables/ablation_main.tex`, `src/eval/ablation.py`, `src/eval/amc_memory_benchmark.py`, `src/training/tst_trainer.py`, `src/alignment/dreambank.py`, `src/memory/hlm_bank.py`, `src/model/amc_transformer.py`, `src/inference/cascade_routing.py`, and `src/safety/constitutional_principles_scorer.py`.
- Planned AMC v2 / UPD surfaces are not present yet: `src/memory/garb_memory.py`, `src/model/clx_modulator.py`, `src/model/hmoe_layer.py`, `src/upd/`, `crates/upd-core/`, and `schema/upd-v1.json`.
- `config/aurelius.yaml` was not present during this verification pass; any TST activation task must first locate or create the active config path rather than assuming that exact file exists.
- Live `AMCTransformerConfig.hlm_bank_read_layers` is `tuple[int, ...] | None`, not `list[int] | None`.
- Live `DreamBankController` exposes `run_cycle(...)` and writes through `HLMPreferenceBank.upsert(...)`; there is no live `DreamBankController.admit()` method. Any later admission facade must delegate to the existing run-cycle/upsert event path.

### Deep audit corrections from the second pass

The merge is stronger after one more uncomfortable truth pass. Several earlier lines were directionally right but operationally too optimistic:

- `src/training/tst_trainer.py` exists, but TST is not a config-only switch today. The current forge launcher is `src/training/launch_amc_training.py`; there is no live `src/training/train.py`. `AMCTransformerConfig` has no `tst` field and `AMCTransformer.forward(...)` has no `mode="superposition"` parameter. Therefore TST is a readiness probe before T34, not a magic 30-minute activation.
- `scripts/fill_tables.py` does not exist. The live helper is `scripts/analyze_ablation.py`; table filling is either manual from the JSONL summary or an explicit new T34 artifact.
- Repo-root `MODEL_CARD.md`, `GAP_LEDGER.md`, and `CLAIMS_LEDGER.md` were not present during this pass. Claims in this document remain useful planning claims, but paper-facing claims must be re-derived from `paper/`, `docs/`, tests, and concrete source files before submission.
- The Brain section is a spec-driven build, not mostly existing integration: `docs/BRAIN_ARCHITECTURE.md` exists, while the named `src/agent/agent_loop.py`, `planning_engine.py`, `brain_controller.py`, `working_memory.py`, and related files were not present. Existing agent surfaces include `react_loop.py`, `workflow_shell.py`, `skill_library.py`, `tool_registry_dispatcher.py`, and `surface_catalog.py`.
- `scripts/run_dreambank_cycle.py` exists and has script tests; DreamBank is a better near-term evaluation lane than any large speculative agent rebuild.

Rule: whenever this plan says "enable", read it as "prove the live code path consumes the switch". Otherwise the plan turns into YAML cosplay, the saddest kind of cosplay.

### Canonical merge decision

The grand unified v4 roadmap remains the master plan. The AMC v2 × UPD plan is
imported as a post-AMC-paper bridge workstream:

```text
Phase 0: close current AMC paper (T33/T34/T35)
  -> AMC paper table lock / submission gate
  -> Phase 1A: AMC v2 Track A only (GARB -> CLX -> H-MoE -> AMCTransformer -> A-05)
  -> Gate G-A: AMC-only evidence exists, v1 defaults preserved
  -> Phase 1B: UPD foundation and adapters (audit/provenance only)
  -> Phase 2+: safety, DreamBank, Brain, APEX consume AMC/UPD evidence without redefining memory ownership
```

The important first-principles split:

- **AMC remains the computational substrate.** It owns memory admission, promotion, quarantine, routing, and compute allocation.
- **UPD is downstream audit/provenance.** It records and verifies AMC decisions after the fact; it does not become a second memory manager.
- **T34 remains the immediate blocker for the current AMC paper.** AMC v2 × UPD is the next architecture extension, not a reason to delay closing the paper already 80% done.

### Conflict-resolution table

| Conflict | Resolution |
|----------|------------|
| v4 says T34 is the only real blocker; AMC v2 × UPD has a 14-week buildout | T34 is the blocker for the current AMC paper. AMC v2 × UPD starts after paper table lock / submission as the next extension track. |
| v4 says four papers, while the UPD plan names an additional UPD paper | Keep four primary papers. Treat UPD as a standards/conformance artifact or companion systems report until Gate G-D proves it deserves a separate paper. |
| v4 Phase 1 pushes DreamBank experiments; AMC v2 Track A also wants July execution | Run DreamBank experiments and AMC v2 Track A as two Phase 1 lanes, but only Track A can define new memory/compute substrate behavior. |
| UPD wants schema, Rust core, Python SDK, connectors, dashboard | UPD schema review may happen early, but implementation waits for Gate G-A. Dashboard/connectors wait until deterministic replay/conformance exists. |
| APEX/HLM flywheel wants to consume DreamBank preferences | APEX consumes signed/audited AMC evidence after AMC paper submission and after Gate G-A where relevant; it cannot reshape the AMC paper claims retroactively. |

### Imported execution spine from `2026-05-28-amc-v2-upd-unified.md`

| Imported gate | Keep? | Role inside v4 |
|---------------|-------|----------------|
| A-00/A-00.5 pre-flight + AMC-native surface audit | Yes | First post-paper action before any new AMC v2 module |
| A-01 GARBMemory | Yes | AMC-native upgrade path over HLM/DreamBank bank state; no second memory owner |
| A-02 CLXModulator | Yes | Cross-layer context read path; reads memory, never writes |
| A-03 HMoELayer | Yes | Memory-conditioned compute allocation |
| A-04 AMCTransformer integration | Yes | All flags default false; v1 behavior preserved |
| A-05 AMC-only evidence harness | Yes | Gate G-A; proves signal before provenance |
| B-01/B-02/B-03 UPD schema/core/SDK | Yes, after G-A | Audit/provenance foundation only |
| C-01/C-02/C-03 adapters/e2e | Yes, after B | Convert AMC memory and agent decisions into signed DAGs |
| D/E evidence, connectors, conformance, dashboard | Yes, late | Standards/product layer after replay/signature gates |

---

## PART 0 — THE GROUND TRUTH

### 0.1 What Both Prior Plans Got Wrong

Both prior plans described Aurelius as "a 6.8B transformer with GQA/MoE that needs paper techniques applied to it." That is incorrect at every level.

| Prior Plans Claimed | Actual Reality |
|--------------------|---------------|
| 6.8B dense transformer | **150M–7B / 1B-class model family** with per-layer AMC hierarchy, grounded in `configs/amc_forge_1b.yaml`, `configs/train_1b.yaml`, and live model code |
| Needs Liger-Kernel for training speed | TST trainer exists, but current launcher/model wiring must be proven before counting the 2.5× speedup |
| AMC is one feature to add | AMC is **the core architecture** — per-layer, differentiable, three-tier, already coded |
| DETER is the primary novel contribution | **7 claims BACKED, paper in LaTeX** — publication is 6 weeks away, not 6 months |
| Need to build memory system | `src/memory/` has 20+ modules: tier2, tier3, constitutional, episodic, federated banks |
| Need to build safety | `src/safety/` has 15+ guards; Constitutional Memory already BACKED (C6) |
| DreamBank is a future idea | `src/alignment/dreambank.py` is **live** — sleep-time preference consolidation running |
| 5+ separate technique papers | **4 architecture papers** — boundaries already exist in the codebase |

### 0.2 The Paper Is 6 Weeks From Submission — Not 6 Months

The AMC paper lives at `paper/` with this tranche map:

| Tranche | Status | Contents |
|---------|--------|---------|
| T32 | ✅ DONE | LaTeX skeleton, abstract, introduction, method outline, training outline, conclusion draft |
| T33 | ⚠️ IN PROGRESS | Formalize math: ZOH equations, surprise gate, Gumbel-softmax, SDB contract |
| T34 | ❌ BLOCKING | Run 4 trained checkpoints, fill tables in `tables/ablation_main.tex`, `tables/safety_audit.tex` |
| T35 | ❌ PENDING | Final assembly, author bios, related work completion, arXiv upload |

**T34 is the only real blocker.** The experiment infrastructure is already built (`src/eval/ablation.py`, `docs/reproducibility/results/`). C8 (ablation with trained checkpoints) closes when `scripts/run_ablation.py --mode engine` emits real JSONL from trained weights. C9 closes only after a verified benchmark adapter path runs GSM8K/MMLU or an explicitly equivalent harness.

### 0.3 The Critical Path (3-week paper unblock, not 72-hour magic)

The highest-ROI action is still T34/T35, but the execution path must be tied to the live launcher, not imagined flags.

```text
Step 0 — Freeze the truth surface (30 min)
  git status --short --branch
  python scripts/validate_amc_forge_config.py
  python -c "from src.training.launch_amc_training import parse_args; from src.training.tst_trainer import TokenSuperpositionTrainer; print('launcher+TST import OK')"

Step 1 — TST readiness decision (half day, hard stop)
  TST exists as src/training/tst_trainer.py, but is not currently a config-only switch.
  Do not add model.tst.* to configs/amc_forge_1b.yaml unless the launcher/model path consumes it.
  Gate: one dry/sanity run proves superposition batches reach the model, or T34 proceeds without TST.

Step 2 — Train the T34 evidence checkpoints (about 1 week once data is ready)
  Live launcher: src/training/launch_amc_training.py
  No live --variant flag exists today. Each variant must be represented by a checked config overlay
  or by an explicit launcher-supported ablation config before training starts.

  Example shape, not a blind copy-paste command:
    python src/training/launch_amc_training.py \
      --config configs/amc_forge_1b.yaml \
      --data data/tokenized/amc_forge_1b \
      --log_dir logs/t34/full_amc \
      --deepspeed configs/deepspeed_zero2.json

Step 3 — Run ablation with trained weights (1 day)
  Live script: scripts/run_ablation.py
    python scripts/run_ablation.py \
      --checkpoint logs/t34/full_amc/checkpoint-final.pt \
      --mode engine \
      --output docs/reproducibility/results/ablation_scores.jsonl

Step 4 — Run standard benchmarks (1 day)
  lm-eval is a target benchmark path, not yet verified here as a working Aurelius adapter.
  Gate: first prove the adapter command on a tiny smoke run; if absent, record C9 as blocked by adapter work,
  not by model quality.

Step 5 — Fill LaTeX tables (2 hours)
  Live helper: python scripts/analyze_ablation.py --results docs/reproducibility/results/ablation_scores.jsonl
  Then edit paper/tables/ablation_main.tex and paper/tables/safety_audit.tex from actual outputs.
  Do not cite oracle-mode values as paper evidence.

Step 6 — T35 Assembly (1 week)
  Complete paper/sections/02_related_work.tex
  Complete paper/sections/06_analysis.tex
  Run: make -C paper pdf
  Upload to arXiv
```

**End state**: AMC paper submitted. T34 is roughly 1 week of training + 2 days of eval after the launcher/data path is green. The deeper insight is simple: do not spend a week integrating a speedup to save a week unless the half-day readiness probe is green.

---

## PART I — THE UNIFIED THESIS

### 1.1 The Corrected Pitch

Stop describing Aurelius as: *"A transformer with GQA, MoE top-2, SimPO, and 25 paper techniques applied."*

Start describing Aurelius as:

> **"A complete end-to-end cognitive platform: per-layer differentiable memory that learns what to remember and forget, sleep-time self-improvement without human annotation, MCTS planning in pure embedding space, architectural safety through non-evictable constitutional values, and a recursive self-upgrade loop that identifies and improves its own weakest module — all in a single trainable 1B–7B model family."**

This repositions the work:
- Against cognitive architectures (ACT-R, SOAR, OpenCog) — not just transformers
- As a platform, not a model — the architecture is the contribution
- With safety as a first-class architectural property — not a fine-tuning afterthought
- As a self-improving system — the model helps design its successors

### 1.2 The Structural Principles (From Karpathy + Software Architecture Skills)

**Surface assumptions explicitly** (Karpathy): Every paper claim must cite a specific file and test. This plan maintains that discipline throughout.

**Keep boundaries clear** (Software Architecture): The four papers have distinct boundaries — memory architecture, sleep learning, safety architecture, cognitive control. They cite each other but are independently publishable.

**Simple abstractions that fit the problem**: The AMC hierarchy (Tier-1 SSM → Tier-2 episodic → Tier-3 consolidated) is the simplest structure that solves the three memory problems stated in the introduction: memory is not learned, not safe, and opaque. Every architectural decision should be justified against one of these three problems.

**Tests confirm the intended design**: All 10 backed claims (C1–C7, C10 for AMC; plus the 132 integration tests) are the executable specification. Code that passes tests but violates the paper's invariants is wrong code.

---

## PART II — COMPLETE IMPLEMENTATION INVENTORY

### 2.1 Source Modules (40+ directories, not 6)

Both prior plans listed 6 source files. The actual `src/` directory has 40+ subdirectories:

| Domain | Directory | Key Files | Status |
|--------|-----------|-----------|--------|
| **Memory** | `src/memory/` | `amc_tier2.py`, `amc_tier3.py`, `constitutional_memory.py`, `hlm_bank.py`, `federated_banks.py` | Core — fully implemented |
| **Alignment** | `src/alignment/` | `dreambank.py`, `cai_pipeline.py`, `constitutional_ai_v2.py`, `absolute_zero.py`, `chain_of_hindsight.py` | Rich — 30+ algorithms |
| **Reasoning** | `src/reasoning/` | `mcts_reasoner.py`, `tot_planner.py`, `self_consistency.py`, `gnn_layer.py`, `chain_of_thought.py` | Complete toolkit |
| **Safety** | `src/safety/` | `hallucination_guard.py`, `jailbreak_detector.py`, `constitutional_principles_scorer.py`, `canary_token_guard.py` | Defense-in-depth |
| **Agent** | `src/agent/` | `react_loop.py`, `workflow_shell.py`, `skill_library.py`, `tool_registry_dispatcher.py`, `surface_catalog.py` | Agent substrate exists; Brain-controller files are future work |
| **Training** | `src/training/` | `tst_trainer.py`, `amc_losses.py`, `amc_data.py`, `amc_dataset.py` | Complete pipeline |
| **Compression** | `src/compression/` | 21 techniques: pruning, quantization, distillation, SVD, DARE, TIES | Deployment-ready |
| **Federation** | `src/federation/` | `differential_privacy.py`, `secure_aggregation_v2.py`, `federated_banks.py` | Novel — federated AMC |
| **Model** | `src/model/` | `amc_transformer.py`, `mamba2_block.py`, `amc_ssm_layer.py`, `amc_promotion.py`, `sparse_moe.py` | Core backbone |
| **Eval** | `src/eval/` | `ablation.py`, `amc_memory_benchmark.py`, `serving_harness.py` | Experiment infra |
| **Multimodal** | `src/multimodal/` | Vision encoder integration | Partially implemented |
| **Interpretability** | `src/interpretability/` | Mechanistic analysis | Available |
| **Privacy** | `src/privacy/` | Privacy-preserving compute | Available |
| **Persona** | `src/persona/` | Personality modeling | Available |

### 2.2 Claims Ledger Status

| Claim | Text | Status | Action Needed |
|-------|------|--------|--------------|
| C1 | Three-tier memory per layer | **BACKED** | None |
| C2 | Mamba-2 ZOH without mean-pooling | **BACKED** | None |
| C3 | Differentiable promotion gate | **BACKED** | None |
| C4 | SDB fail-closed and replayable | **BACKED** | None |
| C5 | Trust-aware KV cache identity | **BACKED** | None |
| C6 | Constitutional memory non-evictable | **BACKED** | None |
| C7 | AMC-Memory benchmark harness | **BACKED** | None |
| C8 | Ablation with trained checkpoints | **PARTIAL** (oracle only) | Run T34 |
| C9 | Standard benchmark parity (GSM8K, MMLU) | **TODO** | Run T34 |
| C10 | Adversarial probes pass 6/6 | **BACKED** | None |

**Summary**: 8/10 claims fully backed. 2 claims need T34 experiments. **This is 2 weeks of work, not 6 months.**

### 2.3 Plans History (20+ Plans, April–May 2026)

Prior implementation plans in `docs/plans/` reveal the development arc:
- April 2026: V1 design, training loop fixes, learned optimizer
- May 2026: AMC first focused build, MOSAIC v2, PRAXIS alignment
- May 27-28 2026: DreamBank implementation, federated DP proof, TrustRAG, AMC v2

This history shows a project that has been actively built for 2+ months. The v4 plan should not restart — it should **complete what is 80% done**.

---

## PART III — THE AMC PAPER: CLOSE IN 6 WEEKS

### 3.1 What T33 Needs (Math Formalization)

`paper/sections/03_method.tex` already has 97 lines with core equations. T33 must add:

**Missing equations** (add to method section):
```latex
% Tier-3 consolidation threshold
\begin{equation}
  \mathrm{consolidate}(e) \iff s_e > \tau_c \;\text{and}\; \mathrm{trust}(e) \in \{\texttt{trusted}, \texttt{verified}\}
\end{equation}

% Constitutional memory invariant
\begin{equation}
  \forall t,\; \mathrm{evict}(m) = 0 \iff m \in \mathcal{M}_{\mathrm{const}}
\end{equation}

% Trust-aware cache identity
\begin{equation}
  \mathrm{key}(m) = H(\mathrm{content}(m) \;\|\; \mathrm{tier}(m) \;\|\; \mathrm{trust}(m) \;\|\; \mathrm{epoch}(m))
\end{equation}

% DreamBank preference pair selection
\begin{equation}
  (c^*, r^*) = \arg\max_{t_1,t_2} \bigl[f(p, g(p, t_1)) - f(p, g(p, t_2))\bigr] \cdot \mathbf{1}[\Delta > \delta_{\min}]
\end{equation}
```

**Files to modify**: `paper/sections/03_method.tex` (add 30–40 lines of equations)

### 3.2 What T34 Needs (Experiments)

The experiment infrastructure is close, but T34 is not "just run a fictional command." It is a live-path evidence tranche:

```bash
# 0. Sanity-check live launcher/config path
python scripts/validate_amc_forge_config.py
python src/training/launch_amc_training.py \
  --config configs/amc_forge_1b.yaml \
  --data data/tokenized/test_run \
  --log_dir logs/sanity_run \
  --max-steps 10 --batch-size 2

# 1. Train the T34 checkpoint family using explicit config overlays.
# There is no live --variant flag in launch_amc_training.py today; do not pretend there is.
# Create/locate one config per ablation condition, then launch each with the live launcher.
python src/training/launch_amc_training.py \
  --config configs/amc_forge_1b.yaml \
  --data data/tokenized/amc_forge_1b \
  --log_dir logs/t34/full_amc \
  --deepspeed configs/deepspeed_zero2.json

# 2. Run AMC ablation with trained weights (closes C8 if mode=engine and backend is real)
python scripts/run_ablation.py \
  --checkpoint logs/t34/full_amc/checkpoint-final.pt \
  --mode engine \
  --output docs/reproducibility/results/ablation_scores.jsonl

# 3. Summarize results for tables
python scripts/analyze_ablation.py \
  --results docs/reproducibility/results/ablation_scores.jsonl

# 4. Standard benchmarks close C9 only after the Aurelius lm-eval adapter is smoke-tested.
# Until then, C9 is benchmark-adapter-blocked, not model-evidence-complete.
```

**Estimated cost**: split into two estimates: TST-green path may reach the prior ~$40 target; TST-not-wired path must be costed from an actual sanity run throughput. Do not book the 2.5× multiplier until the readiness gate is green.

**Estimated time**: half-day launcher/TST readiness probe, then roughly 1 week training, 2 days eval, 2 hours tables if data and launcher paths are green.

### 3.3 Expected Results Structure

Based on claims C1-C7 and architecture theory, expected results pattern:

| Config | AMC-Memory | RULER NIAH | GSM8K | MMLU |
|--------|-----------|-----------|-------|------|
| Baseline (attention only) | ~0.45 | ~0.52 | ~0.48 | ~0.51 |
| Tier-1 only (SSM WM) | ~0.61 | ~0.64 | ~0.52 | ~0.53 |
| Tier-1 + Tier-2 (episodic) | ~0.72 | ~0.73 | ~0.56 | ~0.55 |
| Full AMC (all 3 tiers) | **~0.81** | **~0.79** | **~0.60** | **~0.58** |

*Note: These are expected orderings, not specific numbers. Actual values fill T34.*

Each tier adds meaningful capability. The progression baseline → tier1 → tier12 → full_amc is the core ablation story. If the pattern holds, it confirms per-layer memory is additive and composable.

---

## PART IV — THE DREAMBANK DISCOVERY

### 4.1 What DreamBank Actually Is

DreamBank (`src/alignment/dreambank.py`) implements **sleep-time preference consolidation via self-play**:

```
For each DreamSeed (a recent query prompt):
  1. Generate 4 responses at temperatures [0.2, 0.7, 1.0, 1.3]
  2. Score each response with score_fn (any preference model)
  3. Identify best and worst responses
  4. If margin > min_margin (0.05): form (chosen, rejected) pair
  5. Embed chosen response → upsert into HLMPreferenceBank
  6. Decay old bank entries at end of cycle
```

This is a **fully autonomous DPO-style preference learning cycle** that:
- Requires no human annotation
- Runs offline ("during sleep") without affecting inference
- Continuously improves preferences over time via decay + upsert
- Writes to the HLMPreferenceBank (the bridge to the HLM architecture track)

### 4.2 Why DreamBank Is Novel

No published work does exactly this combination:
1. **Multi-temperature self-play**: Using temperature variation as a diversity source (not beam search, not MCTS)
2. **Margin-filtered pair formation**: Only high-confidence preference pairs (margin ≥ 0.05) enter the bank — noise filtering via the model's own uncertainty
3. **Decaying preference bank**: Old preferences decay, new preferences overwrite — continuous preference evolution without catastrophic forgetting
4. **Sleep-cycle framing**: The cycle runs asynchronously, independent of inference — the model "dreams" about its recent interactions and consolidates preferences

Related work that it improves upon:
- **Self-play RL** (SPIN, Self-Play Fine-Tuning): DreamBank doesn't require iterative training, just bank updates
- **Online DPO**: DreamBank is offline (sleep-time), avoiding online instability
- **RLHF**: DreamBank eliminates the human annotation bottleneck entirely

### 4.3 DreamBank Extension Roadmap

**Extension 1 — Live score_fn** (1 week): Replace the injectable score_fn stub with the AMC model's own constitutional principles scorer (`src/safety/constitutional_principles_scorer.py`). The model uses its own values to judge its own outputs.

**Extension 2 — HLM feedback loop** (2 weeks): The HLMPreferenceBank already stores embeddings from DreamBank cycles. Feed these preferences as training signal for the HLM architecture exploration. This creates: **Aurelius dreams → HLM learns better architectures → APEX trains on better objectives**.

**Extension 3 — Curriculum seeding** (1 week): Instead of random recent queries as seeds, select seeds based on high-uncertainty outputs (low margin in previous cycles) — adaptive difficulty curriculum.

**Extension 4 — Multi-model dreaming** (2 weeks): Run DreamBank across the 150M, 1B, and 3B variants simultaneously. Compare preferences across scales — do smaller models dream differently? This is a measurable hypothesis.

### 4.4 DreamBank as a Novel Publication

**Title**: *"DreamBank: Autonomous Preference Consolidation via Sleep-Time Self-Play"*

**Key claims**:
1. A model can form valid preference pairs from its own temperature-varied outputs (testable: margin correlates with downstream win-rate)
2. Sleep-time preference consolidation improves alignment without human annotation
3. Decaying preference banks prevent preference staleness better than replay buffers

**Effort**: 2–3 weeks to run controlled experiments (DreamBank on vs. off, margin thresholds, decay rates)

**Venue**: NeurIPS 2027 main track or ACL 2027 findings


---

## PART IV-A — THE AMC v2 × UPD BRIDGE

The AMC v2 × UPD plan is not a competing master plan. It is the bridge between
Paper 1's memory architecture and the later platform papers: first make AMC more
capable, then make AMC decisions replayable.

### 4A.1 Mechanism decomposition

| Component | Problem solved | Owner |
|-----------|----------------|-------|
| GARB | Converts preference/memory bank state into a fast-weight associative read/write surface | AMC |
| CLX | Lets later layers read cross-layer memory context without writing to memory | AMC |
| H-MoE | Allocates expert compute tier from AMC bank confidence/alpha | AMC |
| A-05 harness | Proves memory-conditioned compute has signal without provenance | AMC |
| UPD schema/core/SDK | Makes decisions signed, replayable, and explainable | Audit layer |
| MemoryAdmission/AgentDecision adapters | Translate AMC events into DAGs after the decision | Audit layer |

This is the clean version of the architecture: memory first, audit second. UPD is
the receipt printer, not the cashier.

### 4A.2 Execution graph

```text
Current AMC paper evidence (T34)
  -> T35 paper assembly / arXiv
  -> A-00.5 AMC-native surface audit
  -> A-01 GARBMemory over HLM/DreamBank state
  -> A-02 CLXModulator
  -> A-03 HMoELayer
  -> A-04 AMCTransformer integration behind default-off flags
  -> A-05 AMC-only six-row ablation
  -> Gate G-A
  -> B-01/B-02/B-03 UPD schema/core/SDK
  -> C-01/C-02/C-03 AMC decision adapters
  -> D-01 combined AMC v2 × UPD ablation
  -> E-02 conformance suite and E-03 dashboard only after replay is deterministic
```

### 4A.3 Non-negotiable invariants

1. T34/T35 are not delayed by UPD. Current AMC paper closes first.
2. Track A completes before any UPD implementation commit.
3. `emit_upd` defaults to false everywhere. Zero overhead when disabled.
4. Raw memory text never enters a DAG; use `redaction="hash_only"` by default.
5. `DreamBankController.run_cycle(...) -> HLMPreferenceBank.upsert(...)` is the live write surface; no plan should call `admit()` unless a prior tranche creates it as a thin facade.
6. `GARBMemory.M` is not an `nn.Parameter` and must not appear in `model.parameters()`.
7. With all AMC v2 flags false, AMCTransformer behavior is v1-identical.
8. A-05 must show `garb_clx_aug > garb_clx_raw` under the deterministic fixture before D-01 exists.
9. UPD overhead must not change the AMC-only ablation conclusion.
10. Stage only files listed by the active tranche. Do not bundle dirty-tree drift.

### 4A.4 Paper placement

AMC v2 × UPD should be treated as **Paper 1B / standards artifact** until it has
evidence:

- If A-05 is strong but UPD is not ready: fold AMC v2 into the AMC follow-up paper.
- If D-01/G-D are strong: write a separate systems/conformance report around unified decision provenance.
- If UPD perturbs AMC behavior or cannot replay deterministically: keep it as internal instrumentation and do not make publication claims.

---

## PART V — FOUR PRIMARY PAPERS + ONE BRIDGE ARTIFACT

The prior plans proposed 5+ papers (DETER, GRPO Taxonomy, MemEval-AMC, SERE-Bench, MemGuardian). This is too fragmented. Four primary papers remain the publication spine. The AMC v2 × UPD material is imported as Paper 1B / standards artifact: it can become a separate systems paper only after Gate G-D, but it must not dilute the four-paper roadmap before then.

### Paper 1: AMC — The Memory Paper (PRIMARY, 6 weeks)
**Title**: *"Aurelian Memory Core: Per-Layer Differentiable Three-Tier Memory for Language Models"*

| Aspect | Detail |
|--------|--------|
| Status | T32 done, T34 blocks submission |
| Novel claims | Surprise-gated writes, per-layer SSM working memory, trust-aware consolidation, constitutional non-eviction |
| Evidence | C1–C7, C10 BACKED. C8/C9 need T34 |
| Effort remaining | T34: 2 weeks training + 2 days eval. T35: 1 week |
| Venue | NeurIPS 2027 / ICML 2027 |
| Risk | Low — paper is written, code is backed |
| Fallback | If numbers don't support all claims: publish as architecture paper with theoretical contributions |

**Key differentiators from prior memory work**:
- RAG is retrieval-augmented. AMC is **memory-native** — the architecture decides what to remember
- MemGPT has memory as a separate module. AMC is **per-layer** — every transformer layer has its own memory
- Titans has fast weights. AMC has **three explicit tiers** with formalized promotion semantics


---

### Paper 1B / Standards Artifact: AMC v2 × UPD — The Decision Provenance Bridge (POST-AMC PAPER)
**Working title**: *"Memory-Conditioned Compute with Replayable Decision Provenance"*

This is the imported `2026-05-28-amc-v2-upd-unified.md` workstream. It is not a
replacement for the current AMC paper. It starts after the AMC paper is table-locked
or submitted.

| Aspect | Detail |
|--------|--------|
| Status | Detailed plan exists; planned files are not present yet |
| Novel claims | Memory-conditioned compute allocation (GARB/CLX/H-MoE); signed replayable DAGs for memory and agent decisions |
| Evidence gate | A-05 AMC-only six-row ablation before any UPD implementation claim |
| UPD gate | D-01 reruns A-05 with provenance enabled and proves replay/explain/overhead constraints |
| Effort | ~4 weeks to Gate G-A; ~14 weeks to standards/conformance layer |
| Venue | Systems/conformance report, NeurIPS Datasets/Benchmarks-style artifact, or appendix to AMC/Safety depending on evidence |
| Risk | Scope contamination: provenance becomes a second architecture. Mitigation: UPD is audit-only and starts after G-A. |

**Primary result table**:

| Row | Claim tested |
|-----|--------------|
| baseline | No memory-conditioned compute |
| garb_only | GARB improves retrieval quality over the existing HLM/DreamBank bank surface |
| garb_hmoe | Bank signal changes expert-tier allocation |
| garb_clx_raw | Cross-layer modulation without bank augmentation |
| garb_clx_aug | GARB -> CLX coupling improves modulation |
| full_v2 | Memory × compute × context closed loop |

Threshold: `garb_clx_aug` must beat `garb_clx_raw` under a deterministic fixture.
That is the small sharp test. If it fails, the bridge is decorative and should stop.

---

### Paper 2: DreamBank — The Sleep Learning Paper (NEW, not in either prior plan)
**Title**: *"DreamBank: Autonomous Preference Evolution Without Human Annotation"*

| Aspect | Detail |
|--------|--------|
| Status | Core implemented (`dreambank.py`, `hlm_bank.py`) |
| Novel claims | Multi-temperature self-play generates valid preference pairs; decaying bank prevents preference staleness; zero human annotation |
| Evidence | Dry-run runner passes. Live experiments needed |
| Effort | 3–4 weeks experiments + 3 weeks writing |
| Venue | NeurIPS 2027 |
| Risk | Medium — needs controlled experiments showing DreamBank improves downstream alignment |
| Fallback | Workshop paper if main track results inconclusive |

---

### Paper 3: Safety Architecture — The Defense Paper (REVISED from DETER-only)
**Title**: *"Architectural Safety: Defense-in-Depth Across Training, Inference, and Memory"*

This paper is stronger than DETER alone because it presents a **complete safety architecture**, not a single method:

| Layer | Mechanism | Files | Novel Claim |
|-------|-----------|-------|-------------|
| L1: Training-time | MSM: Model Spec Midtraining | `src/training/alignment/ms_midtraining.py` | 54%→7% agentic misalignment |
| L2: Preference-time | DETER: Tampering detection + ZO-SimPO | Novel (DETER) | First alignment tampering defense |
| L3: Inference-time | Trust-Aware Cache Identity | `src/serving/amc_kv_cache.py` | KV keys bind trust state + revocation epoch |
| L4: Memory-time | Constitutional Memory | `src/memory/constitutional_memory.py` | Permanent non-evictable values |
| L5: Runtime | Jailbreak + Hallucination guards | `src/safety/jailbreak_detector.py` | Defense-in-depth at 5 levels |

**The unified claim**: No single attack surface compromises all five layers simultaneously. Each layer is independently published (AMC paper for L4, DETER paper for L2, MSM paper for L1) — this paper synthesizes them into a formal defense-in-depth framework.

| Aspect | Detail |
|--------|--------|
| Status | L4 BACKED (C6). L1 implemented. L2/L3/L5 exist in code |
| Novel claim | Five-layer safety architecture; no prior work formalizes all five jointly |
| Effort | 4–6 weeks (DETER detection module is the remaining piece) |
| Venue | NeurIPS 2027 Safety Track / IEEE S&P 2027 |
| Risk | Medium — DETER AUROC must reach >0.85 |

---

### Paper 4: Cognitive Architecture — The Brain Paper (FLAGSHIP, 6–9 months)
**Title**: *"Recursive Cognitive Self-Architecture: A Complete Brain for Language Models"*

| Aspect | Detail |
|--------|--------|
| Status | Fully specified in `docs/BRAIN_ARCHITECTURE.md`. Phases A-D partially in `src/agent/` |
| Novel claims | MCTS in pure embedding space on 1B model; online skill acquisition via momentum encoding; recursive self-upgrade loop |
| Evidence | `docs/BRAIN_ARCHITECTURE.md` exists; current `src/agent/` has `react_loop.py`, `workflow_shell.py`, `skill_library.py`, `tool_registry_dispatcher.py`, `surface_catalog.py`; named Brain-controller files are future work |
| Effort | 6–9 months (Phase A-D = 9 weeks; Phase E-H = 16 weeks) |
| Venue | NeurIPS 2027 |
| Risk | High (recursive upgrade). Fallback: Phase A-D only = publishable "Minimal Cognitive Architecture" |

**The minimum publishable brain** (Phase A–D, 9 weeks):
- Executive controller + basic reasoning loop
- Working memory (64-slot GRU buffer)
- Long-term memory (factual/procedural/episodic)
- Tool controller (existing tools, learned selection)

At Phase A–D completion: a 1B Aurelius Brain that solves multi-step reasoning benchmarks (BambooQA, MusiQue, HotpotQA) better than baselines. That result is a paper regardless of whether Phase E-H ships.

---

## PART VI — TWENTY NOVEL CONTRIBUTIONS FROM THE CODEBASE

Each of these falls out of what already exists. None appear in either prior plan.

### N1: Surprise-Gated Memory Writes (AMC Paper §3.2)
**What**: The SurpriseGate uses `sg[h_t]` (stop-gradient detached hidden state) to compute surprise — gradients from surprise loss don't flow into the main representation learner. A specific design choice with a specific rationale.
**Why novel**: No prior memory transformer uses information-theoretic surprise as the write controller. The stop-gradient boundary is a carefully designed inductive bias.
**Effort**: Documented in paper already. Add one paragraph explaining the stop-gradient choice.

### N2: Per-Layer Memory Specialization Hypothesis
**What**: Lower transformer layers store syntactic patterns; middle layers store semantic associations; upper layers store episodic memories. The AMC's per-layer architecture enables this — each layer has its own Tier-1/2/3 stack.
**Why novel**: Prior work (MemGPT, Longformer) attaches memory at a single layer. AMC's per-layer structure enables specialization.
**How to test**: Ablate which layers benefit most from each tier. Compare: does Tier-2 ablation hurt lower layers more than upper layers?
**Effort**: 1 week analysis + 1 paragraph in paper §6 (analysis section).

### N3: MCTS in Pure Embedding Space on Sub-7B Models
**What**: `src/reasoning/mcts_reasoner.py` implements Monte Carlo Tree Search in the transformer's embedding space — no symbolic search, no external verifier.
**Why novel**: Published MCTS work on LLMs (AlphaCode, etc.) uses symbolic search or large verifier models. Sub-7B embedding-space MCTS is unpublished.
**How to validate**: Compare MCTS reasoner vs. standard CoT on HotpotQA, MATH500. Show planning depth vs. accuracy curve.
**Effort**: 2 weeks eval + contributes to Paper 4 (Cognitive Architecture).

### N4: Tree-of-Thoughts + MCTS Hybrid
**What**: `src/reasoning/tot_planner.py` (Tree of Thoughts) and `src/reasoning/mcts_reasoner.py` can be composed. ToT proposes candidates; MCTS evaluates them.
**Why novel**: Prior work treats ToT and MCTS as alternatives. Composition is unstudied.
**Effort**: 1 week to compose + 1 week to evaluate.

### N5: GNN-Enhanced Reasoning
**What**: `src/reasoning/gnn_layer.py` and `src/reasoning/gnn_trainer.py` provide graph neural network components for reasoning. Problem structures (arithmetic, logic puzzles) can be represented as graphs; GNN layers can encode structural priors.
**Why novel**: GNN + LLM reasoning on structured problems is studied but the integration into a cognitive architecture with AMC memory is novel.
**Effort**: 2 weeks integration + evaluation on graph reasoning tasks.

### N6: Online Skill Acquisition Without Gradient Updates
**What**: The live skill surface starts at `src/agent/skill_library.py` / `src/agent/skill_catalog.py`; the planned claim is skill acquisition from successful trajectories via momentum-style dense encodings. No retraining should be required if the mechanism is validated.
**Why novel**: Continual learning papers require gradient updates. The Skill Library learns by pure embedding momentum — zero compute overhead at acquisition time.
**How to test**: Run 10,000 agent episodes. Plot task completion rate vs. skill library size. Show monotonic improvement.
**Effort**: 3 weeks experiments.

### N7: Federated AMC (Privacy-Preserving Memory Consolidation)
**What**: `src/memory/federated_banks.py` + `src/federation/differential_privacy.py` + `src/federation/secure_aggregation_v2.py`. Federated learning of AMC memory banks across private data sources.
**Why novel**: No published work applies federated learning specifically to a differentiable memory architecture. Privacy guarantees on memory consolidation are unstudied.
**How to validate**: Show AMC trained on federated shards converges to within ε of centrally-trained AMC.
**Effort**: 2–3 weeks (infrastructure mostly exists). Contributes to Paper 3 (Safety Architecture §7: Privacy).

### N8: Canary Token Guard for Memory Poisoning
**What**: `src/safety/canary_token_guard.py` — injects traceable canary tokens to detect memory poisoning attacks.
**Why novel**: Memory poisoning attacks on RAG are published; canary-based detection specific to a differentiable memory architecture is novel.
**Effort**: 1 week — write attack + defense experiments.

### N9: Constitutional Dimensions Scoring
**What**: `src/alignment/constitution_dimensions.py` and `src/safety/constitutional_principles_scorer.py` score outputs against a multi-dimensional constitutional specification.
**Why novel**: Standard constitutional AI uses binary good/bad. Multi-dimensional scoring (honesty, harmlessness, helpfulness, specificity, accuracy — independently scored) enables granular alignment.
**Effort**: 2 weeks to validate the multi-dimensional scoring on standard alignment benchmarks.

### N10: Absolute Zero RL Without Human Feedback
**What**: `src/alignment/absolute_zero.py` — RL from absolute zero, no human preference data, learning purely from environmental feedback.
**Why novel**: Absolute Zero was published recently (arXiv 2505.03335). Aurelius has an implementation. Running it with AMC's memory architecture could show memory-augmented zero-RL produces better long-horizon behavior.
**Effort**: 2–3 weeks experiments.

### N11: Chain-of-Hindsight as AMC Update Signal
**What**: `src/alignment/chain_of_hindsight.py` provides hindsight relabeling of trajectories. AMC's Tier-3 consolidation can be triggered by hindsight rewards — re-labeling past memories based on eventual outcome.
**Why novel**: Hindsight experience replay (HER) is well-known for RL. Applying it to a differentiable memory consolidation system is novel.
**Effort**: 1–2 weeks integration.

### N12: Stochastic Latent Recall
**What**: `src/reasoning/stochastic_latent_recall.py` — stochastic retrieval from AMC Tier-2/3 rather than deterministic top-k.
**Why novel**: Stochastic memory access during reasoning introduces useful exploration. Avoids getting stuck in high-confidence but wrong memory retrievals.
**Effort**: 1 week — integration + eval on memory-dependent reasoning tasks.

### N13: BOND + DreamBank Joint Training
**What**: `src/alignment/bond.py` (Best-of-N Distillation) combined with DreamBank sleep cycles. DreamBank forms preference pairs; BOND distills the best-scoring responses back into the base model.
**Why novel**: No published work combines sleep-time preference formation with best-of-N distillation in a continuous cycle.
**Effort**: 2 weeks to set up joint training loop.

### N14: CAI + AMC Memory Integration
**What**: `src/alignment/cai_pipeline.py` and `src/alignment/cai_v2_pipeline.py` (Constitutional AI). AMC's constitutional memory stores the principles permanently; the CAI pipeline uses them during self-revision.
**Why novel**: Standard CAI retrieves principles from context. AMC-CAI retrieves from non-evictable architectural memory — the principles are harder to suppress.
**Effort**: 1 week integration.

### N15: Adaptive Span + AMC Tier Selection
**What**: `src/model/adaptive_span_attn.py` — adaptive attention span combined with AMC tier selection. Short-span contexts use Tier-1 only; long-span contexts escalate to Tier-2/3.
**Why novel**: Adaptive compute + adaptive memory in a unified architecture.
**Effort**: 2 weeks.

### N16: Multi-Scale Temporal AMC
**What**: Across the 150M, 1B, 3B, and 7B variants, run DreamBank cycles at different timescales: 150M dreams every 10 minutes (fast, high-frequency); 7B dreams every 24 hours (slow, high-quality). Ensemble preferences across scales.
**Why novel**: Multi-scale temporal learning is studied in neuroscience but not in LLMs.
**Effort**: 2 weeks infrastructure.

### N17: Persona + AMC: Character Memory
**What**: `src/persona/` + AMC Tier-3. Long-term character traits stored as constitutional-class (non-evictable) Tier-3 entries. Character consistency across arbitrarily long conversations.
**Why novel**: Current persona work is prompt-based. Architectural persona encoding is unstudied.
**Effort**: 2 weeks.

### N18: AMC Interpretability via Layer-Wise Analysis
**What**: `src/interpretability/` + per-layer AMC analysis. Map which AMC layers activate most strongly for different memory types (factual vs. procedural vs. episodic). This generates mechanistic understanding of how the model uses its memory.
**Why novel**: Interpretability of differentiable memory architectures is unstudied.
**Effort**: 2 weeks.

### N19: Simulation-Based Curriculum for AMC Training
**What**: `src/simulation/` (exists in the repo) + AMC training. Generate synthetic environments that stress-test each AMC tier independently: Tier-1 stress (rapid context switching), Tier-2 stress (long-range dependencies), Tier-3 stress (multi-session continuity).
**Why novel**: Curriculum learning for memory architectures is unstudied.
**Effort**: 3 weeks.

### N20: CascadeBank — Multi-Scale Memory Routing
**What**: `docs/plans/2026-05-27-cascadebank-implementation.md` already exists as a plan. CascadeBank routes information to different memory tiers based on content type and temporal relevance — a learned cascade policy.
**Why novel**: Extends AMC's fixed tier hierarchy with a learned routing decision.
**Effort**: 3–4 weeks (plan already exists).

---

## PART VII — THE COMPUTE MULTIPLIER STACK

### 7.1 The Budget Math

Available budget: **~$350 cloud compute**

Without multipliers: this buys ~140 GPU-hours at $2.50/hr. A 6.8B pretrain needs 300+ GPU-hours. Budget is insufficient.

**Multiply the budget, not the model:**

| Technique | Multiplier | Status | Effort to Activate |
|-----------|-----------|--------|-------------------|
| TST (Token Superposition Training) | 2.5× | ✅ `tst_trainer.py` exists | Enable in config (30 min) |
| MIRA data selection | 2.0× | ❌ needs integration | 1 week |
| DualKV (shared prompt KV for RL) | 3.8× | ❌ needs integration | 2 weeks |
| BPPO (binary completion selection) | 6.0× GRPO only | ❌ needs integration | 3 days |

**Compound multiplier** (TST × MIRA × DualKV) = **19× effective compute**

**Budget after multipliers**: $350 × 19 = **$6,650 effective purchasing power**

**With this effective budget:**
- 1B full training suite: ~$40 (was ~$760)
- 3B training run: ~$150 (was ~$2,850)
- 6.8B final run: ~$400 (was ~$7,600) — still over budget in real terms, but feasible at 3B

**Decision**: TST activation is 30 minutes of work and immediately halves experiment costs. Do it before any training run. DualKV is 2 weeks and multiplies the budget 3.8×. Do it before scaling to 3B.

### 7.2 TST Readiness Probe (Immediate, but not blind activation)

`src/training/tst_trainer.py` is real. The missing piece is integration into the active training path. Current facts from the second pass:

- live forge launcher: `src/training/launch_amc_training.py`
- no live `src/training/train.py`
- `configs/amc_forge_1b.yaml` has no TST stanza
- `AMCTransformerConfig` has no `tst` field
- `AMCTransformer.forward(...)` does not currently accept `mode="superposition"`

So the correct move is a readiness probe, not a YAML edit:

```bash
python -c "from src.training.tst_trainer import TokenSuperpositionTrainer; print('TST module available')"
python scripts/validate_amc_forge_config.py
python src/training/launch_amc_training.py \
  --config configs/amc_forge_1b.yaml \
  --data data/tokenized/test_run \
  --log_dir logs/tst_readiness_probe \
  --max-steps 10 --batch-size 2
```

**Green condition**: the active training stack either already supports superposition mode or a tiny explicit integration tranche proves it without breaking standard AMC training.

**Red condition**: if the probe finds TST is not wired, T34 proceeds without TST rather than blocking the paper on a speedup integration. This is the right trade: publication evidence beats optimizing the evidence machine.

---

## PART VIII — THE APEX PIPELINE

### 8.1 The Three-Track Flywheel

From memory context: Aurelius (1B-class / 150M–7B family cognitive platform) → APEX (grand-unified alignment trainer, supersedes PRAXIS+MOSAIC v2, NeurIPS/ICML 2027) → HLM (new architecture exploration).

These three tracks form a **research flywheel**:

```
┌─────────────────────────────────────────────────────────────┐
│                    THE RESEARCH FLYWHEEL                     │
│                                                              │
│  Aurelius (Platform)                                         │
│    → runs DreamBank cycles                                   │
│    → generates preference pairs (chosen/rejected)            │
│    → writes to HLMPreferenceBank                             │
│         ↓                                                    │
│  HLM Architecture Exploration                                │
│    → uses preference bank as reward signal                   │
│    → explores new attention/memory architectures             │
│    → finds architectures that satisfy Aurelius's preferences │
│         ↓                                                    │
│  APEX (Grand Unified Alignment Trainer)                      │
│    → incorporates best HLM architecture discoveries          │
│    → trained with DETER + Constitutional Memory              │
│    → publishes as NeurIPS/ICML 2027 paper                   │
│         ↓                                                    │
│  Better Aurelius                                             │
│    → APEX's trained weights initialize next Aurelius version │
│    → DreamBank cycles improve faster                         │
│    → Loop repeats                                            │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 APEX Integration Points (Specific)

| Aurelius Finding | APEX Application | Transfer Mechanism |
|-----------------|-----------------|-------------------|
| AMC three-tier hierarchy | APEX memory backbone | Copy `src/model/amc_transformer.py` |
| DreamBank preference pairs | APEX training signal | HLMPreferenceBank → APEX trainer |
| Constitutional Memory | APEX alignment constraint | Copy/adapt `src/memory/constitutional_memory.py` |
| DETER defense | APEX tamper resistance | DETER module → APEX alignment pipeline |
| GRPO structural fixes (REFT, CalibAdv, AXPO) | APEX training stability | Apply same patches to APEX's training |
| MemEval-AMC benchmark | APEX memory evaluation | Run same benchmark on APEX |
| Federated AMC | APEX private deployment | Copy federated training pipeline |

### 8.3 Handoff Timeline

| Milestone | Date | Handoff |
|-----------|------|---------|
| AMC paper submitted | August 2026 | AMC architecture → APEX architecture decision |
| DreamBank experiments complete | September 2026 | Preference bank → APEX training pipeline |
| DETER detection >0.85 AUROC | November 2026 | DETER module → APEX safety layer |
| Brain Phase A-D complete | October 2026 | Cognitive control → APEX agent design |
| APEX paper draft | February 2027 | All Aurelius findings cited in APEX paper |

---

## PART IX — PHASE-BY-PHASE EXECUTION

### PHASE 0: Close What's Built (June 2026, 3 weeks)

**Goal**: Submit the AMC paper. Unlock the compute multiplier. Establish baselines.

| Task | File | Duration | Cost |
|------|------|----------|------|
| TST readiness probe; only enable if launcher/model consume it | `src/training/tst_trainer.py`, `src/training/launch_amc_training.py`, `configs/amc_forge_1b.yaml` | half day | $0 |
| Establish 1B baseline | `scripts/run_ablation.py --mode engine` after checkpoint smoke | 1 day | measured |
| Train checkpoint family / config overlays (close C8) | `src/training/launch_amc_training.py` | 1 week after probe | measured |
| Smoke-test and run standard benchmark adapter (close C9) | `lm_eval` or equivalent verified harness | 1 day + adapter gap if any | measured |
| Fill LaTeX tables (T34) | `paper/tables/ablation_main.tex` | 2 hours | $0 |
| T33: Add missing equations | `paper/sections/03_method.tex` | 3 days | $0 |
| T35: Final assembly + arXiv | `paper/main.tex` | 1 week | $0 |
| **Milestone** | AMC paper on arXiv | **3 weeks** | **~$50** |

---

### PHASE 1: DreamBank + AMC v2 Track A (July 2026, 4-6 weeks)

**Goal**: Run controlled DreamBank experiments and begin the AMC v2 substrate only
after the current AMC paper is table-locked/submitted. UPD implementation is still
forbidden in this phase; schema review only.

| Task | File | Duration |
|------|------|----------|
| Connect live score_fn to AMC model | `src/alignment/dreambank.py` | 1 week |
| Run DreamBank on/off comparison | `scripts/run_dreambank_cycle.py` | 1 week |
| Measure alignment improvement | `src/eval/amc_memory_benchmark.py` | 3 days |
| A-00.5 AMC-native surface audit | `docs/reports/amc-first-surface-audit.md` | 1 day |
| A-01 GARBMemory over HLM/DreamBank state | `src/memory/garb_memory.py` | 1 week |
| A-02/A-03 CLX + H-MoE | `src/model/clx_modulator.py`, `src/model/hmoe_layer.py` | 2 weeks |
| A-04/A-05 integration + AMC-only ablation | `src/model/amc_transformer.py`, `src/eval/amc_v2_ablation.py` | 1 week |
| MIRA data selection integration | `src/data/` | 1 week |
| DualKV integration | `src/training/` | 2 weeks (parallel, but not on the AMC v2 critical path) |
| **Milestone** | DreamBank experiments complete + Gate G-A ready | **4-6 weeks** |

---

### PHASE 2: Safety Stack Experiments (August — October 2026)

**Goal**: Validate the five-layer safety architecture. Begin Paper 3 (Safety Architecture).

| Task | File | Duration |
|------|------|----------|
| DETER detection module (PCA+dip test) | New: `src/safety/deter.py` | 3 weeks |
| ZO-SimPO loss smoothing | `src/training/simpo.py` | 1 week |
| Trust-Aware Cache evaluation | `src/serving/amc_kv_cache.py` | 1 week |
| UPD B-01/B-03 foundation after Gate G-A | `schema/upd-v1.json`, `crates/upd-core/`, `src/upd/` | 3 weeks |
| C-01/C-02 AMC decision adapters | `src/upd/adapters/` | 2 weeks |
| Adversarial attack suite (all 5 layers) | `src/safety/` | 2 weeks |
| MSM pipeline on 1B Aurelius | `src/training/alignment/ms_midtraining.py` | 3 days |
| Federated AMC privacy proof | `src/memory/federated_banks.py` | 2 weeks |
| **Milestone** | DETER AUROC >0.85, UPD replay gate green, full safety eval | **8-10 weeks** |

---

### PHASE 3: Minimum Publishable Brain (August — November 2026)

**Goal**: Implement Brain Architecture Phases A–D. Validate on multi-step reasoning benchmarks.

| Phase | Files | Duration | Tests |
|-------|-------|----------|-------|
| A: Executive + reasoning loop | `brain_controller.py` | 2 weeks | Multi-step math, instruction following |
| B: Working memory (64 slots) + verifier | `working_memory.py`, `verifier_critic.py` | 2 weeks | 2-step reasoning without context loss |
| C: Long-term memory (3 stores + reranker) | `long_term_memory.py`, `memory_manager.py` | 3 weeks | Cross-session recall, procedural retrieval |
| D: Tool controller (learned selection) | `tool_controller.py`, `tool_registry.py` | 2 weeks | Tool selection accuracy, retry behavior |
| **Milestone** | 1B Aurelius Brain on BambooQA, MusiQue, HotpotQA | **9 weeks** |

**Note**: Phase A–D is no longer described as mostly existing integration. The spec exists; the live agent substrate is `react_loop.py`, `workflow_shell.py`, `skill_library.py`, `tool_registry_dispatcher.py`, and `surface_catalog.py`. Brain-controller files are future surfaces and must be gated accordingly.

---

### PHASE 4: Scale + GRPO Fixes (September — October 2026)

**Goal**: Apply highest-ROI training improvements before 3B run.

| Task | Effort | ROI |
|------|--------|-----|
| REFT (~20 lines): first-token diversification | 3 days | +2–5% Pass@1 |
| CalibAdv (~50 lines): step credit assignment | 1 week | Fixes GRPO penalizing correct steps |
| AXPO (~80 lines): tool-call resampling | 1 week | +25% tool-use rate |
| BPPO: binary completion selection | 3 days | 6× GRPO speedup |
| DualKV: shared prompt KV | 2 weeks | 3.8× training speedup |
| **Milestone** | 3B training run with all fixes: ~$150 compute | **5 weeks** |

---

### PHASE 5: Papers + Brain E–H (November 2026 — April 2027)

| Task | Duration |
|------|----------|
| DreamBank paper writing | 4 weeks |
| Safety Architecture paper writing | 4 weeks |
| Brain Phase E: Agent Router (10 LoRA-tuned agents) | 3 weeks |
| Brain Phase F: Reasoning Core (self-consistency K=5, recursive thought) | 4 weeks |
| Brain Phase G: Reflection + online LTM update | 3 weeks |
| Brain Phase H: Recursive Self-Upgrade Loop | 4 weeks |
| Cognitive Architecture paper writing | 6 weeks |
| **Milestone** | All 4 papers submitted to NeurIPS 2027 | **May 2027** |

---

## PART X — DECISION RECORDS

### ADR-1: Publish AMC First (Not DETER)

**Decision**: The AMC paper (T32–T35) is submitted before any other paper.

**Rationale**: 8/10 claims backed. LaTeX written. T34 is 2 weeks of running experiments. The fastest path to a publication credit is to close this, not start a new paper. DETER has no prior art but also has no experiments.

**Rejected**: Publishing DETER first (no prior art is compelling but paper needs experiments).

---

### ADR-2: 1B → 3B → 6.8B Scaling (Not Jump to 6.8B)

**Decision**: All paper experiments at 1B. GRPO fixes validated at 1B. Final 3B run for paper submission. 6.8B only if compute budget allows after multipliers.

**Rationale**: TST × MIRA × DualKV = 19× budget. At this multiplier, 1B experiments cost $40 and 3B costs $150. 6.8B at the multiplier costs ~$400 — feasible only after both multipliers are active. Validate architecture at 1B, scale only what's confirmed.

---

### ADR-3: DreamBank as a Paper (Not Just Infrastructure)

**Decision**: DreamBank is elevated to Paper 2 (Sleep Learning).

**Rationale**: Sleep-time preference consolidation via self-play, multi-temperature margin filtering, and decaying preference banks are novel contributions. The HLM connection creates a unique story no competitor has.

---

### ADR-4: Four Papers, Not Five or One

**Decision**: Four papers with distinct boundaries: AMC (memory architecture), DreamBank (sleep learning), Safety (defense-in-depth), Cognitive Architecture (brain).

**Rationale**: Too many papers (5+) fragments the narrative. One grand paper is too risky (reviewers reject the whole). Four papers with citations between them tell a coherent story while spreading publication risk.

---

### ADR-5: GRPO Taxonomy Demoted to Appendix

**Decision**: The GRPO Structural Flaws Taxonomy (8 papers, 5 levels) becomes an appendix in the Cognitive Architecture paper, not a standalone publication.

**Rationale**: It's a meta-analysis of other labs' work. High effort for low credit when positioned against Aurelius's genuinely novel contributions.

---

### ADR-6: Constitutional Memory Is an Architectural Property

**Decision**: Constitutional memory is described in the AMC paper as an architectural safety property, not as a standalone contribution.

**Rationale**: It emerges naturally from the trust-aware Tier-3 design (BACKED claim C6). Positioning it as a "feature" undersells it — it's a consequence of the architecture, which is the right framing for an architecture paper.

---

### ADR-7: Federated AMC for Privacy Section

**Decision**: Federated AMC (`src/memory/federated_banks.py`, `src/federation/`) is included in the Safety Architecture paper as a privacy contribution.

**Rationale**: It directly addresses the MemGuardian gap from Plan 1 (memory privacy leakage AUC 0.99-1.00) with existing code.

---

### ADR-8: TST Is a Probe-Gated Multiplier, Not a Default Yet

**Decision**: TST (`tst_trainer.py`) is evaluated before expensive training, but it is not assumed active until the live launcher/model path proves it consumes superposition batches correctly.

**Rationale**: The module exists, but current config/launcher/model contracts do not yet make it a simple switch. Counting a 2.5× multiplier before the wiring is green is how plans quietly lie to budgets. If the readiness probe is green, use TST. If it is red, run T34 without it and preserve the paper schedule.

---

### ADR-9: DreamBank Uses Constitutional Principles Scorer as Live score_fn

**Decision**: Replace the injectable `score_fn` stub in DreamBankController with `src/safety/constitutional_principles_scorer.py`.

**Rationale**: The model judges its own outputs against its own constitutional principles. This creates a closed improvement loop: the model dreams according to its own values, not an external evaluator.

---

### ADR-10: APEX Integration Begins After AMC Paper Submission

**Decision**: No Aurelius findings are integrated into APEX until the AMC paper is submitted.

**Rationale**: Prevents scope contamination. The AMC paper's contributions belong to Aurelius. After submission, they can be freely incorporated into APEX without concern about scoop risk.


---

### ADR-11: AMC v2 × UPD Is Imported After AMC Paper Closure

**Decision**: The AMC v2 × UPD plan is merged into this roadmap as a post-AMC-paper bridge workstream, not as a replacement for T34/T35.

**Rationale**: The current AMC paper has a near-term evidence blocker. Starting a 14-week provenance/architecture build before closing that blocker would create a second critical path and risk never publishing the work that is already 80% done.

**Rejected**: Running UPD implementation in parallel with T34. Schema reading is allowed; code waits until the AMC-only Gate G-A sequence begins.

---

### ADR-12: UPD Is Audit/Provenance, Not Memory Ownership

**Decision**: AMC owns memory admission, promotion, quarantine, routing, and compute allocation. UPD records signed DAGs after AMC decisions and provides replay/explain/conformance.

**Rationale**: A provenance layer that changes the decision it records is not provenance; it is a second system in a trench coat. The paper claim is cleaner if AMC first proves signal without UPD, then UPD proves auditability without perturbing that signal.

**Rejected**: Letting UPD define a new admission gate, raw memory store, or routing controller.

---

## PART X-A — HARD GATES, KILL SWITCHES, AND EVIDENCE LADDER

This section is the anti-fantasy layer. It converts the roadmap into decisions that can actually stop work.

### Evidence ladder

| Level | Name | Meaning | Allowed claim |
|-------|------|---------|---------------|
| L0 | File exists | Source path is present | "implemented surface exists" |
| L1 | Unit tests pass | Local behavior is tested | "component behavior is backed" |
| L2 | Integration smoke | Live launcher/eval path runs end-to-end on tiny data | "execution path is real" |
| L3 | Real checkpoint evidence | Trained checkpoint produces non-oracle results | "paper evidence candidate" |
| L4 | Ablation survives | Effect holds under seeds/ablations/confidence intervals | "paper claim" |
| L5 | External benchmark parity | Standard benchmark adapter validated | "general capability claim" |
| L6 | Reproducibility pack | Commands, configs, hashes, tables, and logs replay | "submission-ready" |

No section may promote a result above its evidence level. This is mostly a guard against accidentally turning a nice unit test into a NeurIPS sentence.

### Kill switches

| Workstream | Kill / pause condition | Action |
|------------|------------------------|--------|
| TST | Readiness probe shows launcher/model path does not consume TST cleanly within half day | Run T34 without TST; schedule TST as post-paper infra |
| T34 | Engine-mode ablation cannot run from trained checkpoint | Freeze paper claim at architecture/theory; open benchmark-adapter gap |
| AMC v2 Track A | A-05 fails to show `garb_clx_aug > garb_clx_raw` | Stop UPD implementation; either revise GARB/CLX or demote bridge to notes |
| UPD | `emit_upd=False` has nonzero overhead or changes outputs | Abort UPD runtime integration; keep schema-only audit work |
| DreamBank | Margin-filtered pairs do not correlate with downstream win-rate | Keep as infrastructure/workshop; do not make main-track alignment claims |
| Brain | Phase A-D does not beat direct/react baselines on multi-step tasks | Drop Phase E-H; write minimal cognitive architecture only if A-D has signal |
| APEX handoff | APEX integration would alter unpublished Aurelius paper claims | Wait until Aurelius claim is submitted/archived |

### Executor guardrails

1. Start every tranche with `git status --short --branch` and a path-existence check for every file named in that tranche.
2. Stage only the active tranche files. No drive-by cleanup, no opportunistic architecture surgery.
3. Any config key must be proven consumed by code before it is counted as enabled.
4. Oracle/smoke results can unblock wiring, but cannot populate paper tables.
5. Every paper-facing number needs: config path, command, checkpoint hash/path, seed, raw JSONL, and table row.
6. Planned files are explicitly future tense. Never write "exists" unless the file existed in the verification pass.
7. AMC owns memory and compute decisions. UPD records them. If a diff violates that sentence, reject it.

---

## PART XI — RISK REGISTER

| # | Risk | Likelihood | Impact | Mitigation | Trigger Date |
|---|------|-----------|--------|-----------|-------------|
| R1 | T34 ablation numbers don't show monotonic tier improvement | Medium | High | Publish as architecture paper with theoretical contributions; still novel | Jul 1 2026 |
| R2 | DETER detection AUROC stuck below 0.85 | Medium | High | Fall back to theoretical tamper bounds paper | Nov 2026 |
| R3 | Liger-Kernel incompatible with MLA | High | Medium | Use only for non-MLA ops; ~20% speed loss vs. TST-only | Jul 2026 |
| R4 | DreamBank alignment improvement not measurable | Medium | Medium | Workshop paper; position as infrastructure | Sep 2026 |
| R5 | AMC scooped (someone publishes per-layer differentiable memory first) | Low | Very High | Submit AMC paper to arXiv immediately after T34 | Monthly arxiv check |
| R6 | 6.8B compute budget exceeded | High | High | Fix at 3B; use TST+DualKV+MIRA multipliers to maximize 3B results | Oct 2026 |
| R7 | Recursive Brain Upgrade Loop doesn't improve accuracy | High | Low | Drop Phase H; Phase A-D is independently publishable | Feb 2027 |
| R8 | APEX scope conflict with Aurelius claims | Medium | Medium | Submit Aurelius papers before APEX integration; clear provenance | Month-by-month |
| R9 | ICLR 2027 deadline (Oct 2026) incompatible with timeline | Very High | High | Target NeurIPS 2027 for all papers; ICLR 2027 workshop for AMC | Aug 2026 checkpoint |
| R10 | Research cron data (52MB) overwhelms synthesis capacity | Current | Medium | Implement coverage tracking matrix; auto-filter below score threshold | Monthly |
| R11 | HLM architecture exploration conflicts with Aurelius paper claims | Medium | Medium | Clear provenance documentation; Aurelius contributions are pre-APEX | Ongoing |
| R12 | GRPO fixes break existing AMC training stability | Medium | Medium | Apply REFT + CalibAdv first (safest). BPPO/AXPO only after validation | Sep 2026 |
| R13 | AMC v2 delays current AMC paper | Medium | High | T34/T35 close first; AMC v2 starts after paper table lock/submission | Immediate |
| R14 | UPD becomes a second memory/admission owner | Medium | High | Gate G-A before UPD implementation; UPD audit-only invariant in every tranche | Phase 1 |
| R15 | UPD overhead changes AMC-only conclusions | Medium | High | D-01 must rerun A-05 and abort if conclusions change | Phase 2 |
| R16 | Raw memory text leaks into DAG fixtures | Low | Very High | `redaction="hash_only"` by default; conformance test fails on raw text | Phase 2 |
| R17 | TST counted before it is wired into the live launcher | High | Medium | Half-day readiness probe; if red, run T34 without TST | Immediate |
| R18 | Paper table accidentally uses oracle/smoke numbers | Medium | Very High | Evidence ladder: only L3+ engine results enter paper tables | T34 |
| R19 | Brain section treats future files as existing files | Medium | Medium | Spec-driven phrasing; path-existence check before each tranche | Phase 3 |
| R20 | Benchmark adapter failure masquerades as model failure | Medium | High | Smoke-test benchmark adapter separately before interpreting C9 | Phase 0 |

---

## PART XII — SUCCESS METRICS

### What "Done" Looks Like for Each Paper

**AMC Paper (Target: August 2026 arXiv)**
- [ ] TST readiness resolved explicitly: green and used, or red and bypassed without delaying T34
- [ ] C8 closed: ablation shows baseline < tier1 < tier12 < full_amc on AMC-Memory
- [ ] C9 closed: GSM8K ≥ 0.55, MMLU ≥ 0.55 for full_amc 1B through a smoke-tested benchmark adapter
- [ ] All 6 adversarial probes pass (T29, already done)
- [ ] Paper tables use only engine-mode/non-oracle outputs with config, checkpoint, seed, and raw JSONL recorded
- [ ] Paper PDFs without errors: `make -C paper pdf`
- [ ] arXiv abstract gets >50 citations in 6 months

**DreamBank Paper (Target: November 2026 arXiv)**
- [ ] DreamBank on vs. off shows +X% win-rate on alignment benchmark
- [ ] Margin filtering (min_margin=0.05) is shown to improve signal quality
- [ ] Bank decay rate ablation shows optimal τ
- [ ] HLM preference transfer shown to improve HLM architecture search

**AMC v2 × UPD Bridge (post-AMC-paper gate)**
- [ ] Gate G-A green: GARB, CLX, H-MoE, AMCTransformer integration, and A-05 all pass
- [ ] A-05 shows `garb_clx_aug > garb_clx_raw` under deterministic fixture
- [ ] All AMC v2 flags default false and preserve v1 behavior
- [ ] UPD emits no DAG when `emit_upd=False`
- [ ] D-01 proves UPD replay/explain works without changing AMC-only conclusions
- [ ] No raw memory text appears in any DAG fixture

**Safety Architecture Paper (Target: January 2027 arXiv)**
- [ ] DETER detection AUROC ≥ 0.85 on synthetic tampering attacks
- [ ] MSM: Aurelius shows 54%→7% misalignment (replicate Anthropic result on Aurelius)
- [ ] Constitutional memory: adversarial fine-tuning cannot overwrite constitutional entries (10/10 attempts blocked)
- [ ] Five-layer defense: no single attack compromises all layers simultaneously

**Cognitive Architecture Paper (Target: April 2027 arXiv)**
- [ ] Brain Phase A-D: solves HotpotQA ≥ 0.62 (baseline 1B ≈ 0.45)
- [ ] MCTS planning: improves over greedy on 5-step reasoning tasks
- [ ] Skill acquisition: 1000+ skills acquired, performance improves monotonically
- [ ] Recursive Upgrade Loop (if Phase H ships): ≥3 successive improvements over 3 upgrade cycles

---

## PART XIII — THE IMMEDIATE ACTION LIST

## Conclusion
## Conclusion

This roadmap consolidates the AMC memory architecture, DreamBank sleep learning, safety stack, and cognitive brain development into a coherent, staged plan. By following the immediate actions and adhering to the hard gates, we aim to submit the four primary papers (AMC, DreamBank, Safety Architecture, Cognitive Architecture) by mid‑2027, while establishing a reusable research platform for subsequent work. The plan remains extensible: future extensions (e.g., latent‑preference diffusion, paradigm‑mixture architecture) are deliberately placed **after** core deliverables are secured.

In priority order, sorted by impact/effort ratio:

```
Priority 1 — DO TODAY (half day)
  Run truth-surface freeze + TST readiness probe
  Decision: TST-green means use it; TST-red means proceed to T34 without blocking on speedup work
  
Priority 2 — DO THIS WEEK
  Train the T34 checkpoint family through src/training/launch_amc_training.py and explicit config overlays
  Smoke-test the standard benchmark adapter path before claiming C9
  
Priority 3 — WEEK 2
  Fill paper tables from T34 results
  Complete T33: add 4 equations to paper/sections/03_method.tex
  Complete paper/sections/02_related_work.tex
  
Priority 4 — WEEK 3  
  T35 final assembly: make -C paper pdf
  Upload to arXiv (AMC paper submitted)

Priority 4.5 — IMMEDIATELY AFTER AMC PAPER TABLE LOCK / SUBMISSION
  Run A-00.5 AMC-native surface audit
  Start AMC v2 Track A: GARB -> CLX -> H-MoE -> A-05
  UPD implementation remains blocked until Gate G-A; schema review only
  
Priority 5 — MONTH 2
  DreamBank live score_fn integration
  MIRA data selection integration
  DualKV integration (2 weeks)
  
Priority 6 — MONTH 3
  Brain Phase A (Executive Controller + reasoning loop)
  DETER detection module
  REFT + CalibAdv GRPO fixes
  
Priority 7 — MONTH 4-6
  Brain Phase B-D
  Safety Architecture experiments
  3B training run with all fixes
  
Priority 8 — MONTH 7-12
  Brain Phase E-H
  All four papers in writing phase
  NeurIPS 2027 submissions
```

---

## PART XIV — FUTURE ARCHITECTURE EXTENSIONS (Post-Core Deliverables, Gated)

This part captures the two major independent implementation plans discovered on 2026-05-29 (`~/Desktop/AI Plans/2026-05-28-latent-preference-diffusion.md` and `~/Desktop/AI Plans/2026-05-28-paradigm-mixture-architecture.md`). Both exceed 1400 lines with rigorous "Review Addendum" constraint sections. 

**Placement rule (first-principles)**:
- These are **not** on the critical path for the four primary papers or the current AMC paper T34/T35.
- They are **explicitly sequenced after** AMC paper submission + DreamBank/Safety/Brain Phase A-D gates have green evidence.
- LPD is a scientific **deepening of the existing DreamBank track** (Paper 2 extension material).
- PMA is an **evolutionary layer on top of the current HLM heterogeneous architecture exploration** that already feeds the APEX flywheel.

No new publication claims or main-track experiments from either extension are allowed until their imported gates are satisfied.

### 14.1 Latent Preference Diffusion (LPD) — Sleep-Time Preference Refinement via Langevin Dynamics

**Source**: full 1414-line plan + 10-item Review Addendum (SHA snapshot taken 2026-05-29).

**Thesis**: During sleep cycles, high-salience memories are encoded into a 512-dim preference embedding space. A small energy-based model (EBM) trained on (chosen, rejected) pairs guides projected Langevin MCMC. Embeddings drift toward lower-energy (higher-alignment) regions with **zero gradient steps or optimizer updates** on the main model. Refined exemplars are re-admitted via the existing safety pipeline.

This is the strict "no-weight-mutation" formalization of the multi-temperature self-play already running in `src/alignment/dreambank.py`.

#### Immediate architectural relationships
- Extends: `src/alignment/dreambank.py`, `src/memory/hlm_bank.py`, `src/memory/manager.py`.
- New surfaces placed exactly as the source plan specifies: `plugins/agents/{dream_bank,preference_encoder,preference_ebm,latent_preference_diffusion,lpd_dream_bank}.py`, `scripts/train_preference_ebm.py`, `src/eval/dream_bank_eval.py`.
- The PreferenceEBM is a **new lightweight vector EBM**, deliberately separate from the token-sequence EBM at `src/model/energy_based_model.py`.

#### Imported Hard Gates — 10 Non-Negotiable LPD Constraints (verbatim from source Review Addendum)

| # | Constraint (short) | Evidence Required Before Any Paper Claim |
|---|--------------------|------------------------------------------|
| LPD-G1 | Energy semantics: always store both raw `energy` (lower=better) **and** calibrated `score`. Never threshold on raw energy. | Calibration report + strict unit test on held-out pairs |
| LPD-G2 | Projected Langevin on the unit sphere (encoder already normalizes). Gradient tangent projection + final L2 norm mandatory. | `test_lpd_projected_langevin` passes after 100 steps |
| LPD-G3 | Track the monotonic best-energy point along trajectory. Emit best-so-far if final energy regresses beyond tolerance. | `monotonic_best` field + injection test |
| LPD-G4 | Six explicit admission gates (finite, score, duplicate, safety, provenance, write success). One failure = full reject. | All six verdicts recorded in every `DreamCandidate`; failure injection tests |
| LPD-G5 | Never swallow exceptions in production (`except: score=0.0` forbidden). | Structured rejection reasons; diagnostic on bad input |
| LPD-G6 | Full provenance on every candidate (source_ids, energies, score, seed, config_hash, verifier, embedding payload). | Dataclass + round-trip serialization test |
| LPD-G7 | Encoder is an explicit seam (`PreferenceEmbeddingProvider` protocol). Char tokenizer only for tests. | Protocol definition + live-model adapter test |
| LPD-G8 | EBM must be calibrated before any memory write. Required report: AUROC, histograms, chosen threshold. | `train_preference_ebm.py` must emit `ebm_calibration_report.json` |
| LPD-G9 | Minimum 5-row ablation table **before** any novelty claim (baseline memory, EBM-only, random noise, LPD w/o safety, full LPD). | Table in every experiment log + downstream metric |
| LPD-G10 | Zero weight mutation invariant: main model always `eval()` + `no_grad()`. EBM gradients allowed; **no optimizer touch** on backbone or encoder. | Parameter-hash diff test before/after every cycle |

**Gating**: No LPD experiment may feed a paper claim until **Gate LPD-GA** (all 10 gates green + measurable alignment lift on real memories from a trained checkpoint with zero backbone gradient updates).

**Effort (post-core)**: 6–8 weeks for a first complete green run following the test-driven structure of the source plan.

### 14.2 Paradigm Mixture Architecture (PMA) — Evolutionary Search over Heterogeneous Layer Sequences

**Source**: full 1445-line plan + 8-item Review Addendum (SHA snapshot 2026-05-29).

**Thesis**: Grow the `HLMProfile.layer_pattern` vocabulary far beyond the current small set (`dense`, `moe`, `mamba`, ...). Add `neural_ode`, `linear_attn`, `gnn_lang`, etc. Then run a grammar-constrained evolutionary searcher that discovers optimal heterogeneous layer sequences under explicit task + compute budgets, going meaningfully beyond fixed hybrids such as Jamba/Samba.

Continues the live HLM exploration (`src/model/hlm.py`) that already feeds the APEX architecture flywheel.

#### Verified live surface constraints (2026-05-29)
- Must extend, never regress, the existing HLM vocabulary.
- Factory signatures for `MambaLayer`, `LinearAttentionLayer`, `NeuralODE*`, MoE*, etc. are immutable — the plan's adapter code already incorporates the correct constructor wrappers.
- `ReMoDELayer` remains metadata-only until a tensor `nn.Module` adapter exists.

#### Imported Hard Gates — 8 Non-Negotiable PMA Constraints

| # | Constraint | Evidence Required |
|---|------------|-------------------|
| PMA-G1 | Existing HLM names (`dense`, `moe`, `mod`, `mamba`, `neural_ode`, `linear_attn`, `gnn_lang`) remain first-class. `remode` metadata-only until tensor adapter. | Registry test + construction round-trips for all legacy names |
| PMA-G2 | Every factory must return a true `nn.Module` that accepts `d_model=32` etc. and emits plain `[B,T,D]`. Use `_TensorOutputAdapter` for layers returning aux losses. | Instantiation + forward test on real 2×8×32 tensor for **every** registered paradigm |
| PMA-G3 | Strict causality. `GNNLangLayer` KNN **never** leaks future tokens. Equivalent future-leakage tests for all mixers. | Causal KNN test + analogs for every paradigm |
| PMA-G4 | Genotypes come from a **grammar repair function** with hard rules (first/last stable, max consecutive ODE=1, GNN fraction ≤15%, no GNN before layer 2, global mixer every 4 layers, etc.). | `repair_pattern` property tests + all random/mutate paths pass repair |
| PMA-G5 | Random-token perplexity is smoke-test only. Production fitness requires a true multi-axis proxy suite (LM loss, copy/induction, long-context retrieval, entity-relation QA, latency, memory) + explicit penalties. | Configurable `ProxyFitness` accepting task suite object |
| PMA-G6 | Fitness **explicitly penalizes** latency, memory, and extra parameters. | Penalty terms present and ablated in every publishable run |
| PMA-G7 | Search must be fully seed-reproducible + cached (`random.Random(seed)`, `tuple(pattern)→fitness` cache, JSONL checkpointing). Interrupted runs must resume identically. | Reproducibility + caching + resumption tests |
| PMA-G8 | Novelty discipline: publishable claim is "search-discovered heterogeneous sequences under explicit constraints", **not** "we mixed layers". Must beat dense-only, Mamba-heavy, fixed Jamba-style, and hand-written HLM profiles. | Required comparative baseline table in every result |

**Gating**: No serious compute or publishable claims until **Gate PMA-GA** (all 8 gates green + at least one reproducible discovered pattern that statistically wins the penalized multi-axis suite while respecting budgets and differing from hand-authored profiles).

**Effort (post-core)**: 8–10 weeks for first evolutionary run at 1B proxy scale.

### 14.3 Cross-Extension Rules

- LPD and PMA are independent; neither blocks the other.
- Both ultimately feed the same long-term flywheel (better memory → richer preferences → better searched architectures → stronger cognitive brain).
- Both inherit the global invariants: AMC owns admission/routing/compute; no weight mutation during sleep/inference (except inside the explicit LPD EBM path under the no-grad guard).
- A third future extension may be added only after it arrives with an equally rigorous 6–10 item first-principles Review Addendum and only after one of the two current GA gates has passed.

---

## ADRs Added 2026-05-29 for Extensions

**ADR-13**: LPD is treated as a DreamBank deepening. It may only become a first-class follow-on paper after Gate LPD-GA + the core DreamBank paper is submitted or table-locked.

**ADR-14**: PMA evolutionary search may not begin serious runs until current HLM profiles have strong multi-axis baselines **and** the full grammar + penalized fitness is implemented.

**ADR-15**: Both extensions default to "infrastructure / future work" status for publication purposes. Zero claims in any primary paper until their respective GA gates are green and documented.

---

## Updated Risk Register (new rows)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R21 | LPD 10-gate set proves significantly harder than expected; "LPD paper" slips past 2027 | Medium | Medium | Treat strictly as DreamBank v2 infrastructure improvement until GA passes cleanly |
| R22 | PMA search without grammar/penalties simply rediscovers expensive variants of Jamba and over-claims | Medium | High | Enforce PMA-G4/G6/G8 before any public result; mandatory comparative baselines |
| R23 | LPD + PMA compete for the same small post-AMC compute budget | High | Medium | Strict sequencing: LPD after DreamBank paper, PMA after HLM baseline + Brain A-D. Budget allocation review at each gate |

---

## Updated Success Metrics — Future Extensions

**LPD (DreamBank v2)** — only consider publication after Gate LPD-GA green:
- All 10 imported LPD gates have committed passing tests + calibration/ablated reports.
- End-to-end cycle on real memories from a trained checkpoint yields ≥ +4% win-rate lift on a held-out preference benchmark **with zero backbone gradient steps**.
- Provenance, monotonic-best, and projected-sphere logic survive systematic failure injection.

**PMA** — only after Gate PMA-GA green:
- All 8 imported PMA gates satisfied with reproducible evolutionary runs.
- At least one discovered grammar-respecting pattern statistically outperforms dense-only, Mamba-heavy, fixed-Jamba-style, and current hand-written HLM profiles on the penalized multi-axis suite.
- Search is fully deterministic under seed + checkpoint-resumable.

---

## Extended Priority List (Future Work)

```
Priority 9 — MONTH 13+ (strictly after all 4 primary papers + Brain Phase A-D green)
  LPD Track
    • Execute the full 7-task TDD plan from the 2026-05-28-latent-preference-diffusion.md source ONLY after DreamBank paper submission AND all 10 LPD-G* gates pass.
    • Target publication criterion: measurable alignment lift from diffusion with zero weight mutation.

  PMA Track
    • Execute full adapter + GNN/ODE layers + ParadigmModel + grammar-constrained searcher + multi-axis penalized fitness ONLY after HLM baselines are strong on the multi-axis suite AND all 8 PMA-G* gates pass.
    • Target: first publishable result on search-discovered heterogeneous architectures.
```

---

## APPENDIX C — SOURCE PLAN CROSSWALK (expanded 2026-05-29)

| Source section | Destination in v4 | Canonical decision |
|----------------|-------------------|--------------------|
| AMC v2 × UPD §0 thesis | Merge addendum + Part IV-A | AMC first, UPD second (audit layer only) |
| ... (unchanged prior entries) | ... | ... |
| latent-preference-diffusion.md (full 1414-line plan + 10-item Review Addendum) | **Part XIV §14.1 + ADRs 13/15 + new risk R21 + LPD-specific success metrics + Priority 9** | DreamBank deepening. All 10 LPD-G* gates are **hard binding prerequisites**. Integration gated behind core DreamBank paper + LPD-GA. No separate paper claims before GA. |
| paradigm-mixture-architecture.md (full 1445-line plan + 8-item Review Addendum) | **Part XIV §14.2 + ADRs 14/15 + new risk R22 + PMA-specific success metrics + Priority 9** | HLM evolutionary extension. All 8 PMA-G* gates are **hard binding prerequisites**. Integration gated behind strong HLM baselines + PMA-GA. Strict novelty discipline vs. prior hybrids required. No claims before GA. |

Future agents needing detailed tranche lists should consult the original source files under the constraints listed here.

---

```
aurelius/
├── paper/                    ← AMC paper (T32 done, T34 needed)
│   ├── main.tex
│   ├── sections/             ← 01-07 + appendix (248 lines total)
│   └── tables/               ← ablation_main.tex (needs T34 numbers)
│
├── src/
│   ├── model/                ← amc_transformer.py, mamba2_block.py, amc_ssm_layer.py
│   │                            amc_promotion.py, sparse_moe.py, mla_wrapper.py
│   ├── memory/               ← amc_tier2.py, amc_tier3.py, constitutional_memory.py
│   │                            hlm_bank.py, federated_banks.py, episodic_memory.py
│   ├── alignment/            ← dreambank.py, cai_v2_pipeline.py, absolute_zero.py
│   │   └── praxis/           ← PRAXIS (superseded by APEX, but code exists)
│   ├── reasoning/            ← mcts_reasoner.py, tot_planner.py, self_consistency.py
│   │                            gnn_layer.py, gnn_trainer.py, chain_of_thought.py
│   ├── safety/               ← jailbreak_detector.py, hallucination_guard.py
│   │                            constitutional_principles_scorer.py, canary_token_guard.py
│   ├── training/             ← tst_trainer.py ✅, amc_losses.py, amc_data.py
│   ├── agent/                ← react_loop.py, workflow_shell.py, skill_library.py, tool_registry_dispatcher.py
│   ├── federation/           ← differential_privacy.py, secure_aggregation_v2.py
│   ├── compression/          ← 21 techniques: pruning, quantization, SVD, DARE, TIES
│   ├── eval/                 ← ablation.py ✅, amc_memory_benchmark.py ✅
│   └── serving/              ← amc_kv_cache.py (trust-aware cache identity)
│
├── docs/
│   ├── ARCHITECTURE.md       ← Inference engine (HOPE, ELF, Pareto router)
│   ├── BRAIN_ARCHITECTURE.md ← 10-module cognitive system spec
│   └── plans/                ← 20+ implementation plans (April-May 2026)
│
└── scripts/
    └── run_dreambank_cycle.py ← DreamBank dry-run runner (live with real model)
```

---

## APPENDIX B — VENUE DEADLINES (NeurIPS 2027 Track)

| Venue | Abstract | Paper | Notes |
|-------|---------|-------|-------|
| ICLR 2027 Workshop | Sep 2026 | Oct 2026 | AMC paper (fast-track if T34 done by Sep) |
| ICML 2027 | Jan 2027 | Feb 2027 | DreamBank paper |
| NeurIPS 2027 Main | May 2027 | Jun 2027 | All 4 papers |
| ACL 2027 Findings | Mar 2027 | Apr 2027 | DreamBank alternative venue |
| IEEE S&P 2027 | Oct 2026 | Dec 2026 | Safety Architecture alternative venue |

**Critical decision**: If T34 experiments are done by September 2026, submit AMC paper to ICLR 2027 workshop. Otherwise, NeurIPS 2027 main track. The ICLR workshop track is a lower bar and gets AMC into the conversation sooner.


---

## APPENDIX C — SOURCE PLAN CROSSWALK

This crosswalk keeps the two source plans combined without duplicating every
tranche inline.

| Source section | Destination in v4 | Canonical decision |
|----------------|-------------------|--------------------|
| AMC v2 × UPD §0 thesis | Merge addendum + Part IV-A | AMC first, UPD second |
| Coupling points CP-1..CP-5 | Part IV-A + Paper 1B | Keep as adapter design; do not let UPD own memory |
| Track A A-00..A-05 | Phase 1 + Paper 1B | First post-paper build sequence |
| Gate G-A | Phase 1 milestone + success metrics | Required before any UPD implementation |
| Track B/C | Phase 2 safety/provenance lane | Starts after G-A only |
| Track D/E | Late standards/product lane | Starts after replay/signature gates |
| Abort criteria §11.5 | Risk register R13-R16 + Part IV-A invariants | Preserve and enforce |
|| Agent invariants §13 | Part IV-A invariants | Imported as non-negotiables |
|| latent-preference-diffusion.md | Future work: latent‑preference diffusion | Placeholder for diffusion‑based preference modeling |
|| paradigm-mixture-architecture.md | Future work: paradigm mixture architecture | Placeholder for exploring mixed‑paradigm model families |

If future agents need tranche-level instructions, use
`docs/plans/2026-05-28-amc-v2-upd-unified.md` as the detailed runbook and this
v4 plan as the priority/schedule authority.

---

*This plan was generated 2026-05-29. It consolidates: 2 prior strategic plans, verified `ARCHITECTURE.md` / `BRAIN_ARCHITECTURE.md` context, paper section reads (introduction, method, experiments, training), DreamBank implementation read, plans-directory inventory, and src/ module inventory. A second-pass audit found repo-root MODEL_CARD/GAP_LEDGER/CLAIMS_LEDGER absent in this checkout, so claims must be re-grounded from live files before submission.*

*Next action remains: Run the truth-surface freeze + TST readiness probe. If green, use TST for T34; if red, run T34 without it. The AMC paper is still the only active blocking deliverable. PART XIV items (LPD + PMA) are deliberately future gated extensions (see Gate LPD-GA and Gate PMA-GA), not on the critical path for the 2026-2027 paper cycle.*
