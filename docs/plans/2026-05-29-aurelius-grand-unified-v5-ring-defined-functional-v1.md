# Aurelius Grand Unified Plan v5 — Ring-Defined Functional v1 (Fully Realized Autonomous Cognitive Agent)

**Date:** 2026-05-29  
**Version:** v5 (complete replacement for v4)  
**Status:** Authoritative master plan — Ring-scoped, mechanism-first, submission-grade epistemic hygiene  
**Maintained by:** Christien Antonio  
**Branch baseline:** `clean/amc-curation-20260521-101220` (HEAD 46ee2f13) + all historical AI Plans (April–May 2026)

**Intellectual Honesty Statement**  
This is not an aspirational wishlist. It is a ruthlessly scoped, evidence-gated roadmap whose success is defined by one thing: a reproducible, closed cognitive loop in which a 1B-class (then 1.5B) Aurelius model uses its own per-layer differentiable memory hierarchy, surprise-gated writing, and autonomous DreamBank sleep cycles to measurably outperform identical-architecture baselines on multi-step agentic tasks — all with full auditability and zero weight mutation during sleep refinement. Everything else is deferred.

---

## PART I — THE FUNCTIONAL v1 VISION (RING 1 DEFINITION)

### 1.1 What "Fully Functioning Like a Capable Autonomous Agent" Actually Means

Ring 1 v1 is the smallest closed, measurable, reproducible system that proves the Aurelian Memory Core (AMC) thesis is load-bearing in real agent behavior.

**Ring 0 (Current Truth Surface — April–May 2026 execution)**  
- 1B-class dense AMC (≈24 layers, d_model 1536–2048, GQA 16/8 or 32/7, SwiGLU, RoPE θ=500k, Pre-RMSNorm, tied embeddings).  
- Per-layer SSM working memory + stop-gradient surprise head + Gumbel straight-through differentiable promotion gate.  
- SDB propose–verify–commit contract for episodic (Tier-2) writes.  
- Trust-aware LTS (Tier-3) with constitutional memory retrieval (non-evictable principles).  
- Live DreamBank controller (`src/alignment/dreambank.py`, `hlm_bank.py`) performing multi-temperature self-play and margin-filtered preference consolidation with HLM ingestion.  
- Agent primitives exist in design (Observe/Think/Act/Reflect + MCTS value head + tool cross-attn + skill acquisition) per MODEL_CARD.md.  
- Evidence level: L2 (integration smoke on components) on most surfaces; L3 on some.

**Ring 1 — Functional v1 Closed Cognitive Loop (Non-Negotiable Target)**  
The minimal artifact that earns the name "v1 functional autonomous agent":

1. A trained 1B-class AMC checkpoint (engine_mode inference) whose per-layer surprise gates actively decide what enters episodic memory during multi-step agent execution (4–12 step realistic traces: HotpotQA multi-hop, tool-use sequences, long-horizon planning).
2. The same agent comfortably uses retrieved episodic + constitutional memory inside the forward pass and shows statistically significant improvement over an identical-architecture "no-memory" or "oracle-only" baseline on the same tasks (ablation required).
3. DreamBank sleep cycles run on the agent's own traces, produce margin-filtered (or future LPD-refined) preference data, and deliver measurable downstream win-rate or task-success lift on held-out splits — with the backbone in eval + no_grad the entire time (zero weight mutation during the refinement phase).
4. The full Observe → Think (MCTS with learned value head) → Act (tool call + skill retrieval) → Reflect (critic + self-correction) loop executes end-to-end, with memory writes and retrievals as visible, auditable parts of the trace.
5. All numbers satisfy the Mandatory Validation & Reproducibility Protocol (below).
6. A single reproducibility pack (config + exact launch command + checkpoint SHA + seeds + raw JSONL artifacts + table rows) replays the key deltas on one H100 in <4 hours.

**Strict Scope Boundary for Ring 1**  
- No Tier-3 Atlas expansion.  
- No full 5-layer safety synthesis paper claims.  
- No LPD or PMA experiments (they remain post-GA only, per PART IX).  
- No production serving heroics (engine_mode + minimal harness is sufficient).  
- No broad benchmark parity as primary goal (C9 remains supporting).  
- Agent traces must be realistic but not "full day in the life" — 4–12 step is the Ring 1 bar.

**Ring 1.5 (Months 6–9, after Ring 1 green)**  
- Exact same functional loop at 1.5B scale.  
- First paper-grade ablations for AMC v1 paper + early DreamBank material.  
- Clean 1B → 1.5B migration playbook (weight mapping, curriculum continuation).

**v2 and Beyond (9+ months)**  
- Scale to 3B–7B.  
- Full Brain Phases C–H (including recursive upgrade loops).  
- LPD + PMA (only after their respective GA gates).  
- APEX grand alignment trainer.  
- Rich skill acquisition and composition.  
- Production-grade serving + BFF completion.  
- All four primary papers + potential journal extensions.

**Kill Criterion**  
If, after 6 months of focused Ring 1 work, the closed agent + DreamBank loop cannot be demonstrated with clean ablations on at least one trained 1B checkpoint, the project pivots to "AMC as strong architectural paper only" and de-scopes agent ambitions for v1.


---

## PART II — MASTER TIMELINE SYNTHESIS (Every Individual Plan in ~/Desktop/AI Plans, April–May 2026)

Method: exhaustive file-by-file audit of every planning artifact produced during the April–May 2026 build. Each entry records the file name, one-sentence summary at time of writing, current evidence/code state as of 2026-05-29, Ring assignment, and precise contribution (or irrelevance) to the Ring 1 closed agent + DreamBank loop. Grouped by original creation waves.

**April 6, 2026 — Foundation Wave**
- `2026-04-06-aurelius-v1-design.md`: High-level codex design establishing four-layer stack (Rust/Python/Node/React) and 150M–7B model family. Current state: L1 document. Ring 0 foundation only; sets the target family from which the 1B Ring 1 checkpoint is drawn. Contribution to Ring 1: target architecture spec, not evidence.
- `2026-04-06-aurelius-v1-implementation.md`: Implementation blueprint for the multi-layer architecture at build time. Current state: L1 document. Ring 0 defines where the serving harness and engine_mode currently live. Contribution to Ring 1: informs where the closed-loop harness must execute.

**April 7–8, 2026 — Engineering Fix Wave**
- `2026-04-07-heavens-gate-handoff.md`: Integration handoff for Rust-side data-engine to Python model bridge. Current state: L1/L2 on Rust crate side; Python wiring partially landed in `src/model/amc_transformer.py`. Ring 0/Bridge only. Contribution to Ring 1: establishes the exact memory-page-table contract that DreamBank + AMC write paths must honor during traces.
- `2026-04-07-heavens-gate-integration-design.md`: Design counterpart to the handoff above. Current state: L1 design doc. Ring 0. Contribution to Ring 1: page-table schema informs reproducibility-pack artifact format.
- `2026-04-07-training-loop-fix-design.md`: Training-launch repair design. Current state: L1 design, superseded by live `src/training/launch_amc_training.py`. Ring 0 only. Contribution to Ring 1: confirms the launcher path used in the 30-day plan is the actual wired path, not a hypothetical.
- `2026-04-07-training-loop-fix-plan.md`: Execution plan for training loop fixes. Current state: L1 plan. Ring 0. Contribution to Ring 1: none directly; defines constraints on optimizer state that Ring 1 DreamBank sleep must respect (zero weight mutation).
- `2026-04-08-learned-optimizer-fix-handoff.md`: Learned-optimizer handoff with optimizer compatibility constraints. Current state: L1. Ring 0 only. Contribution to Ring 1: reinforces that optimizer state must not be present during DreamBank sleep cycles; supports the no_grad invariant.

**April 20–21, 2026 — Contract + Harvest Wave**
- `2026-04-20-harvest-cycles-124-127-design.md`: Cycle pipeline design for harvest subsystem. Current state: L2/L3 on pipeline semantics. Ring 0 only. Contribution to Ring 1: harvest is the upstream source of DreamBank raw episodes; cycle semantics are a known working surface but not in scope for Ring 1 evidence.
- `2026-04-20-harvest-implementation.md`: Implementation of harvest cycles 124–127. Current state: L2/L3 working code. Ring 0. Contribution to Ring 1: confirms episode ingestion pipeline works; DreamBank depends on this format.
- `2026-04-21-aurelius-canonical-interface-contract.md`: Canonical interface contract with API surface definitions. Current state: L3 on API contract. Ring 0 boundary doc. Contribution to Ring 1: JSON schema for trace artifacts and reproducibility packs derives from these contract rules.
- `2026-04-21-aurelius-canonical-interface-contract.schema.json`: JSON schema for the contract. Current state: L3. Ring 0. Contribution to Ring 1: schema fields (`config_path`, `launch_command`, `checkpoint_sha`, `seeds`, `artifact_path`) map directly to Rule 1 (full invocation record).
- `2026-04-21-aurelius-canonical-interface-contract.prompt.yaml`: Prompt template for contract. Current state: L2/L3. Ring 0. Contribution to Ring 1: informs the format of reproducibility pack manifests.
- `2026-04-21-aurelius-canonical-interface-contract.json`: JSON instance of the contract. Current state: L3. Ring 0. Contribution to Ring 1: same as schema above.

**May 9, 2026 — Alignment/Mosaic/Praxis Wave**
- `2026-05-09-aurelius-alignment-design.md`: Alignment design before AMC-first curation. Current state: L1/L2 design, superseded by AMC-first primacy. Ring boundary: post-Ring 1 once R1-GA is green. Contribution to Ring 1: contains APEX tier taxonomy later imported into PART IX; otherwise not active.
- `2026-05-09-aurelius-alignment-v2-design.md`: Alignment v2 design. Current state: L2. Same Ring assignment as above; informs PART IX post-core guardrails only.
- `2026-05-09-mosaic-v2-implementation.md`: MOSAIC v2 implementation plan. Current state: unimplemented plan; MOSAICTrainer is explicitly NOT a Ring 1 path (per MASTER-IMPLEMENTATION-PLAN.md Track B). Ring 2+.
- `2026-05-09-praxis-implementation.md`: Praxis implementation plan. Current state: Praxis modules exist (`src/alignment/praxis/`) but are not Ring 1 dependencies. Contribution: some components (MTAH, steering reward, expert-safety affinity) are later composed by APEX post-core. No contribution to Ring 1 gate metrics.

**May 20, 2026 — APEX Wave**
- `2026-05-20-apex-design.md`: APEX design document. Current state: unimplemented design. Explicitly Ring 2+ (MASTER-IMPLEMENTATION-PLAN.md Track B). Not contributing to Ring 1. Crystal-clear quarantine: no APEX tasks before Ring 1 green.
- `2026-05-20-apex-implementation.md`: APEX implementation plan. Current state: unimplemented. Ring 2+.

**May 28–29, 2026 — Bridge/Post-Core Plans Wave**
- `2026-05-28-latent-preference-diffusion.md`: Full LPD implementation plan with Review Addendum (10 required improvements). Current state: L0/L1 (plan), no committed code. **Quarantine**: Ring 2+ per PART IX. Contribution: the 10 Review Addendum gates become binding pre-GA constraints on LPD.
- `2026-05-28-paradigm-mixture-architecture.md`: Full PMA implementation plan with Review Addendum (8 required improvements). Current state: L0/L1 (plan), no committed code. **Quarantine**: Ring 2+ per PART IX. PMA claims explicitly post-AMC paper and post-Gate GA.
- `2026-05-28-preference-diffused-paradigm-mixture.md`: Cross-extension plan combining LPD + PMA. Current state: L0. **Quarantine + extra constraint**: No joint LPD+PMA paper claim allowed until both individual GA gates are green plus one clean joint ablation vs. pure AMC + pure DreamBank baseline (see PART IX CROSS-EXTENSION RULE).
- `2026-05-29-aurelius-grand-unified-v4.md`: Prior master plan, now superseded by v5. Current state: historical synthesis. Ring 0–1 crosswalk source. v4 risk register rows R1–R23 carried forward into PART X as condensed.

**Historical Pattern Statement**
Fresh design work is consistently excellent. Integration, evidence, and reproducibility degrade. v5 enforces Ring boundaries structurally. Every plan since April 6 adds capability; none adds paper-ready evidence at L4+. The 2026-05-29 evidence ladder in CLAIMS_LEDGER.md confirms zero L4/L5/L6 claims: the entire gap is training + ablation + benchmark adapter smoke + reproducibility-pack stress test. v5 concentrates all effort there.

---

## PART III — TOOLING, DELEGATION & COST STRATEGY (Ring-1 Maximal)

**Non-negotiable rule:** Every tool investment must accelerate R1-GA/GB/GC or the Validation Protocol. No exceptions.

**Tier 1 — P0, install this week**
- Liger-Kernel, lm-eval-harness, triton: all three currently missing (MASTER-IMPLEMENTATION-PLAN.md Phase 0). Unblocked by a single 30-minute readiness probe and a half-day install sprint. Without them, R1-GA (trained checkpoints + adapter) and C9 adapter validation cannot proceed at required speed.
- DeepSeek API: primary bootstrap for reasoning traces, synthetic agent demonstrations, and critique/safety verifiers. Budget $50–150/mo.

**Tier 2 — P1, install after P0 green**
- Unsloth for LoRA/QLoRA speedup on 1B fine-tuning (Ring 1 only).
- SGLang for prefix caching in engine_mode serving.
- Langfuse only if traces need runtime observability (use file-based JSONL artifacts first; avoid extra SaaS dependency before Ring 1 gates).

**Delegation rules for Ring 1**
- High-value research/synthesis: delegate to Claude Code or OpenCode.
- Repeatable training/eval runs: Cursor CLI Composer 2.5 workflow for code paths; manual terminal for `scripts/run_ablation.py --mode engine` to guarantee full invocation record per Rule 1.
- Every 3-day tranche ends in a green test + JSONL evidence artifact. Tool velocity without evidence is treated as failure.

**Realistic cloud compute envelope for the decisive 30–60 day window**
See Appendix B. Baseline: heavy DeepSeek reasoning + local iteration, plus two planned H100/A100 rented bursts for T34 training + CB-07 serving harness runs. Total Ring 1 decisive compute: 200–400 GPU-hours → $800–1,800. Waste factor 30–50%. 1.5B upsize only after Ring 1 green, using exact weight-mapping deltas from 1B config.

---

## PART IV — CODE REVIEW & TECHNICAL DEBT LEDGER (Ring-1 Constraints Only)

The repository currently has ~20 uncommitted dirty files (MASTER-IMPLEMENTATION-PLAN.md §2.3). They must be categorized before any new feature branch is created.

**P0 (blocks Ring 1 gates if left unresolved)**
- `src/model/__init__.py`: may expose AMCContract changes that break the adapter validation contract defined in §1.2 Rule 3.
- `src/serving/aurelius_server.py`: engine_mode path must remain load-bearing for R1-GA.
- `src/memory/amc_tier2.py`: core Tier-2 implementation; any undeclared semantic shift invalidates B1/B4 evidence.
- `src/training/trainer.py`: training loop must match the launcher path in the 30-day plan.
- `.github/workflows/ci.yml`: CI must pass green before PR; if it does not cover the new P0 installs, that is a P0 debt.
- `tests/integration/test_lambda_attention_integration.py`: integration test debt that could block adapter validation if it fails silently.

**P1 (blocks speed but not correctness)**
- `src/alignment/simpo.py`: alignment module drift; not Ring 1 critical unless SimPO is used in the DreamBank ablation (it is not in v1 scope).
- `src/memory/hlm_bank.py`: HLM bank changes; must stay compatible with `hlm_bank_adapter.py` for R1-GA.
- `gateway/aurelius_api.py`: API surface drift; increases merge conflict risk for CB-07 harness.
- `middle/src/server.ts`, `middle/src/provider_router.ts`, `middle/src/routes/auth.ts`: BFF drift; not Ring 1 critical unless CB-07 needs BFF routes (it does not; CB-07 uses Python harness only).
- `src/ui/session_manager.py`: UI drift; zero Ring 1 relevance.

**Recommendation:** Living `TECHNICAL_DEBT.md` at repo root, updated at each monthly gate. Condensed risk register in PART X. Every P0 item must have a resolved SHA before R1-GA is declared green.

---

## PART V — NOVELTY BOOST MATRIX (Ring-1 Scoped Claims Only)

The goal is to identify which claims in the eventual AMC/DreamBank paper are novel and which are incremental. Claims outside Ring 1 are excluded from this matrix.

| Claim Category | Specific Claim | Novel? | Basis | Ring Status |
|---|---|---|---|---|
| AMC Architecture | Per-layer differentiable three-tier memory (SSM → episodic → LTS) with surprise-gated writes in a dense 1B transformer | Yes, at this scale and integration | No prior work combines all three tiers per-layer in a 1B dense model; related work is fragmented (MemTransformer, Differentiable Neural Computer, Compressive Transformer) | Ring 1: prove with ablation |
| AMC Architecture | Gumbel straight-through differentiable promotion gate from Tier-1 to Tier-2 | Yes | Prior work uses hand-tuned thresholds or Hebbian updates; differentiable promotion is novel in this exact form | Ring 1: prove with ablation |
| DreamBank | Sleep-time preference consolidation with zero weight mutation (backbone in eval + no_grad) | Incremental (self-play RLHF exists) but novel integration with AMC traces | Self-play preference generation exists; the specific loop of AMC trace → DreamBank margin filtering → downstream lift without weight update is the novel integration | Ring 1: prove with trace-to-lift ablation |
| Agent Loop | End-to-end Observe→Think→Act→Reflect with auditable memory reads/writes in 1B model | Yes, at this scale and with AMC as the memory substrate | Agent frameworks (AutoGPT, BabyAGI) are modular, not end-to-end trainable; Voyager has skill acquisition but no per-layer AMC | Ring 1: prove with 4–12 step traces |
| Safety | Trust-aware LTS with non-evictable constitutional principles | Incremental on constitutional AI, novel on memory tier | Constitutional AI methods (Bai et al.) apply to weights; applying non-evictable principles to a parametric memory tier is a new safety primitive | Ring 1: prove with red-team probes |
| Inference | engine_mode inference supporting AMC memory reads | Incremental (speculative decoding, KV cache work exists); the AMC-specific memory read integration is novel | PackKV, NestedKV, etc. address KV cache; none address per-layer differentiable memory slots | Ring 1: prove with latency/throughput numbers |

**Bottom line:** Ring 1 has three defensible novelty claims: (1) the full AMC three-tier per-layer integration in a trained 1B model, (2) the differentiable promotion gate, (3) the closed agent + DreamBank loop with zero-weight sleep refinement. The rest are incremental or deferred.

---

## PART VI — THE FOUR PRIMARY PAPERS (Ring-1 Scopes Only)

**Paper 1: AMC v1 — Differentiable Memory Architecture for Autonomous Agents**
- Scope: 1B-class dense model, per-layer three-tier memory, surprise gate, SDB contract, promotion gate, ablation vs no-memory and oracle-only baselines, standard benchmark parity (C9 supporting).
- Venue target: NeurIPS 2027 (main conference) or ICLR 2027.
- Ring boundary: Paper 1 closes when R1-GA is green. No LPD/PMA/APEX claims appear in this paper.
- Evidence requirement: L4+ on all table rows per Validation Protocol.

**Paper 2: DreamBank v1 — Sleep-Time Preference Consolidation for Autonomous Agents**
- Scope: Multi-temperature self-play, margin filtering, HLM ingestion, downstream task-success lift on held-out agent traces, zero weight mutation proved.
- Venue target: NeurIPS 2027 or ICML 2027.
- Ring boundary: Paper 2 closes when R1-GB is green. No LPD/PMA claims.
- Evidence requirement: L4+ with ablation vs random preference injection and vs no sleep.

**Paper 3: Agentic RL with AMC — Closing the Observe-Think-Act-Reflect Loop**
- Scope: End-to-end agent traces with MCTS value head, tool cross-attention, skill retrieval, critic-based self-correction. Memory writes and retrievals are auditable trace components.
- Venue target: ICLR 2027 or NeurIPS 2027.
- Ring boundary: Paper 3 closes when R1-GC is green. Paper 1 and 2 results are prerequisites.
- Evidence requirement: L4+ on multi-step task success vs memory-ablated baselines.

**Paper 4: MemEval-AMC — Benchmark for Parametric Memory Consolidation**
- Scope: Standalone benchmark paper evaluating AMC-style memory across 12 capability axes and 4-stage lifecycle. Can be drafted in parallel after Paper 1 table lock, using the same 1B checkpoint.
- Venue target: ICLR 2027 Workshop or NeurIPS 2027.
- Ring boundary: Paper 4 is Ring 1.5 / supporting; does not block v1 agent completion.

**Deferred papers (post-Ring 1, explicitly Ring 2+)**
- ZO-SimPO method paper (requires LPD at L4+).
- SERE-Bench (small-model reasoning efficiency).
- MemGuardian (DP for parametric memory).
- APEX grand trainer paper.
- LPD-only or PMA-only papers (only after respective GA gates).

---

## PART VII — HARD GATES, KILL SWITCHES, EVIDENCE LADDER (Ring-1 Maximal)

**Evidence Ladder (from CLAIMS_LEDGER.md)**
- L0: Spec only (no code) — not paper eligible.
- L1: Design documented — not paper eligible.
- L2: Integration smoke test passes — proves wiring only, not paper eligible.
- L3: Benchmarked on real data — paper eligible with caveats.
- L4: Ablated (vs baseline) — paper eligible.
- L5: Reproducible (3+ seeds) — paper eligible.
- L6: Published (arXiv/conference) — paper eligible.

**Current Ring 1 Gap (as of 2026-05-29)**
- Paper-ready claims (L3+): 8/32 (25%) per CLAIMS_LEDGER.md.
- Ring 1 target (L4+): 0/32 (0%). This is the gap to close.
- Specifically: C5 (DreamBank lift) is L0; D5 (auditable traces) is L0; B4 (no-memory ablation) is L2; B1 (surprise-gated writes) is L2.

**Hard Gates**

| Gate ID | Name | Required Artifact | Evidence Level | Deadline | Status |
|---|---|---|---|---|---|
| R1-GA | AMC Architecture Gate | 1B checkpoint + ablation table (no-memory, oracle-only, full-AMC) + paper.method.tex lock | L4+ on all rows | TBD (30-day plan) | ❌ Not started |
| R1-GB | DreamBank Sleep Gate | DreamBank cycle JSONL + downstream lift table + zero-weight mutation proof | L4+ | TBD | ❌ Not started |
| R1-GC | Closed Agent Loop Gate | 4–12 step agent trace log with auditable memory reads/writes + task-success delta vs ablated baselines | L4+ | TBD | ❌ Not started |
| R1-GD | Reproducibility Pack Gate | tarball/Git tag replayable on one H100 in ≤4 hours per Rule 6 | L5 | Before arXiv | ❌ Not started |

**Kill Criterion**
If, after 6 months of focused Ring 1 work, the closed agent + DreamBank loop cannot be demonstrated with clean ablations on at least one trained 1B checkpoint, the project pivots to "AMC as strong architectural paper only" and de-scopes agent ambitions for v1.

---

## PART VIII — PRIORITIZED 30-DAY EXECUTION PLAN + MONTHLY GATES (Ring-1 Maximal)

**Week 1: Foundation + P0 Installs**
- Day 1–2: Install Liger-Kernel, triton, lm-eval-harness. Run sanity check on 150M model.
- Day 3: Resolve dirty-tree P0 items. Commit as `chore: resolve working-tree drift pre-Ring1`.
- Day 4–5: Create Ring 1 branch from HEAD 46ee2f13. Freeze truth surface.
- Day 5: Run `make test` on clean tree. Must be green.

**Week 2: Training Pipeline Green**
- Day 6–7: Prove `src/training/launch_amc_training.py` + TST readiness. Half-day probe: confirm superposition batches reach model or T34 proceeds without TST.
- Day 8–12: Train first Ring 1 checkpoint (T34-family variant). DeepSeek bootstrap for synthetic demonstrations if needed.
- Day 12: Verify checkpoint SHA + config hash + seed log. This is Rule 1 compliance.

**Week 3: Ablation + Benchmark Adapter**
- Day 13–15: Run `scripts/run_ablation.py --mode engine` with trained checkpoint. Emit real JSONL (no oracle/smoke in paper tables per Rule 2).
- Day 16–17: Smoke lm-eval-harness adapter on ≤150M model first (Rule 3). Then run C9 benchmarks on 1B checkpoint.
- Day 17: Generate `docs/reproducibility/results/ablation_scores.jsonl` and `benchmark_scores.jsonl`.

**Week 4: DreamBank Integration + Trace Format**
- Day 18–20: Wire DreamBank controller to use 1B checkpoint traces. Run 10–20 sleep cycles on real agent traces.
- Day 21–22: Define Ring 1 trace format (JSONL with auditable memory read/write events). Run 4–12 step agent traces.
- Day 23–24: Compute downstream lift (win-rate or task-success) on held-out split. Zero weight mutation proof (no_grad + eval mode).
- Day 25: Reproducibility pack tarball + 4-hour replay test.

**Month-1 Gate Checklist**
- [ ] R1-GA: Ablation table with L4+ numbers, no-memory baseline, oracle-only baseline, full-AMC condition, 3+ seeds.
- [ ] R1-GB: DreamBank JSONL + downstream lift table ≥ +X pp over no-sleep baseline.
- [ ] R1-GC: Agent trace log ≥ 100 traces, 4–12 steps, memory reads/writes visible, task-success delta significant vs ablated baseline.
- [ ] R1-GD: Reproducibility pack verified on clean H100 in ≤4 hours.
- [ ] All 6 Validation Protocol rules satisfied for every number in papers.

---

## PART IX — FUTURE ARCHITECTURE EXTENSIONS (LPD + PMA + APEX) — STRICTLY POST-CORE

**Status: Quarantined. Ring 2+. May not be touched until all Ring 1 gates (R1-GA/GB/GC/GD) are green and Paper 1/2/3 tables are locked.**

LPD and PMA retain all 18 imported Review Addendum gates verbatim from their respective May 28 desktop plans:

**From `latent-preference-diffusion.md` Review Addendum (10 gates):**
1. Fix score semantics (energy lower-is-better, score higher-is-better; never compare raw energy with `>= admission_threshold`).
2. Use projected Langevin on the unit sphere (tangent-space gradient before update).
3. Keep monotonic candidates (track best-energy point; if `final_energy > initial_energy + tolerance`, reject as `rejected_by_diffusion`).
4. Add explicit admission gate (finite tensor check, energy/score threshold, duplicate/similarity check, safety verifier, provenance completeness, memory write success).
5. Do not swallow exceptions (structured rejection reasons; no `except Exception: score = 0.0`).
6. Preserve provenance (`DreamCandidate` must carry `source_memory_ids`, `initial_energy`, `final_energy`, `score`, `diffusion_seed`, `lpd_config_hash`, `verifier`, and embedding reference).
7. Make the encoder a seam (`PreferenceEmbeddingProvider` protocol; char-token encoder acceptable only for unit tests).
8. Calibrate the EBM before use (held-out validation report: pairwise accuracy/AUROC, energy histogram, admission threshold).
9. Ablate before claiming novelty (baseline memory, EBM-score-only, random-noise diffusion, LPD without safety gate, full LPD).
10. No weight mutation invariant (Dream cycles run with main model in eval/no_grad; EBM may require input gradients for Langevin, but no optimizer step touches backbone or encoder during sleep).

**From `paradigm-mixture-architecture.md` Review Addendum (8 gates):**
1. Preserve existing HLM vocabulary (`dense`, `moe`, `mod`, `mamba`; `remode` as metadata-only until tensor adapter exists).
2. Adapter factories match live constructors (unit test instantiates with `d_model=32`, runs `[2,8,32]`, returns same-shape tensor).
3. Causality non-negotiable (GNN KNN never reads future tokens; add future-leakage test).
4. Genotype grammar with budget/stability rules (first/last layer dense or linear-attn, max consecutive ODE ≤ 1, max GNN ≤ ceil(0.15 * n_layers), no GNN before layer 2, minimum one global mixer per 4 layers).
5. Random-token fitness is only smoke test; production search uses multi-axis proxy suite (short LM loss, copy/induction, long-context retrieval, entity-relation QA/code graph fixture, latency, peak memory).
6. Penalize compute explicitly (fitness = quality_score - λ_latency * latency - λ_memory * peak - λ_params * extra_params).
7. Cache and seed evaluations (local RNG, cache `tuple(pattern) -> fitness`, save generations to JSONL).
8. Novelty claim discipline (publishable claim is search-discovered heterogeneous paradigm sequences under explicit task/cost constraints; compare against dense-only, Mamba-heavy, fixed Jamba-style alternating, and HLM hand-written profiles).

**Explicit post-core controls added by v5:**

- **Hard compute cap before LPD-GA / PMA-GA**: Maximum combined LPD + PMA exploratory compute before either Gate GA is declared: 250 GPU-hours total on rented hardware. Any code or experiment consuming more than this before GA must be re-authorized at a monthly gate review.
- **Publication veto if any Ring 1 gate later fails**: If any Ring 1 gate (R1-GA/GB/GC/GD) is later discovered to have been invalid (e.g., oracle results in tables, adapter not actually smoke-tested on ≤150M, reproducibility pack fails), all downstream LPD/PMA paper submissions are automatically vetoed until the Ring 1 gate is re-passed with clean evidence.
- **Cross-extension rule**: No results that combine LPD + PMA into one system may be published until both individual GA gates are green plus one joint ablation vs. pure AMC + pure DreamBank baseline is also green. This means: LPD-GA ✓ AND PMA-GA ✓ AND (LPD+PMA joint vs AMC-only ✓) AND (LPD+PMA joint vs DreamBank-only ✓) before any cross-extension claim.
- **Explicit tie-back**: Every LPD/PMA paper claim must cite both the Review Addendum gates above and the exact Ring 1 AMC/DreamBank baseline results as prerequisite evidence.

**APEX** remains Ring 2+ grand trainer. No APEX work begins until Ring 1 green + AMC paper table lock + Gate LPD-GA where relevant. APEX consumes signed/audited AMC evidence; it cannot reshape AMC paper claims retroactively.

---

## PART X — UPDATED RISK REGISTER (Unified 6-Column Schema)

All entries: Likelihood (Low/Medium/High), Impact (Low/Medium/High), Mitigation, Trigger Date, Owner.

| ID | Risk | Likelihood | Impact | Mitigation | Trigger Date | Owner |
|---|---|---|---|---|---|---|
| R1 | Trained 1B checkpoint fails to converge on AMC memory curriculum | Medium | High | Dual-launcher strategy (TST + standard); fall back to standard curriculum if TST not ready in 30 days | Week 2 | Lead Researcher |
| R2 | Surprise gate never learns meaningful write policy | Medium | High | Ablation with fixed-threshold baseline; if surprise gate ≤ fixed-threshold at L4, simplify to learned threshold | Week 3 | ML Engineer |
| R3 | SDB propose-verify-commit contract adds unacceptable latency | Low | Medium | Async verification pipeline; measure p95 latency before declaring gate | Week 3 | Systems Engineer |
| R4 | Tier-3 constitutional retrieval degrades generation quality | Low | Medium | Ablation with Tier-3 disabled; only enable if Tier-2 alone is insufficient | Week 4 | ML Engineer |
| R5 | DreamBank margin filtering rejects all candidates on real traces | Medium | High | Sweep margin hyperparameters; fallback to temperature-based filtering if margin fails | Week 4 | Alignment Engineer |
| R6 | Zero-weight mutation invariant broken by hidden gradient paths | Low | High | Automated audit of all sleep-cycle code paths for `requires_grad=True` or optimizer steps; `torch.no_grad()` + `eval()` enforced at entry | Week 4 | ML Engineer |
| R7 | Agent traces never reach 4–12 step realistic complexity | Medium | Medium | Start with 2–4 step traces and scale; HotpotQA + tool-use as separate tracks | Week 3 | ML Engineer |
| R8 | MCTS value head never learns useful planning signal | Medium | High | Ablation: MCTS vs greedy + memory; if no delta, simplify to learned policy | Week 4 | ML Engineer |
| R9 | Tool cross-attention fails to select correct tools | Medium | Medium | Start with 3–5 tools; expand only after select accuracy ≥80% | Week 4 | ML Engineer |
| R10 | Reproducibility pack cannot replay in ≤4 hours | Medium | High | Weekly rehearsal on fresh H100; optimize JSONL artifact size; document exact driver version | Week 5 | DevOps |
| R11 | Benchmark adapter (lm-eval) never validates on ≤150M model | Low | Medium | Smoke on 150M checkpoint before 1B claims; block C9 until green | Week 2 | ML Engineer |
| R12 | C9 standard benchmarks show no AMC lift (or degrade) | Medium | High | If no lift on GSM8K/MMLU, pivot to agentic tasks as primary benchmark; C9 becomes supporting | Week 4 | Lead Researcher |
| R13 | Compute budget exhausted before R1-GA/GB/GC | Medium | High | Weekly cost review; DeepSeek-first reasoning; defer non-essential evals | Ongoing | Lead Researcher |
| R14 | Dirty-tree merge conflicts block Ring 1 branch creation | Low | Medium | Resolve dirty tree before Ring 1 branch; categorize each file as chore/feature | Week 1 | DevOps |
| R15 | DeepSeek distribution shift invalidates bootstrapped traces | Medium | Medium | Maintain ≥2 alternative trace sources; pure from-scratch ablation required before final claims | Week 3 | ML Engineer |
| R16 | Liger-Kernel install fails on current CUDA/Driver stack | Low | Medium | Fall back to manual Triton kernels; measure overhead; document exact driver/CUDA versions | Week 1 | Systems Engineer |
| R17 | Clean branch rebase onto main introduces conflicts | Low | Low | Main is trivially behind; resolve conflicts immediately after rebase | Week 1 | DevOps |
| R18 | Tier-2 memory writes cause silent data corruption under load | Low | High | Stress test with 10K+ concurrent writes; verify SDB contract holds | Week 3 | ML Engineer |
| R19 | CLAIMS_LEDGER.md claims drift from actual code | Medium | Medium | Weekly audit of claims ledger vs `git diff`; lock claims at gate declarations | Ongoing | Lead Researcher |
| R20 | Tier-3 Atlas expansion attempted prematurely | Low | High | Enforce CORE_SURFACE.md hold boundary; code review blocks Tier-3 imports until R1-GA | Ongoing | Lead Researcher |
| R21 | Brain/agentic code path diverges from AMC thesis (oracle-only drift) | Medium | High | Every agent trace must pass through AMC memory read/write; no oracle-only shortcuts in paper claims | Ongoing | ML Engineer |
| R22 | Paper submission attempt before R1-GD (reproducibility pack) | Medium | High | Submission checklist includes R1-GD evidence; auto-block if pack missing | Before arXiv | Lead Researcher |
| R23 | v4/v5 plan authority conflict causes executor confusion | Low | Medium | This v5 document is sole authority; all prior plans are archived | Immediate | Lead Researcher |
| **R24** | **Ring 1 closed agent + DreamBank loop never materializes** | **Medium** | **High** | **Monthly kill review (30/60/90 days); if L4+ evidence not emerging on trace/lift, pivot to "AMC paper only" per Kill Criterion** | **90 days** | **Lead Researcher** |
| **R25** | **DeepSeek bootstrap distribution shift makes traces non-representative** | **Medium** | **Medium** | **Pure from-scratch ablation required before any final claim; maintain ≥2 trace sources** | **Week 3** | **ML Engineer** |
| **R26** | **Tool velocity creates illusion of progress without evidence** | **High** | **High** | **Every 3-day tranche must end in green test + JSONL evidence artifact; tool velocity without artifact = failure** | **Ongoing** | **Lead Researcher** |

---

## PART XI — THE IMMEDIATE ACTION LIST (Ring 1 Maximal)

**Priority 1 — TODAY**
1. Read this document. Confirm Ring 1 definition is accepted as non-negotiable.
2. Verify HEAD is `46ee2f13` on `clean/amc-curation-20260521-101220`. Do not work from `main`.
3. Run `git status --short` and categorize each dirty file as P0/P1/Medium/Low per PART IV. Do not create new branches until dirty tree is resolved.

**Priority 2 — THIS WEEK**
1. Install Liger-Kernel, triton, lm-eval-harness. Half-day install sprint.
2. Run 150M model smoke test through engine path (Rule 3 validation).
3. Create `Ring-1` branch from clean HEAD. Freeze truth surface.
4. Draft 30-day execution tracker (Week 1–4 plan from PART VIII).
5. Weekly cost review: confirm DeepSeek + local iteration budget is tracking.

**Priority 3 — NEXT 30 DAYS**
1. Train first 1B Ring 1 checkpoint (T34-family variant).
2. Run ablation JSONL (`scripts/run_ablation.py --mode engine`).
3. Run C9 benchmarks (lm-eval adapter smoke first, then 1B eval).
4. Wire DreamBank to 1B traces; run 10–20 sleep cycles.
5. Define and execute 4–12 step agent traces with auditable memory events.
6. Compute downstream lift on held-out split.
7. Build reproducibility pack; verify 4-hour replay.

---

## PART XII — ADRS (Decision Records)

**ADR-13 (2026-05-29): Ring 1 is defined as the smallest closed AMC thesis-proving loop**
- Context: Prior plans (v3, v4) described a 12-month roadmap to 4 papers and a full cognitive platform.
- Decision: v5 defines Ring 1 as the minimal artifact proving the AMC thesis: a 1B checkpoint with per-layer surprise-gated memory, DreamBank sleep refinement, and an end-to-end agent loop with ablations.
- Consequences: All scope decisions are subordinate to this definition. LPD, PMA, APEX, Tier-3 Atlas, full serving, and Brain Phases E–H are deferred. Any tranche that cannot trace directly to R1-GA/GB/GC is out of scope for v1.
- Status: Accepted.

**ADR-14 (2026-05-29): Validation Protocol is binding for all Ring 1 claims**
- Context: Prior plans listed targets without enforcing evidence levels or reproducibility.
- Decision: Six-rule Validation & Reproducibility Protocol (§1.2) is binding for every number in any submission table, abstract, or figure. Oracle/smoke results are firewalled from paper tables (Rule 2). Reproducibility pack is mandatory pre-arXiv (Rule 6).
- Consequences: Claims ledger (CLAIMS_LEDGER.md) becomes the executable specification. All L0–L2 claims are ineligible for paper tables until upgraded to L3+ through actual training/ablation.
- Status: Accepted.

**ADR-15 (2026-05-29): AMC remains the computational substrate; UPD is downstream audit**
- Context: AMC v2 × UPD plan attempted to make UPD a co-equal memory manager.
- Decision: AMC owns memory admission, promotion, quarantine, routing, and compute allocation. UPD is downstream audit/provenance only; it records and verifies AMC decisions after the fact. UPD does not become a second memory manager.
- Consequences: All UPD work is deferred until after Gate GA. No UPD code touches active memory paths before Ring 1 green.
- Status: Accepted.

**ADR-16 (2026-05-29): Cross-extension rule for LPD + PMA**
- Context: The cross-extension plan (`preference-diffused-paradigm-mixture.md`) proposed combining LPD and PMA before either was proven independently.
- Decision: No joint LPD+PMA paper claim is allowed until both individual GA gates are green plus one joint ablation vs. pure AMC + pure DreamBank baseline is green.
- Consequences: Cross-extension experiments may run for exploratory purposes only, under the 250 GPU-hour compute cap, and may not appear in any paper until all three conditions are met.
- Status: Accepted.

---

## PART XIII — EXECUTOR GUARDRAILS (Preserved & Ring-Constrained)

1. **Read before writing.** Run `git log -1 --oneline` to confirm HEAD before any commit.
2. **Stage only named files.** Each task's "Stage only" section is exhaustive. Do not `git add -A`.
3. **TDD required for all new production code.** Write failing test first. Run it. Implement minimum passing code. Re-run focused tests. Only then consider refactoring.
4. **Frozen contracts must not change.** The `AMCModelOutput`, `HLMPreferenceBank`, and `AMCTransformer` interfaces are read-only consumers for all downstream code.
5. **DreamBank suite must remain green.** After every commit, run: `.venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py -q`
6. **Do not push.** Unless the task explicitly says "push and open PR."
7. **Do not modify dirty-tree files as part of a feature commit.** Pre-existing drift is a separate `chore:` commit.
8. **Compose, don't rewrite.** Existing alignment modules are the building blocks. Import them. Do not duplicate their logic.
9. **Ring 1 gate test for every tranche.** Before starting any task, ask: "Would this move one of R1-GA/GB/GC/GD closer, or is it deferred?" If deferred, the tranche is out of scope for v1.
10. **Validation Protocol compliance check.** Before any commit that adds/changes numbers, confirm all six rules are satisfied for those numbers.

---

## PART XIV — LEGACY APPENDIX (Historical)

This section intentionally contains no active claims. All historical context is preserved in Appendix C (crosswalk) and the v4 plan archive. Readers seeking pre-2026-05-29 details should consult:
- `/Users/christienantonio/aurelius/docs/plans/2026-05-29-aurelius-grand-unified-v4.md`
- `/Users/christienantonio/Desktop/AI Plans/` (April–May 2026 wave files)

No active Ring 1 work depends on content in this appendix.

---

## PART XV — KNOWN BLIND SPOTS (20+ Grouped Items — High-ROI Addition)

**Architectural**
1. **Surprise gate dynamics uncharacterized.** The stop-gradient surprise head (A3 in CLAIMS_LEDGER.md) is L1 design only; we have no empirical characterization of its activation distribution across layers. Severity: High. Ring Impact: R1-GA.
2. **Promotion policy brittleness at scale.** Gumbel straight-through gate (A4) is L1; temperature annealing schedule and straight-through estimator variance under deep stacks are untested. Severity: High. Ring Impact: R1-GA.
3. **SSM horizon limits vs. long agent traces.** SSM working memory may forget critical state over 12+ step traces. Severity: Medium. Ring Impact: R1-GB (trace quality).
4. **Per-layer memory write churn under high agent activity.** Tier-2 SDB (B2/B3) is L2/L3 on simple traces; no stress test with 10K+ writes per trace exists. Severity: Medium. Ring Impact: R1-GB.
5. **Interaction between Tier-2 episodic and Tier-3 constitutional memory under retrieval pressure.** No empirical study of which tier dominates when both are present. Severity: Medium. Ring Impact: R1-GA (ablation completeness).

**Empirical**
6. **No joint three-tier + DreamBank + MCTS ablation.** All existing ablations are single-surface (e.g., no-memory vs Tier-2 only). Severity: High. Ring Impact: R1-GA/GB/GC.
7. **DreamBank margin filtering parameters never swept on real agent traces.** The margin threshold is currently a fixed design constant (C2 is L3 on synthetic data only). Severity: High. Ring Impact: R1-GB.
8. **Effect size of memory benefit vs. scale unknown.** We do not know if AMC benefit is linear, log-linear, or saturating between 150M and 1B. Severity: Medium. Ring Impact: R1-GA (generalization).
9. **Reproducibility pack never stress-tested on 4–12 step traces.** Rule 6 compliance is untested on the actual Ring 1 workload. Severity: High. Ring Impact: R1-GD.
10. **C9 adapter on real checkpoints still missing.** No evidence that lm-eval-harness can consume a 1B AMC checkpoint via engine_mode (C9 is L0 in CLAIMS_LEDGER.md). Severity: High. Ring Impact: R1-GA.
11. **Training data curriculum for 1B AMC not finalized.** `configs/amc_forge_1b.yaml` and `configs/train_1b.yaml` exist, but the exact token counts and memory-curriculum mix are not validated against convergence. Severity: Medium. Ring Impact: R1-GA.
12. **No multi-seed stability report for AMC memory writes.** L4 requires 3+ seeds; we have zero. Severity: High. Ring Impact: R1-GA.

**Safety & Alignment**
13. **Memory quarantine + revocation epochs untested under DreamBank writes.** SDB quarantine exists (B2, L3) but no test covers the case where DreamBank itself writes a candidate that later triggers quarantine. Severity: High. Ring Impact: R1-GB + safety.
14. **No red-team probe on Tier-2 retrieval leakage.** Evidence B5 (provenance) is L3, but adversarial retrieval prompts that try to extract stored secrets are not in the test suite. Severity: High. Ring Impact: Safety.
15. **Constitutional memory non-evictable principle only L1.** F2 in CLAIMS_LEDGER.md is L2; we have not proved that constitutional entries survive promotion pressure from high-surprise episodic writes. Severity: Medium. Ring Impact: Safety + R1-GA.
16. **DreamBank safety verifier is deterministic stub in unit tests.** The Review Addendum requires a real safety verifier; current test fixtures use `verifier="deterministic_safety"`. Severity: Medium. Ring Impact: R1-GB.

**Scalability & Systems**
17. **Paged LTS memory CPU fallback latency untested.** `src/memory/async_memory.py` has CPU fallback for evicted entries; no latency benchmark exists under p95 pressure from agent traces. Severity: Low. Ring Impact: R1-GC (trace latency).
18. **Async consolidation pipeline thread-pool saturation untested.** Background graph consolidation (B2) may stall under concurrent agent writes. Severity: Low. Ring Impact: R1-GB.
19. **NUMA-aware memory placement not validated on multi-GPU.** `unified_manager.py` distributes LTS across NUMA nodes; single-H100 Ring 1 window makes this moot, but it resurfaces at 1.5B. Severity: Low. Ring Impact: Ring 1.5.

**Epistemic / Measurement**
20. **Benchmark saturation risk for C9.** GSM8K/MMLU may be saturated for 1B models; the AMC signal may be invisible on these benchmarks. Severity: Medium. Ring Impact: R1-GA.
21. **Framework variance in agent task evaluation.** SNARE-style framework variance (56% safety variance from framework) likely applies to agentic tasks; we have no controlled eval harness for multi-step tool use. Severity: Medium. Ring Impact: R1-GC.
22. **Runtime collapse under serial workflows.** RAMP-style collapse (100%→20% over serial workflows) may hit DreamBank sleep cycles if run serially on trace batches. Severity: Medium. Ring Impact: R1-GB.

**Tooling & Process**
23. **No living TECHNICAL_DEBT.md.** Dirty-tree categorization in PART IV is a snapshot; without a maintained file, debt re-accumulates silently. Severity: Medium. Ring Impact: All gates.
24. **Claims ledger not auto-updated from JSONL artifacts.** Manual update invites drift; automation is post-Ring 1 but a lightweight script is feasible now. Severity: Low. Ring Impact: R1-GD.
25. **No weekly rehearsal of reproducibility pack.** Rule 6 is a point-in-time claim; without weekly rehearsal, the pack rots as dependencies drift. Severity: Medium. Ring Impact: R1-GD.

---

## APPENDIX A — REPRODUCIBILITY & PUBLICATION CONTRACT

Every paper claim must satisfy the six-rule Validation & Reproducibility Protocol (§1.2). Reproducibility packs are mandatory before arXiv or conference submission.

- Rule 1 (Full invocation record): Every table row must have a corresponding JSONL artifact with `config_path`, `launch_command`, `seeds`, `checkpoint_sha`, and `artifact_path`.
- Rule 2 (Oracle/smoke firewall): No oracle or smoke result may populate a paper table. Paper tables contain only L3+ results from trained checkpoints.
- Rule 3 (Adapter validation): Before any benchmark claim, a ≤150M model must smoke through the exact same engine path.
- Rule 4 (Ablation symmetry): Every positive result is reported alongside failure modes at the same granularity.
- Rule 5 (Seed + compute): All seeds and total A100-equivalent GPU-hours are declared per major result.
- Rule 6 (Reproducibility pack): A minimal tarball/Git tag allows replay of key numbers on one GPU in ≤4 hours. This is verified weekly during Ring 1.

Failure to satisfy any rule for a given claim blocks that claim from submission until remediated.

---

## APPENDIX B — COMPUTE & GPU-HOUR MODEL FOR RING 1

**Budget philosophy:** Spend on training and ablations, not on infrastructure polish. Use DeepSeek API aggressively for synthetic traces and reasoning to avoid rented GPU costs.

**Envelope for the decisive 30–60 day window**

| Category | Estimate | Basis |
|---|---|---|
| DeepSeek API (reasoning traces, synthetic demos, safety verifiers) | $50–150/mo | Heavy usage in Weeks 1–4 |
| Local compute (MacBook + any existing GPUs) | $0 | Already owned |
| Rented H100/A100 burst 1: T34 training (1B checkpoint, 200B tokens) | 100–200 GPU-hours | Based on 1.3B existing training run patterns; 200B tokens at ~1.5 days on 8×H100 |
| Rented H100/A100 burst 2: Ablations + CB-07 serving harness | 50–150 GPU-hours | 10–20 ablation runs + serving latency measurement |
| **Total decisive compute** | **150–350 GPU-hours** | Conservative |
| **Waste factor (failed runs, debugging, replay)** | **30–50%** | Standard for research training runs |
| **Total envelope** | **200–400 GPU-hours** | With waste |
| **Cost estimate (rented H100 at ~$4–8/hr)** | **$800–1,800** | Market rates 2026 |

**1.5B upsize path** is Ring 1.5 only, after Ring 1 green. Upper bound: exact weight-mapping deltas from 1B config, plus 50% additional training compute. Not budgeted in Ring 1 envelope.

---

## APPENDIX C — SOURCE PLAN & RESEARCH REPORT CROSSWALK

| Item | Date | Core Thesis | Evidence Level | Ring Status | Notes for v1 |
|---|---|---|---|---|---|
| `CORE_SURFACE.md` | 2026-05-29 | Canonical stabilization boundary; AMC-first, Tier-2 evidence required before Tier-3 | L3 (doc + code contracts) | Ring 0 | Defines release-critical surfaces; DreamBank and AMC memory are in-scope. |
| `MASTER-IMPLEMENTATION-PLAN.md` | 2026-05-28 | Single source of truth for roadmap; Track A (AMC), Track B (Alignment), Track C (Serving) | L3 (doc, branch state verified) | Ring 0 | Confirms HEAD, dirty-tree state, and merge order. Ring 1 follows Track A. |
| `MODEL_CARD.md` | 2026-05-29 | Model description, AMC architecture, agent loop, 21 memory optimizations | L3 (doc + code refs) | Ring 0 | Defines agent surface (Observe/Think/Act/Reflect) and memory tiers targeted by Ring 1. |
| `CLAIMS_LEDGER.md` | 2026-05-30 | Evidence inventory for 32 claims across 6 categories | L3 (doc) | Ring 0 | 0/32 claims at L4+; defines the exact gap Ring 1 must close. |
| `2026-04-06-aurelius-v1-design.md` | 2026-04-06 | Four-layer stack design, 150M–7B family | L1 | Ring 0 | Target family spec. |
| `2026-04-07-heavens-gate-handoff.md` | 2026-04-07 | Rust-Python bridge handoff | L1/L2 | Ring 0 | Page-table contract for memory writes. |
| `2026-04-20-harvest-implementation.md` | 2026-04-20 | Harvest cycle pipeline | L2/L3 | Ring 0 | Upstream of DreamBank episodes. |
| `2026-04-21-aurelius-canonical-interface-contract.md` | 2026-04-21 | API surface + JSON schema | L3 | Ring 0 | Trace artifact schema for Rule 1. |
| `2026-05-09-aurelius-alignment-design.md` | 2026-05-09 | Alignment design (pre-AMC-first) | L1/L2 | Post-Ring 1 | APEX taxonomy; PART IX only. |
| `2026-05-20-apex-design.md` | 2026-05-20 | APEX three-tier design | L1 | Ring 2+ | Quarantined. |
| `2026-05-28-latent-preference-diffusion.md` | 2026-05-28 | LPD implementation plan + 10 Review Addendum gates | L0/L1 | Ring 2+ | 10 gates become binding pre-GA. |
| `2026-05-28-paradigm-mixture-architecture.md` | 2026-05-28 | PMA implementation plan + 8 Review Addendum gates | L0/L1 | Ring 2+ | 8 gates become binding pre-GA. |
| `2026-05-28-preference-diffused-paradigm-mixture.md` | 2026-05-28 | Cross-extension LPD+PMA plan | L0 | Ring 2+ | Joint claims blocked until 3 conditions. |
| `2026-05-29-aurelius-grand-unified-v4.md` | 2026-05-29 | Prior master plan; R1–R23 risk register | L2/L3 | Historical | R1–R23 carried forward to PART X. |
| `~/Desktop/AI:ML Research/` (high-value reports) | Various | Specific paper deep-dives and syntheses | L2–L4 (varies) | Mixed | Indexed in research loop; highest-value reports inform Paper 1–4 novelty matrix. |

---

## APPENDIX D — FINAL DISCLAIMER & AUTHORITY

This v5 document is now the single source of truth for all Aurelius Ring 1 work. Ring 1 is the only gravity. Any tranche, plan, or paper claim that cannot trace directly to R1-GA/GB/GC/GD or the Validation Protocol is out of scope for v1.

All prior plans (v3, v4, April–May wave files) are archived for historical context. They do not override v5. Where conflicts appear, v5 wins by definition.

---

**End of v5 Plan (Ring-1 Authoritative as of 2026-05-29)**

**Next concrete step requested:**
1. Confirm acceptance of Ring 1 definition and Validation Protocol.
2. Approve 30-day execution plan and compute budget.
3. Begin Week 1, Day 1: P0 install sprint (Liger-Kernel, triton, lm-eval-harness) + dirty-tree categorization.
