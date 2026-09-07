# AURELIUS MASTER PLAN v4 — HISTORICAL-SWEEP-AUGMENTED
## Generated 2026-06-02 — Updated from v3 + 6-month historical arXiv sweep (2026-H1)
## 32 new deep-dived papers, 8 new novel contribution opportunities (N6-N13), 10 new blind spots (B23-B32)

### v3 → v4 Changes
- **8 new novel contributions** added (N6 TRUSTTIER, N7 JUDGE-AUDIT, N8 MARLSAFE-CMDP, N9 PROCESS-VERIFY, N10 PRIVACY-KERNEL, N11 AURELIUS-EVAL, N12 KV-LIBRARY, N13 SLINGSHOT-DEFEND)
- **10 new blind spots** added (B23-B32)
- **New process-supervision pipeline** synthesized from 6 papers: CurioSFT → TreePS-RAG → HISR → VeRPO → Rubric-GRM
- **New multi-agent evaluation recipe** synthesized from 4 papers: STAT + ARMS + Cattle Trade + HC-MAPPO-L
- **KV-cache and hierarchical memory unified** as a single design space
- **Pillars explicitly extended** to 0-6
- **Source corpus:** 45,538 arXiv papers Jan-Jun 2026 (cs.CL/cs.LG/cs.AI/cs.MA/cs.IR/cs.CR/cs.CV/stat.ML)

---

# AURELIUS MASTER PLAN v3 — COMPREHENSIVE
## Generated 2026-05-29 — Consolidated from 24 sweeps, 72+ deep-dives, ~1.3MB analysis

### Context
Architecture: GQA 32/7, MoE top-2, SwiGLU, Pre-RMSNorm, pair-dim RoPE, MLA
Memory: AMC (activation consolidation)
Training: SimPO + spec decoding

---

## PHASE 0: WEEK 1 — Infrastructure + Quick Wins

| # | Action | Effort | Papers | Status |
|---|--------|--------|--------|--------|
| 0.1 | **Install Liger-Kernel** (RMSNorm 3-8x, CE 2.4x, FusedMoE 10-63x) — ⚠️ MLA gap: no kernel exists | 1 hr | — | ❌ NOT INSTALLED |
| 0.2 | **Install lm-eval-harness** — standardized evals | 1 hr | — | ❌ NOT INSTALLED |
| 0.3 | **Install triton** (dep for Liger-Kernel) | 30 min | — | ❌ NOT INSTALLED |
| 0.4 | **AdaSimPO** — per-pair adaptive coefficients | 1 day | ✓ | DONE |
| 0.5 | **Symmetry MoE Optimizer** — row-norm router updates | 2 days | ✓ | DONE |
| 0.6 | **Install Langfuse** — LLM observability | 1 day | — | ❌ NOT INSTALLED (demote to P1) |
| 0.7 | **Daily Research Loop cron fix** — dict-sort bug patched | 5 min | — | ✅ FIXED |
| 0.8 | **Qwen Guard 4B** — deploy as content safety guardrail | 1 day | 2605.28830 | P1 |

**Total Phase 0: ~3 days** — zero-retraining infra improvements.

---

## PHASE 1: WEEKS 2-4 — Inference + Safety (No Retraining)

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 1.1 | **EAGer entropy-aware decoding** | 2510.11170 | 5 days |
| 1.2 | **NestedKV cache compression** | 2605.26678 | 1 week |
| 1.3 | **Ellipsoid Control safety guard** | 2605.24552 | 1 week |
| 1.4 | **RotMoLE in MoE-LoRA fine-tuning** | 2605.25565 | 3 days |
| 1.5 | **BPPO: 6x GRPO speedup** (compatible with Pair-GRPO) | 2605.28028 | 3 days |
| 1.6 | **PackKV INT8 quantization** for KV cache | 2512.24449 | 1 week |
| 1.7 | **DualKV** — shared-prompt KV for RL training (1.6-3.8x) | 2605.15422 | 2 weeks |
| 1.8 | **MTP-3** — multi-token prediction (from MiMo V2, Step 3.5 Flash) | 2601.02780 | 1 week |

**Key insight from Step 3.5 Flash + MiMo V2**: MTP with lightweight SWA + dense heads (0.41% param overhead) enables aggressive speculative decoding. Combine with DualKV for 3-5x training speedup.

---

## PHASE 2: MONTHS 2-3 — Architecture + Training Upgrades

### Architecture Changes

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 2.1 | **P-RoPE for infinite context** | 2605.27980 | 2-4 weeks |
| 2.2 | **Hybrid SWA+GA attention** (3:1 SWA:Full, from MiMo V2) with **head-wise gated attention** (beats sink tokens +1.97) | 2601.02780 + 2602.10604 | 2 weeks |
| 2.3 | **UNIQUE sparse attention** | 2605.27740 | 1-2 weeks |
| 2.4 | **MoA activation mixing in FFN** | 2605.26647 | 1-2 weeks |
| 2.5 | **Meta-Attention — Bayesian per-token routing** (fills Oryx gap) | 2605.28384 | 1 week |
| 2.6 | **Attention Sink → Native MoE** — head collapse fix (+0.8-1.4 avg) | 2602.01203 | 3 days |
| 2.7 | **GARFA** — GQA-specific RoPE frequency adaptation | 2604.07766 | 1 week |
| 2.8 | **EP-Group Balanced MoE Routing** — from Step 3.5 Flash | 2602.10604 | 1 week |

**Key architecture stack recommendation**:
```
Hybrid SWA+GA (3:1) + GQA-8 (aligned with GQA 32/7)
  → Head-wise gated attention (replaces sink tokens)
  → Meta-Attention (per-token Bayesian routing between strategies)
  → EP load-balanced MoE top-k routing
  → GARFA (adaptive RoPE frequencies per head group)
```
This stack adapts MiMo V2/Step 3.5 Flash hybrid attention patterns to Aurelius scale.

### Training Pipeline

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 2.9 | **G2D+SimPO alignment pipeline** | 2605.21266 | 2-3 weeks |
| 2.10 | **PEAM-style AMC scoring/trigger** (ADOPT concepts only; REJECT LoRA trajectory internalization) | 2605.27762 | 1 week |
| 2.11 | **Tensor Memory — fixed-size recurrent state** | 2605.27686 | 2-3 weeks |
| 2.12 | **MGRetrieval — pyramid retrieval** for AMC | 2605.27437 | 1 week |
| 2.13 | **MIS-PO** (Metropolis Independence Sampling PO) — binary masking replaces importance sampling for RL stability | 2602.10604 (Step 3.5 Flash) | 2-3 weeks |
| 2.14 | **MOPD** (Multi-Teacher On-Policy Distillation) — from MiMo V2 | 2601.02780 | 2-3 weeks |
| 2.15 | **Self-Verified Distillation pipeline** | 2605.26132 | 2-3 weeks |
| 2.16 | **Routing-Aligned MoE Fine-Tuning** (+1-6 pp on non-English) | 2605.28306 | 1 week |
| 2.17 | **MIRA data selection** — 50% fewer tokens, full performance | 2605.30288 | 1 week |

---

## PHASE 3: MONTHS 3-6 — Advanced Alignment + Reasoning

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 3.1 | **AXPO + BPPO + Pair-GRPO combined alignment pipeline** | AXPO 2605.28774 + BPPO 2605.28028 + Pair-GRPO 2605.06375 | 4-6 weeks |
| 3.2 | **GPRL + Pair-GRPO hybrid** | 2605.18721 + 2605.06375 | 3-4 weeks |
| 3.3 | **DeepTool — process-supervised RL for tool use** (fills Thinking-Acting Gap) | 2605.29568 | 2-3 weeks |
| 3.4 | **Counterfactual Credit Assignment** — fixes GRPO penalizing correct intermediates | 2605.16302 | 1-2 weeks |
| 3.5 | **Verifiable Process Rewards** — token-level reward for agentic training | 2605.10325 | 2-3 weeks |
| 3.6 | **AXPO-style tool-call resampling** — +25% resampling > +100% rollout | 2605.28774 | 1 week |
| 3.7 | **PaCoRe test-time scaling** — parallel reasoning trajectories | 2602.10604 | 2 weeks |
| 3.8 | **HyperTree Planning** — hierarchical divide-and-conquer reasoning (3.6x vs o1-preview) | 2505.02322 | 2-3 weeks |
| 3.9 | **TEMPO — test-time training** with calibration (prevents reward drift) | 2604.19295 | 3-4 weeks |
| 3.10 | **SERE-Bench** — small-model reasoning efficiency benchmark (NOVEL) | Novel | 3-4 weeks |

### Combined Alignment Stack (recommended)
```
Token-level:  Verifiable Process Rewards + Counterfactual Credit Assignment
Sampling:     AXPO tool-call resampling (fixes Thinking-Acting Gap)
Selection:    BPPO binary completion selection (6x speedup)
Optimization: Pair-GRPO with MIS-PO masking (convergence + stability)
Distillation: MOPD multi-teacher on-policy (consolidation)
```

---

## PHASE 4: MONTHS 6-12 — Memory + Research Frontier

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 4.1 | **MemCog navigable memory** — Memory-as-Cognition paradigm | 2605.28046 | 4-6 weeks |
| 4.2 | **MemGuard memory contamination prevention** | 2605.28009 | 1-2 weeks |
| 4.3 | **HGMEM hypergraph working memory** — upgrades AMC T1 | 2512.23959 | 3-4 weeks |
| 4.4 | **RAT+ exponential memory** — recurrence-augmented attention | 2605.28640 | 4-6 weeks |
| 4.5 | **Bilevel AutoResearch meta-loop** | 2603.23420 | 3-4 weeks |
| 4.6 | **CIRF functional tokens** | 2605.28292 | 2-3 weeks |
| 4.7 | **Alignment tampering defense (ZO-SimPO)** — NOVEL | 2605.29396 + 2605.27355 | 2-3 weeks |
| 4.8 | **MemGuardian** — DP for parametric memory (NOVEL) | 2605.27825 | 8-12 weeks |
| 4.9 | **WorldMemArena-AMC** — parametric memory eval (NOVEL) | 2605.29341 | 4-6 weeks |
| 4.10 | **MemTrace — AMC error diagnosis** | 2605.28732 | 2-3 weeks |

### Complete Memory Stack (Recommended Architecture)
```
Layer 0: AMC Tier-1 — SSM working memory (upgraded to HGMEM hypergraph)
Layer 1: AMC Tier-2→Tier-3 — PEAM scoring/trigger consolidation
Layer 2: MemGuard — contamination prevention filter
Layer 3: MGRetrieval — pyramid-structured retrieval
Layer 4: MemCog — proactive navigation + Cognitive protocol
Layer 5: FluxMem — evolution (future)
     + MemTrace — periodic diagnostics
     + WorldMemArena-AMC — evaluation
```

---

## PHASE 5: NOVEL CONTRIBUTION OPPORTUNITIES (ICLR/NeurIPS 2027)

| # | Contribution | Base Papers | Effort | Type |
|---|-------------|-------------|--------|------|
| **N1** | **ZO-SimPO: Alignment Tampering Defense** — zeroth-order optimization smoothes SimPO loss landscape, prevents preference data manipulation | 2605.29396 + 2605.27355 + 2605.25189 | 2-3 weeks | Method paper |
| **N2** | **MemEval-AMC: First Parametric Memory Consolidation Benchmark** — 12 capability axes, 4-stage lifecycle, 18-error taxonomy | WorldMemArena 2605.29341 + MemGuard 2605.28009 + AMA-Bench | 4-6 weeks | Benchmark paper |
| **N3** | **AXPO+BPPO+Pair-GRPO: Combined Agentic Alignment** — sampling + selection + optimization in one pipeline | 2605.28774 + 2605.28028 + 2605.06375 | 4-6 weeks | Method paper |
| **N4** | **SERE-Bench: Small-Model Reasoning Efficiency Benchmark** — first benchmark for <7B model reasoning efficiency | HRBench 2605.28398 | 3-4 weeks | Benchmark paper |
| **N5** | **MemGuardian: DP for Parametric Agent Memory** — first privacy defense for activation-based memory | MRMMIA 2605.27825 | 8-12 weeks | Method paper |

**Priority recommendation**: N2 (MemEval-AMC) or N4 (SERE-Bench) first — lowest effort, benchmark papers have clear acceptance path.

---

## KNOWN BLIND SPOTS (22 Total)

### Safety & Security
| Blind Spot | Paper | Impact |
|-----------|-------|--------|
| Alignment Tampering | 2605.27355 | All alignment methods assume exogenous preferences |
| RLHF Artifacts Survive | 2605.28102 | 5 artifacts persist after prompt replacement |
| Memory Privacy Leakage | MRMMIA 2605.27825 | AUC 0.99-1.00 on membership inference, no defense |
| SkillSurface Attacks | SkillSafetyBench 2605.12015 | 15-50% ASR on all CLI agents |
| Eval Meta-Knowledge Gaming | 2605.28591 | Structural leakage inflates safety scores |
| Web Retrieval Degrades Safety | 2605.29224 | AMC retrieval = same vulnerability |
| No Refusal Experts in MoE | 2605.29708 | Safety in representations, not routing |

### Training Flaws
| Blind Spot | Paper | Impact |
|-----------|-------|--------|
| GRPO Under-Trains Tool Use | AXPO 2605.28774 | 30% tool-use rate |
| GRPO Penalizes Correct Steps | CalibAdv 2604.18235 | Credit assignment fundamentally flawed |
| GRPO Causes Modal Collapse | MAPO 2605.27741 | Modal abandonment |
| Agent RL Unique Dynamics | CyclEnt 2605.27954 | Entropy cycles, not collapse |
| RLVR Doesn't Generalize | Multiple | Only math/code |
| Alignment Fragility | 2605.18309 | Alignment degrades under further FT |
| No Memory Eval Standard | AgingBench + TriMem | No parametric consolidation eval |

### Evaluation
| Blind Spot | Paper | Impact |
|-----------|-------|--------|
| Benchmark Saturation | TASTE 2605.28556 | 35-60pt drops on harder tasks |
| Framework > Model | SNARE 2605.28122 | 56% safety variance from framework |
| Runtime Collapse | RAMP 2605.27492 | 100%→20% over serial workflows |
| Reasoning ≠ Tool-Use Training | AXPO 2605.28774 | G2D underperforms for agents |
| LoRA Subspaces Near-Orthogonal | 2605.28896 | Base SAEs blind to adapter features |
| Prompt Optimization May Be Harmful | Coin Flip 2604.14585 | 49% of runs below zero-shot |
| LLMs Can't Do Causal Discovery | Kernel Obstruction 2605.27567 | Formal proof |

---

## INFRASTRUCTURE TRACK

| Tool | Priority | Status | Purpose |
|------|----------|--------|---------|
| **Liger-Kernel** | P0 | ❌ NOT INSTALLED (MLA gap) | Fused Triton ops: RMSNorm 3-8x, CE 2.4x, FusedMoE 10-63x |
| **lm-eval-harness** | P0 | ❌ NOT INSTALLED | Standardized evals |
| **triton** | P0 | ❌ NOT INSTALLED | Dep for Liger-Kernel |
| **Langfuse** | P1 | ❌ NOT INSTALLED | Observability |
| **Unsloth** | P1 | ❌ NOT INSTALLED | 2x fine-tuning speed |
| **Qwen Guard 4B** | P1 | ❌ NOT INSTALLED | Content safety guardrail |
| **SGLang** | P2 | ❌ NOT INSTALLED | Prefix caching |
| **dstack** | P2 | ❌ NOT INSTALLED | GPU orchestration |
| **OpenRLHF** | P3 | ❌ NOT INSTALLED | Multi-turn RLHF |

---

## RESEARCH LOOP OPERATIONS

### Active Cron Jobs
| Name | Schedule | Status |
|------|----------|--------|
| deep_research_runner | every 4h | ✅ Healthy |
| AI Skill Deep Research | every 360m | ✅ Healthy |
| Daily Research Loop | 0 9 * * * | ✅ Fixed (dict-sort bug) |
| Small-Device Architecture | every 180m | ⚠️ Intermittent |
| track-step37-flash | every 360m | 🆕 Created May 28 |

### Upgrade Path
| Stage | Status | Next Step |
|-------|--------|-----------|
| T1: Keyword sweep + deep-dives | ✅ | Structured JSON trial traces |
| T1+: Cron every 4h | ✅ | Coverage tracking matrix |
| T2: Cross-session paper DB | ❌ | EvolveMem-style database |
| T3: Meta-loop | ❌ | Bilevel optimization |
| T3+: Self-evolving prompts | ❌ | ADAS-style meta-agent |

---

## KEY ARCHITECTURE REFERENCES

| Paper | Key Finding for Aurelius |
|-------|------------------------|
| **Step 3.5 Flash** (2602.10604) | S3F1+Head hybrid, MIS-PO, expert collapse analysis, head-wise gated attention +1.97 over sink |
| **DeepSeek V3.2** (2512.02556) | DSA sparse attention, specialist distillation, GRPO with unbiased KL + off-policy masking |
| **MiMo V2 Flash** (2601.02780) | 5:1 hybrid SWA+GA with sink bias, MOPD distill, MTP 2.6x |
| **Llama 4** (2601.11659) | iRoPE, 10M context (Scout), MetaP hyperparameter optimization |
| **Meta-Attention** (2605.28384) | Bayesian per-token routing between 3 attention strategies |
| **DeepTool** (2605.29568) | Process-supervised RL for tool use — fills AXPO gap |
| **Counterfactual Paths** (2605.16302) | Reduces credit assignment variance — fixes GRPO flaw |

---

*Consolidated from: 24 research sweeps, 72+ deep-dived papers, ~1.3MB analysis, 20+ deep-dive reports, 12 batch syntheses, 5 novel contribution blueprints. Continuous research cron runs every 4 hours.*


---

# ADDENDUM v4 — From 6-Month Historical Sweep (2026-H1)

# AURELIUS HISTORICAL SWEEP 2026-H1 — MASTER SYNTHESIS
## 32 deep-dived papers, 5 research angles, 8 novel contribution opportunities

**Generated:** 2026-06-02
**Source data:**
- 6 months × 8 categories = 48 arXiv month/category fetches
- 45,538 unique papers retrieved (raw)
- 34,789 broad-keyword matches → 2,891 focused LLM-core matches
- 1,000 top-ranked gap papers vs Master Plan v3 (61 IDs) + 12 cron runs (88 IDs) = 160 covered IDs
- 32 high-signal papers deep-dived across 5 thematic groups
- 5 deep-dive reports: deepdive_G1.md (KV-cache), deepdive_G2.md (GRPO/RLHF), deepdive_G3.md (distillation/reasoning), deepdive_G4.md (multi-agent benchmarks), deepdive_G5.md (privacy/safety/quant)

---

## 1. Cross-Paper Convergent Findings (the "consensus layer" for 2026-H1)

### 1.1 Inference + KV-cache — three orthogonal trends

**(a) Asymmetric K/V treatment is now consensus.** PolyKV (2604.24971), Leyline (2606.01065), WindowQuant (2605.02262), and the older KIVI/KVQuant family all exploit the K-dynamic-range/V-dynamic-range asymmetry. *No* new method treats K and V identically. Aurelius should adopt asymmetric treatment as a default assumption in any KV-cache work.

**(b) Per-agent isolation is a first-class concern.** Four of seven G1 papers (2603.04428, 2604.24971, 2602.21477, 2603.02240) implement isolation with different security models — block pool, content-hash addressing, FSM-cluster isolation, Bayesian trust. The lack of a common security model is an *open gap* Aurelius can address.

**(c) Span-level mutation is the right interface.** Leyline (2606.01065) generalizes CacheBlend's "high-deviation span recompute" and EPIC's "chunk-boundary recompute" into a (span, replacement) directive. The community is converging on "directive verbs" (AMORTIZE, FORGET, DECAY, PIN, SCRATCH, SHARE) as the agent↔serving interface.

### 1.2 RL/RLHF — five converging failure modes with remedies

| Failure mode | Paper | Remedy |
|---|---|---|
| Reward hacking via sharpness | 2602.18037 | Explicit gradient regularization (‖∇J‖²) |
| Mode collapse / rank degeneration | 2603.16157 | Joint-rank baseline (DyJR) |
| Sparse binary reward | 2601.03525 | Cardinality-bias-corrected partial-success (VeRPO) |
| Process credit fragmentation | 2603.18683 | Segmental + hindsight PRM (HISR) |
| Adaptive adversarial bypass | 2602.02395 | Test alignment against RL-trained attackers |

**Consensus:** Group-mean baseline in GRPO is broken on hard problems; KL penalty is *optional* if gradient regularization is present; verifiable dense rewards dominate sparse binary rewards when cardinality is corrected.

### 1.3 Distillation + Process Supervision — three-way synthesis

The trio **CurioSFT (2602.02244) + TreePS-RAG (2601.06922) + Rubric-GRM (2604.16335)** is a tightly-coupled recipe for small reasoning models:
1. **Upstream:** CurioSFT's entropy-preserving SFT prevents small-LM collapse before RL.
2. **Online:** TreePS-RAG's tree-based process advantage injects step-wise credit at fixed rollout cost.
3. **Filtering:** Rubric-GRM's weighted-judge trajectory filter beats rejection sampling, *and a weaker GRM can guide a stronger generator* (generation-verification gap).

The cross-cutting theme: replace single broken signals (one-hot CE, single outcome reward, single binary test) with richer multi-faceted signals computable in the same training loop.

### 1.4 Multi-Agent Benchmarks — three evaluative axes are now mature

- **Combinatorial scaling (STAT, 2605.06557):** process-level diagnostics (conflict rate, redundant-assignment rate) reveal coordination failures hidden by return.
- **Mixed-motive strategic reasoning (Cattle Trade, 2605.14537):** spending efficiency, resource discipline, and phase-adaptive bidding predict rank more than spending volume.
- **MARL-specific reward shaping (ARMS, 2605.23562):** first Nash-equilibrium-preserving automatic shaping; identifies a MARL-specific oscillatory failure mode.

**Open evaluation gap:** Long-horizon coordination (9+ agents, 1000+ turns) and joint partial-observability × combinatorial-scaling benchmarks are unstudied — exactly the regime Aurelius targets.

### 1.5 Privacy / Safety / Quantization — the safe-MARL + privacy stack

- **HC-MAPPO-L (2603.00129)** provides the constrained-MARL + Lagrangian + hierarchical MAPPO template.
- **RLFTSim (2605.19033)** provides the composite scene-level Realism Meta-Metric with Leave-One-Out anti-goodhart reward.
- **PP-MAPF (2605.14119)** formalizes planning-level (k-Privacy) + execution-level (Runtime k-Privacy) privacy for multi-agent systems.
- **GPU INT8 KV-cache (2601.04719)** establishes a 1,694× speedup baseline that *any* quantization contribution must beat with kernel engineering, not just algorithmic innovation.

**Open nexus:** A CMDP with both a delay constraint (Lagrangian) and a privacy budget (DP-style) is the natural Aurelius safety/privacy contribution.

---

## 2. Recurring Methodological Patterns (the "meta-signals")

1. **Empirical consensus > theoretical consensus.** In G2 specifically, every GRPO/RLHF improvement comes with a >5-point empirical gain; theoretical motivation is supportive but not leading.

2. **"Failure mode" papers are now first-class.** PAL-KV (2601.08343), mRAG-leakage (2601.17644), GRPO modal collapse (MAPO), and DyJR (2603.16157) are *negative results* that reframe existing techniques as unsafe. Aurelius should include failure-mode analyses alongside positive results.

3. **Multi-agent is converging on the right unit of analysis.** Whether it's a coordination mechanism (STAT), a memory operation (Pancake), a privacy budget (PP-MAPF), or a reward signal (ARMS), the *unit of analysis* is shifting from per-agent to per-relation/per-joint-action. Aurelius's MARL contribution should adopt a relation-centric framing.

4. **Test-time scaling requires a learned selector, not just more compute.** PROMISE (2601.04674), TreePS-RAG (2601.06922), and PaCoRe (in v3) all show that brute-force beam scaling gives marginal gains; a learned PRM/PRL gives >10 points on hard problems.

5. **KV-cache and hierarchical memory are converging.** Pancake (2602.21477), SuperLocalMemory (2603.02240), and Leyline (2606.01065) are all developing interface-level abstractions for "memory" that blur the line between retrieval-time and attention-time caches. Aurelius's Pillar 2 + Pillar 4 should be developed together, not separately.

6. **Kernel engineering matters as much as algorithmic innovation.** GPU INT8 KV (2601.04719) shows 1,694× from kernel engineering alone. Any "Aurelius quantization" claim must pair a quantization scheme with a kernel-engineering story.

7. **RLHF and process supervision compose via the reward layer.** Privacy-preserving RLHF (2603.22563) shows that DP cost paid only on the linear reward head is sufficient — privatizing the policy is not required. This generalizes: any RLHF/RLVR improvement that touches the reward model alone can compose with other improvements (GR, DyJR, HISR) without paying double the training cost.

8. **Multi-agent is feasible without a critic.** Distributed zeroth-order policy gradient (2605.15697) achieves ε-stationary convergence with O(ε⁻⁴) samples — for black-box agent settings (LLM-as-agent, API-only models), the ZO estimator is the natural choice.

---

## 3. Contradictions / Tensions (where the literature disagrees)

### 3.1 Edge vs Datacenter KV
- **2603.04428 (Persistent Q4 KV on edge UMA):** Per-agent persistent caches; isolated storage; minimal sharing.
- **2604.24971 (PolyKV shared pool):** Datacenter; shared pool; multi-reader concurrent access; 97.7% memory reduction.
- **Resolution:** Hybrid edge↔pool architecture. Per-agent persistence on edge, shared pool in datacenter. The two are not contradictory but at different deployment regimes.

### 3.2 Shared cache vs Judge invariance
- **2604.24971 (PolyKV):** Shared pool is the future of multi-agent serving.
- **2601.08343 (PAL-KV):** Judge-side cache reuse is *unsafe* (Judge Consistency Rate drops to 37-43% on hard tasks).
- **Resolution:** A shared pool must come with a judge-invariance audit (Aurelius sub-contribution). PolyKV's shared pool has not been validated for judge invariance.

### 3.3 PRM vs ORM for long-horizon tasks
- **2603.18683 (HISR):** Segmental + hindsight PRM is the dominant approach for 40+ turn horizons.
- **2601.03525 (VeRPO):** Binary outcome-only is fine when the test suite is rich enough; dense rewards from per-test partial success work.
- **Resolution:** Depends on test-suite richness. When the verifier is rich (test cases, sub-tasks), VeRPO-style partial-success rewards suffice. When the verifier is sparse (final success only), HISR-style segmental PRM is required. *Both* should be in Aurelius's toolbox.

### 3.4 KL penalty vs Gradient Regularization
- **2602.18037 (Gradient Regularization):** Explicit GR (‖∇J‖²) can *replace* KL penalty entirely while allowing more aggressive updates.
- **Standard practice (Olmo, GLM-4.5, etc.):** Drop the KL penalty in RLVR.
- **Resolution:** Standard practice may be missing a stability mechanism. GR (2602.18037) is a *better* replacement than no-replacement. Aurelius should default to GR-on, KL-off.

### 3.5 Reward shaping for MARL: ARMS vs PBRS
- **ARMS (2605.23562):** Trajectory-ranking with Nash-equilibrium-preservation guarantee; identifies oscillation failure mode.
- **PBRS (Ng et al. 1999, still used):** Potential-based shaping; *does not have* an equilibrium-preservation guarantee in MARL.
- **Resolution:** ARMS dominates on the equilibrium-preservation axis; the open question is whether ARMS scales to LLM-policy-as-shaper or rubric-as-shaper regimes. Aurelius has an opening here.

---

## 4. New Contributions to Add to Master Plan v4

These are *novel* opportunities that emerged from this 6-month sweep, not yet in v3.

### N6: TRUSTTIER — Trust-Tiered Hierarchical Memory (NEW)
**Combines:** Pancake (2602.21477) + SuperLocalMemory (2603.02240)
**Method:** Pancake's 3-level cluster cache (L0/L1/L2) + FSM pattern matching, but with SuperLocalMemory's Beta(2,1) Bayesian trust score as the eviction policy: L0 caches only entries with trust > θ, eviction cascades L0→L1→L2 along decreasing trust. Adds a write-time trust update and a read-time trust-oracle.
**Why novel:** No paper combines tiered cache performance with tiered cache security. Trust-tiered eviction is a new design lever.
**Aurelius fit:** Pillar 2 (Hierarchical Memory) — directly upgrades AMC's memory tier design.
**Effort:** 3-4 weeks
**Tier:** Method paper.

### N7: JUDGE-AUDIT — Judge-Invariance Audit for Multi-Agent RL (NEW)
**Combines:** PAL-KV (2601.08343) + any RL pipeline with LLM-as-judge.
**Method:** Run PAL-KV's JCR (Judge Consistency Rate) methodology on Aurelius's RL training pipeline. Specifically: cache-vs-dense judge consistency for (a) reward models, (b) value functions, (c) policy evaluators. Report JCR as a standard metric alongside accuracy. Propose a "judge-invariance budget" ε_JI; a pipeline is "auditable" iff JCR ≥ 1-ε_JI on a held-out validation set.
**Why novel:** First systematic application of judge-invariance methodology to RL training. The 2601.08343 paper focuses on inference; extending to training is open.
**Aurelius fit:** Pillar 3 (Multi-Agent RL) — adds a "RL reward safety" sub-contribution.
**Effort:** 2-3 weeks
**Tier:** Method paper.

### N8: MARLSAFE-CMDP — Joint Safety + Privacy CMDP for Multi-Agent LLM (NEW)
**Combines:** HC-MAPPO-L (2603.00129) + PP-MAPF (2605.14119) + RLFTSim (2605.19033).
**Method:** Extend HC-MAPPO-L's CMDP formulation to have *two* constraint dimensions: (a) a delay/safety constraint (Lagrangian, as in HC-MAPPO-L) and (b) a privacy budget (ε, δ)-DP-style. The privacy constraint bounds the *information leakage* of the centralized critic's gradients to per-agent private data, computed via SSIM-based reconstruction metrics. Train on RLFTSim-style realistic multi-agent simulator; evaluate privacy leakage against a gradient-inversion attack. Report a unified utility / safety / privacy Pareto frontier.
**Why novel:** No 2026-H1 work closes the joint safety + privacy CMDP for MARL. The privacy-RL composition guarantees (DP composition across actor-critic updates) is also open.
**Aurelius fit:** Pillar 3 (Multi-Agent RL) + Pillar 4 (Safety/Privacy).
**Effort:** 6-8 weeks
**Tier:** Method paper; strong ICLR 2027 candidate.

### N9: PROCESS-VERIFY — Online Process Supervision for Tool-Calling Agents (NEW)
**Combines:** HISR (2603.18683) + TreePS-RAG (2601.06922) + Proxy State-Based Eval (2602.16246) + Rubric-GRM (2604.16335).
**Method:** A four-stage process supervision pipeline for tool-calling agents (DeepTool-style): (1) TreePS-RAG's tree-based process advantage for online step credit; (2) HISR's segmental+hindsight PRM for long-horizon reward shaping; (3) Proxy State-Based Eval for sparse-reward tool environments without deterministic backends; (4) Rubric-GRM filtering for RFT data collection when (1)-(3) are too expensive.
**Why novel:** No 2026-H1 paper integrates all four. The combination is more than the sum of its parts: TreePS provides online credit, HISR provides offline credit, Proxy-State handles no-deterministic-backend cases, Rubric-GRM handles the data selector. Different regimes use different combinations.
**Aurelius fit:** Pillar 5 (Process Supervision) + Pillar 6 (Agentic Tool Use) — directly addresses the AXPO 2605.28774 gap.
**Effort:** 4-6 weeks
**Tier:** Method paper; strong ICLR/NeurIPS 2027 candidate.

### N10: PRIVACY-KERNEL — Implicit-Privacy via Quantization Kernel Engineering (NEW)
**Combines:** GPU INT8 KV-cache (2601.04719) + DP-SGD (Abadi et al. 2016) + the open question of whether INT8 quantization rounds away memorization.
**Method:** Systematic empirical study: for a fixed LLM at fixed (ε, δ) DP-SGD budget, what is the *interaction* between INT8 KV-cache quantization and membership-inference attack success? Hypothesis: INT8 quantization rounds away ~0.4% of the activation signal, which may *also* round away memorization. Build a "privacy budget reduction" framework: INT8 quantization is a *free* privacy mechanism on top of DP-SGD.
**Why novel:** 2601.04719 establishes the kernel-engineering baseline; 2603.00129 establishes the SSIM-based privacy metric; combining them with an MIA attack study is a single-table empirical result that no one has published.
**Aurelius fit:** Pillar 4 (Quantization) + Pillar 7 (Privacy) — directly fills a gap in v3.
**Effort:** 2-3 weeks
**Tier:** Empirical-study paper; lower risk, clear acceptance path.

### N11: AURELIUS-EVAL — Process-Diagnostic Multi-Agent Benchmark (NEW)
**Combines:** STAT (2605.06557) + ARMS (2605.23562) + Cattle Trade (2605.14537) + HC-MAPPO-L (2603.00129).
**Method:** A new benchmark that exposes *both* process-level coordination diagnostics (from STAT) *and* sparse-reward shaping challenges (from ARMS) *and* mixed-motive strategic reasoning (from Cattle Trade) *and* constrained-MARL safety (from HC-MAPPO-L). Evaluates 8–10 cooperative, mixed-motive, and competitive tasks with 2–100 agents, 10–1000 turn horizons, and full state-logging for behavioral diagnosis. Reports return *plus* process-level diagnostics *plus* safety-constraint violation rate *plus* privacy leakage.
**Why novel:** No 2026-H1 benchmark stresses all four dimensions jointly. Long-horizon coordination (100+ agents, 1000+ turns) and process-level diagnostics for shaped-reward MARL are both unstudied.
**Aurelius fit:** Pillar 3 (Multi-Agent RL) + Pillar 4 (Evaluation). Direct complement to v3's N2 (MemEval-AMC) and N4 (SERE-Bench).
**Effort:** 6-8 weeks
**Tier:** Benchmark paper; high ICLR/NeurIPS 2027 acceptance probability (cf. the success of the v3 N2 and N4 templates).

### N12: KV-LIBRARY — Composable KV-Cache Directive Library (NEW)
**Combines:** Leyline (2606.01065) + PolyKV (2604.24971) + Pancake (2602.21477) + WindowQuant (2605.02262) + GPU INT8 KV (2601.04719).
**Method:** An open-source library implementing Leyline's (span, replacement) directive interface, parameterized by a *backend* (RadixCache / paged block / poly-K shared pool / Pancake hierarchical). Each backend implements a different kernel for the directive (e.g., Leyline's δ-rotation for radix, content-hash rewrite for poly-K, FSM-cascade for Pancake). Includes the WindowQuant tiered-precision policy and the GPU INT8 kernel. Users compose: "given directive (AMORTIZE, span=s, replacement=r), use backend=PolyKV with tiered-precision."
**Why novel:** No open-source library unifies the 2026 KV-cache landscape. The composability claim is testable: run the same directive across backends and report end-to-end latency / memory / PPL.
**Aurelius fit:** Pillar 1 (Distributed MoE) + Pillar 4 (KV-Cache) + Pillar 2 (Hierarchical Memory). Foundation work for all three.
**Effort:** 8-12 weeks
**Tier:** Library + methods paper; high community impact.

### N13: SLINGSHOT-DEFEND — Defense via Adaptive RL Attacker-Aware Training (NEW)
**Combines:** Slingshot (2602.02395) + Gradient Regularization (2602.18037) + HISR (2603.18683) + DyJR (2603.16157).
**Method:** Train the *defender* model with awareness that the attacker is a small RL-trained agent. Iterative best-response: in each round, (1) attacker is updated via PPO/GRPO to find worst-case prompts; (2) defender is updated via GR + DyJR + HISR to be robust to the current attacker. After convergence, evaluate against a held-out Slingshot-style attacker. Report attacker-ASR curve as a function of defense training rounds.
**Why novel:** 2602.02395 establishes the attacker; the defender-side response is open. The "iterative best-response" is essentially a *game-theoretic* alignment procedure not yet published.
**Aurelius fit:** Pillar 4 (Safety) + Pillar 5 (Process Supervision). Directly addresses the alignment-brittleness gap.
**Effort:** 3-4 weeks
**Tier:** Method paper.

---

## 5. New Blind Spots Identified (additions to v3's 22)

| # | Blind Spot | Paper | Impact | Recommended Aurelius Action |
|---|-----------|-------|--------|---|
| B23 | Judge-invariance under cache reuse | 2601.08343 | Reward-model silently flips winners | N7 JUDGE-AUDIT |
| B24 | Gradient inversion in centralized critic | 2603.00129 | Critic gradients leak per-agent private data | N8 MARLSAFE-CMDP |
| B25 | Adaptive RL attacker beats static safety | 2602.02395 | All alignment is brittle to adaptive adversaries | N13 SLINGSHOT-DEFEND |
| B26 | Cardinality bias in partial-success rewards | 2601.03525 | Easy-test majority dominates baseline → collapse on frontier | Adopt VeRPO weighting as default in all verifiable-reward setups |
| B27 | Reward-hacking via sharpness, not just KL | 2602.18037 | Standard reward hacking is *not* a KL problem | Adopt GR-on, KL-off as default post-training config |
| B28 | Privacy budget fragmentation in multi-stage RL | 2603.22563 | Privacy degrades when split across stages | Adopt decoupled-reward DP design |
| B29 | Memory poisoning in agentic RL | 2603.02240 | Attacker shapes policy via memory writes | Add trust-oracle to AMC Tier-2 (N6 TRUSTTIER) |
| B30 | Quantization × privacy interaction is unstudied | 2601.04719 | INT8 may help OR hurt MIA defense | N10 PRIVACY-KERNEL |
| B31 | Long-horizon coordination (>9 agents, >1000 turns) | 2605.06557 | Current benchmarks max at 9 agents / 100 tasks | N11 AURELIUS-EVAL |
| B32 | Process-level diagnostics for *shaped* MARL | 2605.23562 | ARMS × STAT interaction is unstudied | N11 AURELIUS-EVAL |

---

## 6. Updated Aurelius Pillars (synthesis of v3 + this sweep)

```
Pillar 0: Inference Infrastructure
  Existing (v3): Liger-Kernel, EAGer, NestedKV, BPPO, DualKV, MTP-3
  New (sweep): KV-LIBRARY (N12), GPU INT8 KV baseline (1,694×), trust-tiered cache (N6)

Pillar 1: Architecture (MoE + Attention)
  Existing (v3): GQA, MLA, P-RoPE, Hybrid SWA+GA, GARFA, EP-Group Balanced
  New (sweep): PolyKV shared pool as expert storage, Pancake as memory substrate, KV-LIBRARY

Pillar 2: Hierarchical Memory
  Existing (v3): AMC T1-T3, HGMEM, MemCog, MemGuardian, MGRetrieval
  New (sweep): Pancake (4.29× end-to-end SOTA), TRUSTTIER (N6), SuperLocalMemory (Bayesian trust)

Pillar 3: Multi-Agent RL
  Existing (v3): STAT, AXPO/BPPO/Pair-GRPO combined, HINT, ARMS
  New (sweep): MARLSAFE-CMDP (N8), Distributed ZO gradient (2605.15697), Slingshot defender (N13)

Pillar 4: Safety / Privacy / Quantization
  Existing (v3): ZO-SimPO, Ellipsoid Control, MemGuardian, Qwen Guard 4B
  New (sweep): HC-MAPPO-L (CMDP+Lagrangian), PRIVACY-KERNEL (N10), JUDGE-AUDIT (N7), PP-MAPF framework

Pillar 5: Process Supervision / Reasoning
  Existing (v3): Verifiable Process Rewards, Counterfactual Credit, DeepTool, SERE-Bench
  New (sweep): CurioSFT (entropy-preserving SFT), HISR (segmental+hindsight PRM), VeRPO (cardinality-bias correction), DyJR (joint-rank baseline), TreePS-RAG (online process advantage), Rubric-GRM (trajectory filter), PROCESS-VERIFY (N9)

Pillar 6: Benchmarks
  Existing (v3): SERE-Bench, WorldMemArena-AMC, MemEval-AMC
  New (sweep): AURELIUS-EVAL (N11) — joint process-diagnostics + sparse-reward + mixed-motive + safety
```

---

## 7. Novel Contribution Priority Ranking (combined v3 N1-N5 + sweep N6-N13)

| # | Title | Effort | Acceptance Path | Recommendation |
|---|-------|--------|-----------------|----------------|
| **N2** | MemEval-AMC | 4-6 wk | Benchmark | Start |
| **N4** | SERE-Bench | 3-4 wk | Benchmark | Start |
| **N6** | TRUSTTIER | 3-4 wk | Method | Start (builds on Pancake, easy win) |
| **N10** | PRIVACY-KERNEL | 2-3 wk | Empirical | Start (low risk, clear table) |
| **N7** | JUDGE-AUDIT | 2-3 wk | Method | Start (extends 2601.08343, methodological) |
| **N9** | PROCESS-VERIFY | 4-6 wk | Method | Mid |
| **N11** | AURELIUS-EVAL | 6-8 wk | Benchmark | Mid |
| **N13** | SLINGSHOT-DEFEND | 3-4 wk | Method | Mid |
| **N3** | AXPO+BPPO+Pair-GRPO | 4-6 wk | Method | Mid |
| **N8** | MARLSAFE-CMDP | 6-8 wk | Method | Late (high impact, high effort) |
| **N1** | ZO-SimPO | 2-3 wk | Method | Late (v3 carryover) |
| **N5** | MemGuardian | 8-12 wk | Method | Late (v3 carryover) |
| **N12** | KV-LIBRARY | 8-12 wk | Library+Method | Late (foundational, longest tail) |

**Recommended Q3-Q4 2026 sequencing:**
1. N10 (PRIVACY-KERNEL) — 2-3 weeks, single-table result, can publish first.
2. N6 (TRUSTTIER) — 3-4 weeks, builds on existing Pancake work, clear method contribution.
3. N7 (JUDGE-AUDIT) — 2-3 weeks, methodological, fast publication.
4. N9 (PROCESS-VERIFY) — 4-6 weeks, the headline method paper for v4.
5. N11 (AURELIUS-EVAL) — 6-8 weeks, the benchmark paper.
6. N8 (MARLSAFE-CMDP) — 6-8 weeks, the safety/privacy paper.
7. N12 (KV-LIBRARY) — 8-12 weeks, the foundational infrastructure paper.

---

## 8. Recommendations for Master Plan v4

1. **Add Pillars 0-6 explicitly with the new synthesis above.**
2. **Add N6-N13 to the Novel Contribution Opportunities section.**
3. **Add B23-B32 to the Known Blind Spots section.**
4. **Update the alignment stack** to include Gradient Regularization (2602.18037) + DyJR (2603.16157) + VeRPO (2601.03525) + HISR (2603.18683) on top of the existing AXPO+BPPO+Pair-GRPO stack.
5. **Update the inference stack** to include KV-LIBRARY (N12) as the unifying infrastructure and trust-tiered cache (N6) for the security dimension.
6. **Replace "MemGuardian" (N5) with "MARLSAFE-CMDP" (N8)** as the privacy-pillar flagship; MemGuardian can remain as supporting work.
7. **Add a Process Supervision Recipe** section that lists CurioSFT → TreePS-RAG → HISR → VeRPO → Rubric-GRM as the small-LM reasoning pipeline.
8. **Add a Multi-Agent Evaluation Recipe** section that lists STAT + ARMS + Cattle Trade + HC-MAPPO-L as the test bed for multi-agent contributions.

---

*End of synthesis. See deepdive_G1.md through deepdive_G5.md for per-paper analysis. See synthesis-report.md (final report) for executive summary.*
