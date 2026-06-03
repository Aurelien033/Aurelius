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
