# AURELIUS RESEARCH PLAN — 2026-05-28 (v2)

**Generated**: 2026-05-28 (continuous research loop, 24 reports, ~640KB analysis)  
**Last updated**: 2026-05-28 23:25 UTC — Batch 15: DeepSeek V3.2 (DSA sparse attention, specialist distillation, GRPO with unbiased KL/off-policy masking), Llama 4 (iRoPE, 10M context, MetaP hyperparam optimization, MoE 109-400B), MiMo V2 Flash (5:1 hybrid SWA+GA with attention sink bias, MOPD multi-teacher on-policy distill, MTP 2.6x speedup). Top Aurelius recommendations: hybrid SWA+GA (easiest high-impact change), MOPD (best alignment paradigm), GRPO scaling tricks from DeepSeek.  
**Context**: Custom transformer (6.8B GQA/MoE/MLA) with AMC memory, SimPO training, spec decoding

---

## I. CURRENT STATE

| Thing | What Aurelius Has | Gap |
|-------|-------------------|-----|
| Architecture | GQA, MoE (top-2), MLA, SwiGLU, RMSNorm | No Oryx-style per-token mixing, no P-RoPE, no UNIQUE, no Tensor Memory |
| Training | Accelerate/DeepSpeed, Muon+AdamW, SimPO | No AdaDPO, no G2D pipeline, no symmetry optimizer |
| Memory | AMC (activation memory consolidation) | No PEAM/MemCog consolidation, no FluxMem evolution |
| Inference | Speculative decoding, KV cache (PackKV) | No EAGer branching, no UNIQUE/NestedKV, no EvoSpec vocab adapt |
| Safety | safety_token_regularization | No Ellipsoid Control, no ACT, no SPARD; alignment tampering undefended |
| Research loop | 4h cron sweep + deep-dive agents | No trial traces, no meta-loop, no coverage tracking |
| Tooling | vLLM, transformers | No Liger-Kernel, no lm-eval-harness, no Unsloth |

---

## II. IMMEDIATE ACTION (Week 1) — <10 line changes

These are already implemented or have ready-to-merge patches:

| # | Change | Code | Effort | Gain |
|---|--------|------|--------|------|
| 1 | **AdaSimPO** — per-pair adaptive coefficients | `src/alignment/simpo.py` | 1 day ✓ | 1-3% win rate |
| 2 | **Symmetry MoE Optimizer** — row-norm router updates | `src/training/symmetry_optimizer.py` | 2 days ✓ | Stable routing |
| 3 | **Install Liger-Kernel** — fused Triton ops | `pip install liger-kernel` | 1 hr | +20% training speed, -60% memory |
| 4 | **Install lm-eval-harness** — standardized evals | `pip install lm-eval` | 1 hr | Reproducible benchmarks |
| 5 | **Install Langfuse** — LLM observability | `pip install langfuse` | 1 day | Trace, debug, monitor |

**⚠️ Caveat**: AdaSimPO assumes pairwise comparisons. **GraphDPO** (2605.08037) shows multi-preference graphs outperform pairs. If confirmed at scale, the alignment stack should use graph-level objectives instead. See Section XI.

**Total: ~3 days.** Zero-retraining, zero-risk infrastructure + loss function improvements.

---

## III. WEEK 2-4 — Inference Optimization (no retraining)

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 6 | **EAGer entropy-aware decoding** | 2510.11170 | 5 days |
| 7 | **NestedKV cache compression** | 2605.26678 | 1 week |
| 8 | **Ellipsoid Control safety guard** | 2605.24552 | 1 week |
| 9 | **RotMoLE in MoE-LoRA fine-tuning** | 2605.25565 | 3 days |
| 10 | **BPPO: 6x GRPO speedup** | 2605.28028 | 3 days |

**New addition**: BPPO (2605.28028) is algorithm-agnostic — compatible with Pair-GRPO, DAPO, GSPO, any GRPO variant. Uses binary completion selection + prefix-only updates. ~0.5% accuracy loss for **6.08x speedup**. Highest priority addition in this batch.

All are **training-free** — modify inference path or add lossless safety layers.

---

## IV. MONTH 2-3 — Architecture + Training Upgrades

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 11 | **G2D+SimPO alignment pipeline** | 2605.21266 + AdaDPO + SC-SDPO | 2-3 weeks |
| 12 | **P-RoPE for infinite context** | 2605.27980 | 2-4 weeks |
| 13 | **UNIQUE sparse attention** | 2605.27740 | 1-2 weeks |
| 14 | **MoA activation mixing in FFN** | 2605.26647 | 1-2 weeks |
|| 15 | **PEAM-style AMC consolidation** — ADOPT: PEAM's scoring (PV) + trigger (STC) concepts; REJECT: LoRA+trajectory internalization (domain-specific) | 2605.27762 | 1 week (scoring/trigger only) |
|| 16 | **Tensor Memory — fixed-size recurrent state** | 2605.27686 | 2-3 weeks |
|| 16A | **Meta-Attention — Bayesian per-token routing** (fills per-token mixing gap) | 2605.28384 | 1 week |
|| 16B | **Attention Sink → Native MoE — head collapse fix** — sink-aware auxiliary loss, +0.8-1.4 avg accuracy, <2% overhead | 2602.01203 | 3 days |
|| 16C | **MGRetrieval — pyramid retrieval** to augment AMC flat retrieval | 2605.27437 | 1 week |

**New addition**: Tensor Memory (2605.27686) fills a gap the plan previously had — a genuine linear-complexity recurrent architecture for long-horizon transformers. Unlike UNIQUE (sparse attention, still O(n) per token) or P-RoPE (infinite position but still full attention cost), Tensor Memory provides fixed-size processing state. It's complementary: P-RoPE handles the position encoding, UNIQUE handles sparse attention, Tensor Memory handles the memory bottleneck.

---

## V. MONTH 4-12 — Research Frontier

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 17 | **GPRL + Pair-GRPO hybrid** (combined) | 2605.18721 + 2605.06375 | 3-4 weeks |
| 18 | **MemCog navigable memory** | 2605.28046 | 4-6 weeks |
| 19 | **Self-Verified Distillation pipeline** | 2605.26132 | 2-3 weeks |
| 20 | **DelTA token credit for GRPO** | 2605.21467 | 1-2 weeks |
| 21 | **Bilevel AutoResearch meta-loop** | 2603.23420 | 3-4 weeks |
| 22 | **CIRF functional tokens** | 2605.28292 | 2-3 weeks |
| 23 | **Alignment tampering defense** | 2605.27355 + 2605.18309 | 3-4 weeks |

**Updated recommendation**: **Combine GPRL + Pair-GRPO**, don't replace one with the other.
- GPRL (2605.18721) contributes: multi-dimensional GPM reward, per-dimension advantages, drift monitoring
- Pair-GRPO (2605.06375) contributes: local probability constraints (Hard-Pair-GRPO), gradient equivalence theorem, formal convergence guarantees
- Together: GPRL's GPM reward → per-dimension advantages → Hard-Pair-GRPO's constrained optimization win
- Pair-GRPO alone is just a better GRPO optimizer. GPRL alone lacks convergence guarantees. Together they're strictly better than either alone.
- Gradient equivalence theorem shows Soft-Pair-GRPO ≈ GRPO under Taylor expansion — the binary reward switch is "free"

**⚠️ CRITICAL UPDATE — GRPO structural flaws confirmed by 3 independent papers:**
1. **Uniform token weighting is structurally wrong** — GRPO treats all tokens equally but correct intermediate steps get penalized when the final answer is wrong (CalibAdv 2604.18235). All GRPO-based methods (Pair-GRPO, G2D, DelTA) need token-level advantage decomposition.
2. **Tool use is under-trained** — GRPO spends ~70% of rollouts on text reasoning, neglecting action tokens (AXPO 2605.28774). Fix: tool-call resampling where +25% resampling budget > +100% rollout budget.
3. **Agent RL has unique dynamics** — cyclical entropy eruption, not monotonic collapse seen in math RL (CyclEnt 2605.27954). Caused by high representation similarity inducing gradient interference. Fix: SEAL (representation separation loss).
4. **Multimodal models collapse under GRPO** — MAPO (2605.27741) shows uniform token weighting causes late-stage modality abandonment. Fix: differential entropy mask + attention penalty.

**Implication for plan**: Any GRPO-based alignment method MUST include token-level advantage decomposition + SEAL-style representation separation as prerequisites. Without these, GRPO is fundamentally flawed for agentic and multimodal training.

**New**: Alignment Tampering (2605.27355) reveals that the LLM can influence its own preference data, causing RLHF to amplify misaligned behaviors. None of the plan's alignment methods defend against this. Requires dedicated mitigation: preference data provenance tracking, adversarial rollout filtering, independent reward verification.

---

## VI. SYNERGY PRIORITIES (Multiplicative Combinations)

| Priority | Combination | Effort | Total Gain |
|----------|------------|--------|------------|
| **P0** | AdaDPO + Symmetry Optimizer + Liger-Kernel | 3 days ✓ | Training quality + efficiency |
|| **P1** | **AXPO + BPPO + Pair-GRPO** (combined agentic alignment pipeline) | 2-3 weeks | AXPO's Thinking-Acting Gap fix + BPPO's 6x speedup + Pair-GRPO's convergence guarantees |
| **P2** | EAGer + Speculative Decoding | 1 week | Token budget reallocation |
| **P3** | UNIQUE + NestedKV + GQA + PackKV | 3-4 weeks | 16-32x total KV reduction |
| **P3** | G2D + AdaDPO + SC-SDPO + Pair-GRPO | 3-4 weeks | 4x compute reduction + formal guarantees |
| **P4** | PEAM + MemCog + AMC + Tensor Memory + MemGuard + MGRetrieval | 10-12 weeks | Complete memory stack: retrieval → contamination prevention → consolidation → navigation → fixed-size recurrence |
| **P4** | **Liger-Kernel** (RMSNorm 3-8x speedup, -71% memory) — **⚠️ MLA gap: no kernel for Aurelius's defining attention mechanism** | Partial — install for non-MLA ops only |

**Updated**: P1 now includes EvoSpec. P3 now includes Pair-GRPO as theoretical upgrade over pure G2D. P4 adds Tensor Memory for the linear-time recurrence layer.

---

## VII. CONTRADICTIONS TO MONITOR

| Debate | Our Position | Watch For |
|--------|-------------|-----------|
| DPO vs GRPO | G2D hybrid (short warmup + offline) | GraphDPO may supersede both with graph objectives |
| Sparse vs Full Attention | UNIQUE + RLKV (both) | MLA-specific compression results |
| Chinchilla vs Shannon | Shannon SNR framework | Empirical validation on Aurelius |
| Black-list vs White-list Safety | Ellipsoid Control (both layers) | New adversarial benchmarks |
| Memory as Tool vs Cognition | Maturity stages (AMC→PEAM→MemCog→FluxMem) | Each stage validated before next |
| **Pairwise vs Graph Preferences** | AdaDPO assumes pairs — GraphDPO challenges this | GraphDPO scale-up results |
| **Pair-GRPO vs GPRL** | COMBINE: GPRL reward + Pair-GRPO optimizer | Need validation at 70B+ scale |
| **Online RL vs Offline DPO** | G2D says offline with informative rollouts suffices | If GRPO keeps winning at scale, more online needed |
| **Alignment Tampering Feasibility** | Likely real, mitigation unknown | Leakage detection benchmarks |

---

## VIII. KNOWN BLIND SPOTS (NEW)

These are failure modes that the plan's recommended methods do not address:

| Blind Spot | Paper | Description | Impact on Plan |
|-----------|-------|-------------|---------------|
| **Alignment Tampering** | 2605.27355 | LLM influences its own preference data → RLHF amplifies misalignment | All alignment methods assume preference data is exogenous. Detection possible (AUROC 0.74) via PCA+dip test. |
| **RLHF Artifacts Survive** | 2605.28102 | Training Stratigraphy: 5 artifacts persist after prompt replacement (sexual latency, attention absorption, entity blindness, attention-RLHF antagonism α1(S_t), anti-hallucination suppressing first-person) | Safety/reward model contamination may outlast alignment pipeline. Aurelius needs longitudinal eval (47K msgs, 8 months). |
| **No Memory Evaluation Standard** | AgingBench 2605.26302, TriMem 2605.19952 | AgingBench tests agent lifespan (function-calling degradation over 1-60+ round deployments). TriMem proposes lifelong memory beyond atomic facts. | Closest existing work but neither evaluates parametric memory consolidation (AMC). Must build custom eval. |
| **Benchmark Saturation** | TASTE 2605.28556 | Models score 0.82-0.94 on τ²-Bench → drop to 0.28-0.61 on harder TASTE tasks (35-60pt drop) | All plan recommendations need validation on fresh (not saturated) benchmarks |
| **Framework > Model** | SNARE 2605.28122 | Agent framework accounts for 56% of safety variance vs model's 21% | Aurelius agent architecture choices may dominate model quality for safety |
| **Runtime Collapse** | RAMP 2605.27492 | Task completion drops 100%→20% across serial workflows | Static benchmarks miss critical failure modes |
| **Alignment Fragility** | 2605.18309 | Alignment degrades under any further fine-tuning | Even perfectly aligned models need ongoing maintenance; alignment is not one-time |
| **RLVR doesn't generalize** | Multiple papers | RLVR/GRPO training only helps math/code, not general QA | G2D and DelTA effectiveness limited to narrow domains |
| **GRPO Under-Trains Tool Use** | AXPO 2605.28774 | GRPO focuses on text reasoning, neglects action-level tool use (Thinking-Acting Gap) | G2D and Pair-GRPO for agentic tasks may underperform; need agent-specific alignment |
| **GRPO Penalizes Correct Steps** | CalibAdv 2604.18235 | When final answer is wrong, GRPO penalizes correct intermediate reasoning | Credit assignment in GRPO is fundamentally flawed for multi-step reasoning |
| **GRPO Causes Modal Collapse** | MAPO 2605.27741 | In multimodal models, GRPO abandons source signals for text priors | Relevant if Aurelius goes multimodal — GRPO may destroy vision capabilities |
|| **Agent RL Unique Dynamics** | Cyclical Entropy 2605.27954 | Agent RL entropy cycles differ fundamentally from single-turn reasoning RL | Memory-augmented agent training may need fundamentally different alignment |
| **Prompt Optimization May Be Harmful** | Coin Flip 2604.14585 | 49% of prompt optimization runs score below zero-shot baseline | Assumption that prompt engineering always helps may be wrong. Test before deploying. |
| **LLMs Can't Do Causal Discovery** | Kernel Obstruction 2605.27567 | Formal proof: LLMs cannot distinguish causal graphs from observational data alone | Agentic causal reasoning needs explicit causal models, not just LLM inference |
|| **SkillSurface Attacks** | SkillSafetyBench 2605.12015 | All agent components are attack surfaces (plugins, skills, memory, tools, Docker, search) — 15-50% ASR on all current CLI agents | Every Aurelius component needs security audit; scaffold choice dominates vulnerability profile |
|| **Eval Meta-Knowledge Gaming** | 2605.28591 | Models trained on documents about evaluation design score safer without verbalizing awareness — structural leakage, not data contamination | Aurelius pretraining corpus may contain eval methodology docs → eval scores may overstate true safety by unknown margin |
|| **Reasoning ≠ Tool-Use Training** | AXPO 2605.28774 | GRPO creates structural Thinking-Acting Gap: tool-using subgroups all-wrong ~40% of time, suppressing learning signal at the tool calls | G2D+Pair-GRPO pipeline doesn't address this. Need AXPO-style resampling or alternative agentic alignment mechanism |
|| **Memory Privacy Leakage** | MRMMIA 2605.27825 | Members can be identified from agent memory with AUC 0.99-1.00 via multi-recall probes. System prompt defenses ineffective. No known defense exists. | AMC's compressed activations likely detectable. Privacy-preserving consolidation is a research gap. |
|| **Alignment Tampering — No Defense Exists** | 2605.27355 + Directional Alignment 2605.25189 | Detection possible (AUROC 0.74). Closest mitigation: directional alignment (gradient projection to trusted subspace). No dedicated defense paper. | NOVEL CONTRIBUTION OPPORTUNITY: Design first alignment tampering defense for SimPO. |

---

## IX. PAPERS THAT COULD CHANGE RECOMMENDATIONS (NEW)

| Paper | Current Recommendation | Threat Level | Why |
|-------|-----------------------|-------------|-----|
| **Pair-GRPO** (2605.06375) | GPRL in Phase 4 | **COMBINED** | Not a replacement — combined with GPRL (GPRL reward → Pair-GRPO optimizer) |
| **GraphDPO** (2605.08037) | AdaDPO in Phase 1 | MEDIUM | If graph-level objectives beat pairwise at scale, AdaDPO is suboptimal |
| **Tensor Memory** (2605.27686) | No linear-complexity option | **ADDED ✓** | Fills a genuine gap — plan now includes it. Official MIT code exists. |
| **EvoSpec** (2605.27390) | Static spec decoding | **HOLD** | Good concept (1.13-1.16x speedup) but no public code. Add after verification. |
| **BPPO** (2605.28028) | G2D pipeline in Phase 4 | **HIGH** | Algorithm-agnostic 6x speedup, directly combinable with Pair-GRPO. If confirmed, should be integrated before any GRPO training. |
| **Alignment Tampering** (2605.27355) | No defense | **CRITICAL** | Validated: applies to SimPO. Detection possible (AUROC 0.74). No mitigations work without quality loss. Add monitoring. |

---

## X. AUTORESEARCH LOOP UPGRADE PATH

| Stage | Current State | Next Step | Trigger |
|-------|---------------|-----------|---------|
| T1 | ✅ Keyword arxiv sweep + deep-dive agents | Add structured JSON trial traces | This week |
| T1+ | ✅ Cron every 4h | Add coverage tracking matrix | This week |
| T2 | ❌ No cross-session memory | Build EvolveMem-style paper DB | Month 1 |
| T2+ | ❌ No diversity scoring | Add cosine novelty filter | Month 1 |
| T3 | ❌ No meta-loop | Bilevel: outer loop tunes inner loop | Quarter 2 |
| T3+ | ❌ No self-evolving prompts | ADAS-style meta-agent | Year 1 |

---

## XI. INFRASTRUCTURE TRACK

| Tool | Install | Purpose |
|------|---------|---------|
| Liger-Kernel | P0 | Training speed (+20%, -60% memory) |
| lm-eval-harness | P0 | Standardized evals |
| Langfuse | P0 | Observability, prompt management |
| Unsloth | P1 | Fine-tuning speed (2x) |
| SGLang | P2 | Prefix caching (RadixAttention) |
| OpenRLHF | P3 | Multi-turn RLHF |
| dstack | P2 | GPU orchestration |

---

## XII. NEXT RESEARCH SWEEP PRIORITIES

For the continuous cron loop (deployed, runs every 4h):

1. **Track coverage gaps** — the 6-slot taxonomy matrix should guide sweep focus
2. **Monitor GraphDPO** — if it scales beyond current results, pairwise DPO may be obsolete
3. **Monitor Pair-GRPO** — need to see if formal guarantees hold at 70B+ scale
4. **Monitor Alignment Tampering** — need leakage detection and mitigation papers
5. **Deep-dive Agentic Evaluation** — 162 papers found with score >=3, this area is exploding
6. **Every 4th run**: focus on one underpopulated SLOT
7. **Monthly**: generate updated landscape document

---

## XIII. RESEARCH LOOP COVERAGE MATRIX

Last updated: 2026-05-28

| Slot | Papers Covered | Deep-Dives Done | Depth Score | Next Sweep Target |
|------|---------------|----------------|-------------|-------------------|
|| A (Sequence Mixing) | 14 | 5 (Oryx, UNIQUE, P-RoPE, B^3D-RWKV, Meta-Attention) | HIGH | Tensor Memory, Grammatically-Guided Sparse |
|| B (Parameter Efficiency) | 12 | 5 (RotMoLE, MoA, Symmetry Optimizer, CIRF, Structured Distillation) | HIGH | ArcGate, PEFT-Arena |
|| C (Memory & Context) | 27 | 9 (PEAM, MemCog, FluxMem, NestedKV, RLKV, EAGer, P-RoPE, ConvMemory, MGRetrieval) | VERY HIGH | IndexMem, MemGuard, CrossAug |
|| D (Training & Alignment) | 18 | 10 (AdaDPO, SC-SDPO, G2D, GPRL, DelTA, Shannon, Self-Verified, Pair-GRPO, GraphDPO, GPRL) | VERY HIGH | BPPO, Training Stratigraphy |
|| E (Normalization & Inference) | 17 | 6 (Symmetry, EAGer, RLKV, Outer-Momentum, Attention Sink, DREAM-R) | MEDIUM | Liger-Kernel deep-dive, StableGrad |
|| F (Safety & Interpretability) | 26 | 7 (Ellipsoid Control, ACT, CRaFT, SPARD, SA-GSAE, SA-OT, SkillSafetyBench) | HIGH | Alignment Tampering, Open-Weight Defenses |
|| **AutoResearch** (cross-cutting) | 10 | 5 (Bilevel, EvolveMem, Sibyl, GEAR, ADAS) | VERY HIGH | ARIS, NanoResearch |
|| **NEW: Agentic Evaluation** | 162 | 2 (landscape + HRBench+AMA-Bench+UnifiedEval) | LOW | **DEEPEST GAP** — no memory eval for parametric consolidation, SkillSecurity critical |

---

*This plan is derived from ~1.4MB of synthesized research across 46 reports, covering ~195 papers, 50+ GitHub repos, and 8,000+ papers screened across 10 batch sweeps (4 manual today). Batch 10: MemGuard (2605.28009) adds memory contamination prevention to the memory stack; FPMoE (2605.27849) provides MoE routing topology template; HGMEM (2512.23959) upgrades AMC T1 working memory to hypergraph; Liger-Kernel P0 confirmed but CRITICAL MLA gap — no kernel for Aurelius's defining attention mechanism; TaperNorm (2602.10408) most promising norm alternative; EgoBench+VeriTrip provide evaluation methodology templates; 4 reasoning efficiency approaches found for 1.4B scale — no small-model reasoning efficiency benchmark exists (novel opportunity). 3 infrastructure tools confirmed uninstalled. Plan now at 245+ lines, 4 novel contribution opportunities.*
