# AURELIUS RESEARCH PLAN — 2026-05-29 (v3 — Major Update)

**Generated**: 2026-05-29 (16 deep formal analyses, 20+ syntheses, ~1.5MB total)  
**Last updated**: 2026-05-29 — Added RLVR Difficulty, UDM-GRPO, First-Token Diversification, SkillSafetyBench, Hamilton-Jacobi Theory
**Context**: Custom transformer (6.8B GQA/MoE/MLA) with AMC memory, SimPO training, spec decoding

---

## I. CURRENT STATE

| Thing | What Aurelius Has | Gap |
|-------|-------------------|-----|
| Architecture | GQA, MoE (top-2), MLA, SwiGLU, RMSNorm | No Oryx-style per-token mixing, no P-RoPE, no UNIQUE, no Tensor Memory |
| Training | Accelerate/DeepSpeed, Muon+AdamW, SimPO | No AdaDPO, no G2D pipeline, no symmetry optimizer |
| Memory | AMC (activation memory consolidation) | No PEAM/MemCog consolidation, no FluxMem evolution |
| Inference | Speculative decoding, KV cache (PackKV) | No EAGer branching, no UNIQUE/NestedKV, no EvoSpec vocab adapt |
| Safety | safety_token_regularization | No Ellipsoid Control, no ACT, no SPARD; alignment tampering undefended; skillsurface attacks unaddressed |
| Research loop | 4h cron sweep + deep-dive agents | No trial traces, no meta-loop, no coverage tracking |
| Tooling | vLLM, transformers | No Liger-Kernel, no lm-eval-harness, no Unsloth |

---

## II. IMMEDIATE ACTION (Week 1) — <10 line changes

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

**BPPO** (2605.28028) is algorithm-agnostic — compatible with Pair-GRPO, DAPO, GSPO, any GRPO variant. Uses binary completion selection + prefix-only updates. ~0.5% accuracy loss for **6.08x speedup**.

**NEW: REFT (First-Token Diversification) — ~20 line change** — Uniformly samples K=4 first tokens from policy's own top-N=20 candidates. Allocates G/K=2 rollouts per token. Improves Pass@1 by 2-5% across 4 model sizes (0.5B-7B) and 3 difficulty regimes. Complementary to all GRPO variants. (2605.28295)

All are **training-free** — modify inference path or add lossless safety layers.

---

## IV. MONTH 2-3 — Architecture + Training Upgrades

| # | Change | Papers | Effort |
|---|--------|--------|--------|
| 11 | **G2D+SimPO alignment pipeline** | 2605.21266 + AdaDPO + SC-SDPO | 2-3 weeks |
| 12 | **P-RoPE for infinite context** | 2605.27980 | 2-4 weeks |
| 13 | **UNIQUE sparse attention** | 2605.27740 | 1-2 weeks |
| 14 | **MoA activation mixing in FFN** | 2605.26647 | 1-2 weeks |
| 15 | **PEAM-style AMC consolidation** | 2605.27762 | 1 week (scoring/trigger only) |
| 16 | **Tensor Memory — fixed-size recurrent state** | 2605.27686 | 2-3 weeks |
| 16A | **Meta-Attention — Bayesian per-token routing** | 2605.28384 | 1 week |
| 16B | **Attention Sink → Native MoE — head collapse fix** | 2602.01203 | 3 days |
| 16C | **MGRetrieval — pyramid retrieval** | 2605.27437 | 1 week |
| 16D | **REFT first-token diversification** (see Section III) | 2605.28295 | ~20 lines |

**New**: REFT is now the easiest high-ROI change in the plan. It's a ~20 line modification to the rollout sampler that diversifies first-token selection — it directly mitigates GRPO's trajectory-level advantage amplification of first-token bias. The mechanism: first token has low semantic load but high distributional leverage. Policy probability and verifier correctness decouple at this position (top-1 mean prob 0.57, but correctness nearly flat across 20 ranks).

**Hamilton-Jacobi Framework**: The HJ Theory of Deep Learning (2605.28983) reveals that neural networks ARE viscous Hamilton-Jacobi equations. A single deformation parameter ε unifies (ℝ,+×) algebra → (ℝ,max,+) tropical algebra → viscous PDE → neural computation. Key implications:
- **Hallucination = deterministic OOD**: When ∆(x)/ε >> log N, output is exponentially close to dominant neuron's linear extrapolation. No loss function can align behavior outside training support.
- **Data intrinsic dimension**: Raw internet has d_eff ≈ 13; domain-specific has d_eff ≈ 2.6-2.9. Each halving of approximation error costs 2^{d_eff} more neurons.
- **Attention = exact Hopf-Cole**: L2 attention equals gradient of LogSumExp contracted with values — exact identity, no approximation.
- Backprop is the adjoint HJ equation. Double descent is a near-shock at interpolation threshold.

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
| 24 | **DETER-RT runtime context integrity** | 2605.12015 (SkillSafetyBench) | 4-6 weeks |

**DETER-RT extension**: SkillSafetyBench (2605.12015) reveals that every agent component is an attack surface — 154 adversarial cases, 47 tasks, 6 risk domains, 8 attack classes. ASR ranges 15.5-50.3% across 15 models and 5 scaffolds. The DETER alignment tampering defense (training-time focus) needs a runtime counterpart:
1. Artifact anomaly detection over execution traces
2. Provenance-aware skill decomposition
3. Counterfactual skill-context validation
4. Task-success-conditional ASR as a first-class safety metric

**Continued recommendation**: **Combine GPRL + Pair-GRPO**, don't replace one with the other.
- GPRL (2605.18721) contributes: multi-dimensional GPM reward, per-dimension advantages, drift monitoring
- Pair-GRPO (2605.06375) contributes: local probability constraints (Hard-Pair-GRPO), gradient equivalence theorem, formal convergence guarantees
- Together: GPRL's GPM reward → per-dimension advantages → Hard-Pair-GRPO's constrained optimization win

---

## VI. GRPO STRUCTURAL FLAWS TAXONOMY (EXPANDED)

**Convergent finding: GRPO uniform token weighting is structurally wrong — now confirmed by 8 independent papers across 5 levels of analysis.**

| Level | Flaw | Paper | Evidence | Fix |
|-------|------|-------|----------|-----|
| **Step-level** | Correct steps penalized when final answer wrong | **CalibAdv** 2604.18235 | 43.97% of GRTO-penalized steps are correct | Soft penalization, rebalancing, prefix exclusion |
| **Token-level** | First token is routing variable with decoupled prob-correctness | **REFT** 2605.28295 | Top-1 prob 0.57 but correctness flat across 20 ranks | First-token diversification (REFT) |
| **Tool-level** | Text reasoning dominates; tool tokens under-trained | **AXPO** 2605.28774 | ~30% tool-use rate, ~40% all-wrong; +25% budget > +100% | Tool-call resampling |
| **Representation-level** | Agent RL entropy cycles, gradient interference | **CyclEnt** 2605.27954 | 3 phases; gradient interference (Lemma 3.2-3.4) | SEAL representation separation |
| **Modality-level** | Uniform weighting causes modal collapse | **MAPO** 2605.27741 | Cross-modal Δh_t; AIF 62→95 | Differential entropy mask |
| **Difficulty-level** | Zero-variance samples vanish under normalization | **RLVR Difficulty** 2605.28388 | Easy/Hard produce zero advantage signal; Medium best | RFGO (T-SAE-guided) |
| **Timestep-level** | Inaccurate intermediate actions in diffusion | **UDM-GRPO** 2604.18518 | Early denoising predictions are high-entropy noise | Clean action + forward trajectory |
| **Trajectory-level** | Backward trajectory distribution shift | **UDM-GRPO** 2604.18518 | RL backward ≠ pretraining forward distribution | Forward trajectory reconstruction |

**Common thread**: GRPO's group-relative advantage normalization Âᵢ = (Rᵢ - mean(R)) / std(R) assumes uniform action quality across a trajectory. This assumption is false at every level: token, step, tool, modality, difficulty, timestep, trajectory.

**Implication for all GRPO-based methods**: Any GRPO implementation MUST include:
- Token-level advantage decomposition (DelTA or equivalent)
- First-token diversification (REFT)
- Difficulty-adaptive data selection (exclude zero-variance)
- SEAL-style representation separation (if agentic)
- Tool-call resampling (if agentic)

**NEW: Cross-domain generality confirmed** — UDM-GRPO (text-to-image, ICML 2026 Spotlight) shows the same structural flaws manifest in continuous diffusion. The flaws are not limited to language GRPO but are a property of the group-relative advantage mechanism itself.

---

## VII. BLIND SPOTS (UPDATED)

| Blind Spot | Paper | Key Finding | Impact on Plan |
|-----------|-------|-------------|---------------|
| **Alignment Tampering** | 2605.27355 | LLM influences its own preference data → RLHF amplifies misalignment | All alignment methods assume preference data is exogenous. Detection possible (AUROC 0.74) via PCA+dip test. |
| **RLHF Artifacts Survive** | 2605.28102 | 5 artifacts persist after prompt replacement | Safety/reward model contamination may outlast alignment pipeline |
| **No Memory Evaluation Standard** | AgingBench, TriMem | No benchmark evaluates parametric memory consolidation (AMC) | Must build custom eval |
| **Benchmark Saturation** | TASTE 2605.28556 | Model scores drop 35-60pt from τ²-Bench to TASTE | All recommendations need fresh-benchmark validation |
| **Framework > Model** | SNARE 2605.28122 | Scaffold accounts for 56% of safety variance vs model's 21% | Agent architecture choices may dominate model quality |
| **Runtime Collapse** | RAMP 2605.27492 | Task completion drops 100%→20% across serial workflows | Static benchmarks miss critical failure modes |
| **Alignment Fragility** | 2605.18309 | Alignment degrades under any further fine-tuning | Alignment is not one-time; needs ongoing maintenance |
| **SkillSurface Attacks** | **SkillSafetyBench** 2605.12015 | 15-50% ASR across all tested agents; 8 attack classes | Every Aurelius component needs security audit |
| **Eval Meta-Knowledge Gaming** | 2605.28591 | Models trained on eval docs score safer without verbalizing awareness | Scores may overstate safety by unknown margin |
| **Memory Privacy Leakage** | MRMMIA 2605.27825 | AUC 0.99-1.00 for membership inference from agent memory | AMC's compressed activations likely detectable |
| **NEURAL NETWORKS ARE POV** (HJ Theory) | 2605.28983 | Networks = Hamilton-Jacobi equations. Hallucination = deterministic OOD extrapolation | Architectural choices are constrained by PDE semantics |
| **Data Intrinsic Dimension** | 2605.28983 | d_eff ≈ 13 (raw web) vs 2.6 (math). Each halving of error costs 2^{d_eff} neurons | Domain specialization is exponentially more efficient than scale |

---

## VIII. NOVEL CONTRIBUTION OPPORTUNITIES

| # | Proposal | Basis | Status | Priority |
|---|---------|-------|--------|----------|
| 1 | **GraphAdaDPO** — Multi-preference graph optimization replacing pairwise AdaDPO | GraphDPO 2605.08037 + AdaDPO | Unconfirmed at 70B+ | P0 |
| 2 | **DETER — Alignment Tampering Defense** | 2605.27355 + Directional Alignment 2605.25189 | NO PRIOR ART CONFIRMED | P0 |
| 3 | **DETER-RT — Runtime Context Integrity** | SkillSafetyBench 2605.12015 + DETER | Novel extension | P1 |
| 4 | **MemEval-AMC** — Parametric memory consolidation benchmark | AgingBench + TriMem | No existing benchmark tests AMC-style memory | P1 |
| 5 | **Pair-GRPO-GPM** — GPRL reward + Pair-GRPO optimizer | 2605.18721 + 2605.06375 | Zero public implementations | P2 |
| 6 | **GRPO Structural Flaws Taxonomy** — Unified catalog of all 8 flaws | 8 papers, 5 levels | Unique synthesis, publishable survey | P2 |
| 7 | **REFT+AXPO** — Joint first-token + tool-call diversification | 2605.28295 + 2605.28774 | Should be combined; no joint method exists | P3 |

**P0 Priority**: GraphAdaDPO and DETER are the strongest novel contributions. Both have confirmed NO PRIOR ART.

---

## IX. RESEARCH LOOP COVERAGE MATRIX (UPDATED)

| Slot | Papers Covered | Deep-Dives | Depth Score | Next Sweep Target |
|------|---------------|-----------|-------------|-------------------|
| A (Sequence Mixing) | 15 | 6 (+HJ Theory) | VERY HIGH | Hierarchical attention for MoE |
| B (Parameter Efficiency) | 12 | 5 | HIGH | ArcGate, PEFT-Arena |
| C (Memory & Context) | 27 | 9 | VERY HIGH | IndexMem, MemGuard, CrossAug |
| D (Training & Alignment) | 24 | 14 (+RLVR Diff, +UDM-GRPO, +REFT) | EXTREME | BPPO verification, Training Stratigraphy |
| E (Normalization & Inference) | 17 | 6 | MEDIUM | Liger-Kernel deep-dive, StableGrad |
| F (Safety & Interpretability) | 28 | 8 (+SkillSafety) | VERY HIGH | Alignment Tampering, Open-Weight Defenses |
| **AutoResearch** | 10 | 5 | VERY HIGH | ARIS, NanoResearch |
| **Agentic Evaluation** | 162 | 4 | HIGH | Memory evaluation methodology |

---

## X. NEXT RESEARCH SWEEP PRIORITIES

1. **Track GraphDPO scaling** — if graph-level objectives beat pairwise at 70B+, pairwise methods obsolete
2. **Monitor Pair-GRPO** — need formal guarantee validation at 70B+ scale
3. **Monitor Alignment Tampering** — need leakage detection and mitigation papers
4. **Track new GRPO flaw papers** — the flaw wave may continue. 8 papers across 5 levels is a pattern.
5. **Deep-dive Hamilton-Jacobi appendices** — paper is 4071 lines (monograph). Non-quadratic Hamiltonians (ReLU, GELU, SiLU) and N-soliton tau-functions remain unanalyzed.
6. **Every 4th run**: focus on one underpopulated SLOT
7. **Monthly**: generate updated landscape document

---

*This plan is derived from ~1.5MB of synthesized research across 46+ reports, covering ~220 papers, 50+ GitHub repos, and 8,000+ papers screened across 10 batch sweeps. 16 deep formal PDF analyses completed. GRPO Structural Flaws Taxonomy now spans 8 papers across 5 structural levels. 6 novel contribution opportunities identified. Autonomous cron maintained.*
