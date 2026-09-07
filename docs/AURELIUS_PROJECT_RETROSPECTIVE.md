# Aurelius Project — Comprehensive Retrospective

**Generated:** 2026-05-22  
**Scope:** Full project history — all plans, implementations, design decisions, and research artifacts  
**Status:** Living document

---

## Table of Contents

1. [Project Vision & Genesis](#1-project-vision--genesis)
2. [Architecture — The 1.395B Core](#2-architecture--the-1395b-core)
3. [Data Pipeline](#3-data-pipeline)
4. [Training System](#4-training-system)
5. [Alignment — The Full Arc](#5-alignment--the-full-arc)
   - 5.1 [Original v1 Plan: SFT + DPO + Safety](#51-original-v1-plan-sft--dpo--safety)
   - 5.2 [PRAXIS: Architecture-Aware Alignment](#52-praxis-architecture-aware-alignment)
   - 5.3 [ARIA → AURORA → MOSAIC](#53-aria--aurora--mosaic)
   - 5.4 [APEX: The Grand Unified Framework](#54-apex-the-grand-unified-framework)
6. [Inference Engine](#6-inference-engine)
7. [Harvest Cycles: Research Integration](#7-harvest-cycles-research-integration)
8. [DAIES: AMC Memory & Scaling Ledger](#8-daies-amc-memory--scaling-ledger)
9. [Agent System](#9-agent-system)
10. [API Gateway & Serving](#10-api-gateway--serving)
11. [Frontend & Full-Stack Architecture](#11-frontend--full-stack-architecture)
12. [Canonical Interface Contract](#12-canonical-interface-contract)
13. [Security & Safety Audit](#13-security--safety-audit)
14. [Evaluation & Benchmarking](#14-evaluation--benchmarking)
15. [Infrastructure & DevOps](#15-infrastructure--devops)
16. [Research Paper Pipeline](#16-research-paper-pipeline)
17. [Chronological Milestone Log](#17-chronological-milestone-log)
18. [Summary Statistics](#18-summary-statistics)

---

## 1. Project Vision & Genesis

Aurelius began as a direct response to a single constraint: **no external black-box APIs**. Every algorithm — the transformer core, the optimizer, the alignment trainer, the inference engine — would be implemented natively in PyTorch from the original research papers.

The project was scoped at the outset to three weeks with a ~$350 cloud budget (primarily for H100 time). The target was a **1.3B dense decoder-only transformer** (general reasoning + code generation) that could run locally on an M1 Pro at ~25–35 tokens/second via GGUF quantization, while also being deployable to a cloud inference server at production scale.

The deeper goal was continuous self-improvement: a **brainstorm → plan → implement → evaluate → improve** cycle that would compound over time without requiring any new external labels or human preferences.

### Core Design Principles (Established Day 1)

1. All algorithms are native PyTorch — no HuggingFace Transformers, no flash-attn runtime, no bitsandbytes
2. The model is fully testable with random weights on CPU (no trained checkpoint required for any test)
3. Every plan must be self-contained enough for a downstream agent to implement it without asking clarifying questions
4. No hardcoded absolute paths anywhere in the codebase
5. The alignment pipeline is a research artifact, not just fine-tuning scaffolding

---

## 2. Architecture — The 1.395B Core

### 2.1 Core Specification

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only causal LM |
| Total parameters | ~1.395B |
| Transformer layers | 24 |
| Hidden dimension | 2,048 |
| FFN dimension | 5,632 (SwiGLU: ≈2/3 × 4 × d_model) |
| Attention heads | 16Q / 8KV (GQA, 2:1 ratio) |
| Head dimension | 128 |
| Position encoding | RoPE (θ=500,000) + YaRN context extension |
| Normalization | Pre-norm RMSNorm (no bias) |
| Activation | SwiGLU |
| Vocabulary size | 8,192 (final) — initial design was 128,000 |
| Embeddings | Tied input/output (saves ~262M params) |
| Context length | 8,192 tokens (extendable to 128K via YaRN, factor=4.0) |
| Precision | BF16 training |

### 2.2 Parameter Breakdown

| Component | Parameters |
|-----------|------------|
| Token embeddings (tied) | 262M (shared) |
| 24 × Attention (Q, K, V, O projections) | ~402M |
| 24 × SwiGLU FFN (W1, W2, W3) | ~660M |
| 24 × RMSNorm (pre-attn + pre-ffn) | ~196K |
| **Total (excl. tied embed)** | **~1.06B** |
| **Total (incl. tied embed)** | **~1.32B** |

### 2.3 Architecture Choices and Why

**Grouped-Query Attention (GQA):** 16Q/8KV reduces KV cache to 50% of MHA at the same hidden dimension. This mirrors DeepSeek-V3's component choices at 1.3B scale. The 2:1 ratio was chosen as a balance between MHA quality and MQA efficiency.

**SwiGLU FFN:** Three weight matrices (W1, W2, W3); `FFN(x) = (W1(x) ⊙ SiLU(W3(x))) @ W2`. The 5,632-dim FFN (≈2/3 × 4 × 2048) is the standard SwiGLU-corrected dim rounded to a multiple of 64 for tensor-core efficiency.

**RoPE θ=500,000:** Extended from the original 10,000. Enables longer context without YaRN extension up to ~32K tokens. YaRN adds a second layer of positional extension via `AureliusConfig(rope_scaling_type="yarn", rope_scaling_factor=4.0)`.

**No bias terms:** Improves training stability, reduces parameter count, consistent with modern LLM practice (LLaMA, Mistral, DeepSeek).

**Tied embeddings:** The input embedding matrix and the output projection share weights. At vocab=8,192 and d_model=2,048, this saves 16M parameters (at the final vocab size) — minimal savings, but preserves the design principle.

### 2.4 Forward Contract

The transformer forward signature:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    past_kv: list[tuple[Tensor, Tensor] | None] | None = None,
    return_hidden_states: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, list[tuple[Tensor, Tensor]], ...]:
```

Returns `(loss, logits, present_key_values)` as a plain tuple — **not** an object with `.loss`. This was a critical design decision that broke the initial HuggingFace-style trainer and required an explicit fix (the training loop fix in `2026-04-07-training-loop-fix-design.md`).

When `return_hidden_states=True`, returns `(loss, logits, pkv, x)` where `x` is the final layer's hidden states `(B, T, D)`. This is required by PRAXIS, APEX, and all architecture-aware alignment methods.

### 2.5 Model Extensions (Implemented)

The `src/model/` directory grew to contain **200+ files** covering the full landscape of 2024-2025 LLM architecture research:

**Attention variants:** Flash attention (v2, v3, sim), MLA (Multi-head Latent Attention), GQA-absorbed, sliding window, sparse attention (Longformer), block-sparse, linear attention (v2, v3), chunked attention, ring attention, NyströmFormer, Performer, Hyper-attention, H3, retention, SSMs (Mamba, Mamba2, S4, RWKV, RWKV6, Griffin, Hawk, xLSTM, GLA, HGRN2)

**Position encodings:** RoPE, NTK-aware RoPE, YaRN, LongRoPE, ALiBi, xPos, contextual position encoding, hierarchical RoPE, Fourier position, positional interpolation

**MoE architectures:** SparseMoE (top-k=2, 8 experts), Expert Choice, Soft MoE, MoE router variants (load balancing, z-loss, temperature, noise, collapse prevention), dense→MoE upcycle (`moe_upcycle.py`)

**Efficiency:** Mixture-of-Depths (MoD) v2/v3/v4, token merging, token pruning, token skipping, dynamic depth, early exit, stochastic depth, weight sharing, weight tying

**Multimodal:** Vision encoder, vision cross-attention, multimodal projector, MoonViT patch packer, audio encoder, vision projector

**Long context:** Infini-attention, streaming LLM (attention sink), flash sliding window, hierarchical attention

**Quantization:** BitNet, AQLM, 4-bit/8-bit, FP8 quantization-aware

**Speculative/diffusion:** Medusa heads, MTP (multi-token prediction), diffusion LM head

**Modern architectures:** nGPT, Titans, Delta Net, Linear Recurrent Unit, TTT layer, ReMode, NSA (Native Sparse Attention), MLA-256, DSA (Dynamic Sparse Attention)

### 2.6 MoE Upcycle (v2 Path)

The v2 design calls for upcycling the 24-layer dense model into a ~7B active / 56B total sparse MoE:
- Copy each of 24 FFN layers into 8 expert copies
- Add a top-2 router (nn.Linear(d_model, n_experts))
- Result: 8 experts × 5,632 FFN dim × 24 layers
- `src/model/moe_upcycle.py` implements `upcycle_model()` and `upcycle_ffn()`

---

## 3. Data Pipeline

### 3.1 Pretraining Corpus Design

The v1 design specified a **300B token** corpus with the following mix:

| Source | Tokens | % | Rationale |
|--------|--------|---|-----------|
| FineWeb | 195B | 65% | Primary web corpus, best ablation quality at scale |
| The Stack v2 (code) | 60B | 20% | 619 languages; FIM on 50% of examples |
| FineWeb-Edu | 24B | 8% | 10× benchmark efficiency on educational tasks |
| OpenWebMath | 9B | 3% | Math reasoning foundation |
| Wikipedia + Books | 6B | 2% | Clean factual grounding |
| ArXiv | 6B | 2% | Scientific reasoning depth |

**Chinchilla analysis:** 300B tokens / 1.32B params ≈ 227 tokens/parameter — roughly 2.4× inference-adjusted Chinchilla optimal, designed for production deployment scale.

### 3.2 Processing Pipeline (DataTrove)

```
WARCs / HuggingFace datasets
  → trafilatura extraction
  → URLFilter (adult content blocklist)
  → LanguageFilter (fastText, ≥0.65 confidence English)
  → GopherQualityFilter
  → C4QualityFilter
  → FineWebQualityFilter
  → MinhashDedup (5-gram, 14 buckets × 8 hashes, per-snapshot)
  → SentenceDedup (exact paragraph matching)
  → Tokenizer (aurelius BPE)
  → Writer (Parquet shards, 512MB each)
```

### 3.3 Tokenizer

- **Algorithm:** Byte-level BPE
- **Training corpus:** 10B token sample proportional to pretraining mix
- **Final vocab size:** 8,192 (revised from initial 128,000 design to match training shards)
- **Special tokens:** `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|tool_call|>`, `<|tool_result|>` — 512 total reserved
- **Multi-space tokens:** 2-space and 4-space added for Python indentation

### 3.4 Fill-in-the-Middle (FIM) for Code

50% of code examples use FIM transformation:
- **PSM (50% of FIM):** `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}`
- **SPM (50% of FIM):** `<|fim_suffix|>{suffix}<|fim_prefix|>{prefix}<|fim_middle|>{middle}`

Split points are chosen randomly within the example. Enables tab-completion capability during inference without any architectural changes.

### 3.5 Data System Components (Implemented in `src/data/`)

The data module grew to cover the full 2024-2025 dataset research landscape:

**Loaders:** FineWeb2, The Stack, GitHub code, HuggingFace hub, Kaggle NLP, OpenAI datasets, Anthropic datasets, Civitai, ModelScope/DagsHub, SWE-chat, UsagePipeline

**Quality filters:** Perplexity filter, quality scorer, QURA scorer, LM-based filter, dataset cartography, decontamination, deduplication (MinHash + n-gram)

**Synthesis:** Magpie generator, self-instruct, synthetic CoT, synthetic math, synthetic code, synthetic preference, RAFT pipeline, rejection sampling, FIM transform, backtranslation

**Mixing:** DataMixer, LossAdaptiveMixer, DomainReweighting, curriculum data mixing, importance sampler, difficulty scorer

**Packing:** Sequence packing v2, instruction dataset packer, dynamic batching

**Tokenizer pipeline:** BPE tokenizer (v2), byte tokenizer, tokenization pipeline, tokenizer cache, tokenizer contract

---

## 4. Training System

### 4.1 Core Trainer (`src/training/trainer.py`)

The `AureliusTrainer` implements a hybrid optimizer stack:

- **Muon** (Newton-Schulz 8+2 orthogonalization, Nesterov momentum, RMS rescaling) for weight matrices
- **AdamW** (β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1) for embeddings and norms
- **ZClip** (z-score gradient clipping) — clips based on gradient distribution, not fixed norm

**Learning rate schedule:** Cosine decay with warmup
- Peak LR: 3e-4
- Warmup: 2,000 steps
- Min LR: 3e-5 (10% of peak)
- Total steps: ~143,000 (300B tokens ÷ 2M tokens/step)

**Batch size:** 2M tokens/step globally (micro-batch 4, gradient accumulation as needed).

### 4.2 The Training Loop Fix

A critical early bug was discovered when the transformer's forward contract changed to return a plain tuple `(loss, logits, present_key_values)` while the trainer still expected an object with `.loss`. Three concrete breaks were identified and fixed:

1. **Output unpacking:** `outputs.loss` → `loss, _, _ = self.model(...)`
2. **Batch format mismatch:** Dataset returned `(input_ids, labels)` tuples; trainer expected dicts → added `_collate_fn`
3. **No dataloader wiring:** Added `build_dataloaders(cfg)` function

### 4.3 Training Infrastructure Scale

The `src/training/` directory grew to **370+ files** covering:

**Optimizers:** Muon, AdamW, Adam Mini, Adan, LARS/LAMB, Lion, GrokFast, SAM, Sophia, Shampoo, Prodigy, Mechanic, SOAP, Mars, Nesterov Adan, Schedule-Free AdamW, Power-SGD, SignSGD, GaLore, FLoRA, VERA, SIFT, BAdam, DELLA, Cautious, 8-bit optimizers

**LoRA variants:** DoRA, AdaLoRA, QLoRA, LoftQ, LoRA+, prefix tuning (v2, v3), adapter tuning, PEFT config

**Curriculum learning:** AbsoluteZero, curriculum pacing, curriculum scheduling, curriculum sampling, curriculum transition, curriculum RL

**Distillation:** Offline KD, cot_distillation, context distillation, intermediate distillation, feature distillation, layer distillation, sequence KD, patient KD, knowledge distillation (v2, v3)

**Distributed:** Pipeline parallel, FSDP lite, FSDP wrapper, tensor parallel, optimizer state sharding, optimizer offload, elastic coordinator, sequence parallel

**Continual learning:** EWC, continual replay, experience replay, continual strategy, multi-task EWC, task arithmetic, task vector

**Misc:** Gradient checkpointing, activation offload, activation checkpointing, dynamic batch, throughput profiler, self-improvement loop, model merge (SLERP, TIES, DARE, Fisher, model soup, evolutionary merge, model breadcrumbs), quantization-aware training, pruning, structured pruning

### 4.4 Checkpointing Strategy

- Save every 10B tokens (~4,800 steps)
- Run lm-evaluation-harness at each checkpoint (MMLU, HellaSwag, GSM8K)
- Keep last 5 checkpoints + best validation-loss checkpoint
- Format: HuggingFace safetensors (legacy `.pt` with deprecation warning)
- Muon optimizer state is saved and loaded

---

## 5. Alignment — The Full Arc

The alignment system went through four distinct design generations, each superseding the last.

### 5.1 Original v1 Plan: SFT + DPO + Safety

The initial plan was a standard three-phase pipeline:

**Phase 1 — SFT:**
- Data: ~50K examples (OASST2 top-rated, Dolly-15k, filtered ShareGPT)
- Format: ChatML template
- Method: LoRA (r=64, alpha=128) on M1 Pro; full fine-tune on H100
- LR: 2e-5, cosine, 3 epochs
- Cost: ~$12–18

**Phase 2 — DPO:**
- Data: UltraFeedback binarized (256K pairs, filtered: score gap ≥ 1.0)
- β=0.1, LR=5e-7, 1 epoch
- Cost: ~$18

**Phase 3 — Safety Validation:**
- Garak (automated red-teaming)
- Promptfoo (structured attack scenarios)
- Target: <5% attack success rate per category

**Expected benchmarks at 1.3B:**
- MMLU: 42–48%
- HellaSwag: 65–72%
- ARC-Challenge: 45–55%
- TruthfulQA: 35–45%
- GSM8K: 20–30%
- HumanEval: 25–35%

### 5.2 PRAXIS: Architecture-Aware Alignment

**PRAXIS** (Policy Refinement through Aligned eXpert Integration System) was the first major research contribution — a unified alignment trainer with three novel architecture-aware signals.

**The 6-Signal Reward Bundle:**

| Signal | Module | What It Measures |
|--------|--------|------------------|
| R_prime | `src/alignment/prime.py` | Dense implicit process rewards (log π/π_ref per token) |
| R_const | `src/alignment/constitutional_ai_v3.py` | Per-principle harmlessness from hidden states |
| R_ccot | Computed | Chain-of-thought quality (outcome × log(1 + len/T)) |
| R_odin | Computed | Length-normalized outcome reward |
| R_hier | `src/alignment/hierarchical_reward.py` | Hierarchical multi-objective aggregation |
| R_src | SteeringRewardCorrespondence | Steering-reward correspondence (architectural) |

**PrecisionFusion:** Bayesian inverse-variance weighting over all 6 signals:
```
precision_i = 1 / (σ_i² + ε)
w_i = precision_i / Σ_j precision_j
R_fused = Σ_i w_i · μ_i
```

High-variance signals contribute less. Low-variance signals that are confident dominate.

**Three Novel Contributions:**

**SRC (Steering-Reward Correspondence):** Captures hidden states with and without an additive constitutional steering vector at designated layers (12, 16, 20). Computes cosine distance between steered and unsteered representations. A model already aligned needs minimal steering (distance → 0). Returns `−λ_src · mean_layers(cosine_dist)`. Implemented via forward hooks.

**ESA (Expert Safety Affinity):** For MoE layers, calls `layer.ffn.router.gate(h)` directly to get routing logits, then applies cross-entropy toward a uniform target over designated safety experts (indices 0, 1). Pushes constitutionally-unsafe tokens toward safety experts, encoding safety-relevant computation into a specific expert subset.

**MTAH (Multi-Token Alignment Horizon):** Extends per-token advantages with discounted future credit:
```
Ā_t = Σ_{k=0}^{K} γ^k · ā_{t+k}
```
Prevents myopic alignment where a token is rewarded only for its immediate contribution, ignoring misaligned continuations. `K=2, γ=0.95`.

**PRAXIS Loss:**
```
L_PRAXIS = L_DAPO(DAPO asymmetric clip on fused advantages)
         + β_kl · KL(π_θ ‖ π_ref)
         + L_ESA (routing auxiliary loss)
         + constitutional gate (zero gradient for R_const < τ_gate)
```

The constitutional gate zeroes the policy gradient for sequences below `τ_gate=0.4` — unsafe completions receive no alignment signal regardless of their task reward.

**PRAXIS was fully implemented and all tests pass** (`src/alignment/praxis/` — 8 components: config, precision_fusion, steering_reward, expert_safety_affinity, mtah, reward_signals, praxis_loss, trainer).

### 5.3 ARIA → AURORA → MOSAIC

This three-stage curriculum was the second generation, designed after PRAXIS proved the architecture-aware signals worked.

**ARIA (Adaptive Reward Interleaving Alignment) — Stage 1:**
- Sequential phases: Task Performance (Dr.GRPO) → Constitutional Correction (CAI v3) → Dense Process Reward (PRIME)
- WARP merge every 5 cycles (SLERP + SFT anchor)
- Purpose: validate each component independently before combining

**AURORA (Adaptive Uncertainty-Oriented Resonance Alignment) — Stage 2:**

The core innovation: at every step, four reward signals run simultaneously and are fused via Bayesian precision weighting. Constitutional rewards then gate the gradient (not just filter outputs).

Four simultaneous signals:
- R_outcome: verifiable task score
- R_prime: Σ_t log(π_θ/π_ref) [PRIME]
- R_const: CritiqueHead(hidden).aggregate()
- R_self: SelfRewardTrainer.score()

```
R_combined_i = Σ_j (1/σ²_j) · R_j_i / Σ_j (1/σ²_j)
A_i = R_combined_i − mean(R_combined)   [no std normalization — Dr.GRPO]

if R_const_i < const_threshold:
    A_i ← 0   [gradient zeroed for unsafe, not just output filtered]
```

DAPO asymmetric clip on the gated advantage:
```
r_i = exp(log π_θ(y_i|x) - log π_old(y_i|x))
clip to [1-ε_low, 1+ε_high] if A_i ≥ 0  (ε_low=0.10, ε_high=0.28)
clip to [1-ε_low, 1+ε_low]  if A_i < 0
```

Five claimed original contributions:
1. Precision-weighted reward fusion (Bayesian — no prior paper applies inverse-variance to multi-signal advantage)
2. Constitutional gradient gating (not output filtering)
3. PRIME as background dense signal (reuses existing log-prob computation)
4. DAPO asymmetric clip on multi-signal combined advantage
5. AbsoluteZero self-curriculum (data-free after initial SFT)

**MOSAIC (Multi-Objective Self-Aligned Iterative Curriculum) — Stage 3:**
- Fully self-contained loop: model proposes tasks, solves them, judges quality
- AbsoluteZero task proposal + DAPO filter + self-reward scoring + online DPO
- No human labels, no external reward model
- Hierarchical reward re-weighting based on Pareto objective feedback
- WARP merge every K steps toward AURORA checkpoint

### 5.4 APEX: The Grand Unified Framework

**APEX** (Adaptive Preference and EXpert alignment) — the final design, superseding PRAXIS and MOSAIC v2 before MOSAIC was implemented.

**Design doc:** `docs/plans/2026-05-20-apex-design.md`  
**Paper target:** NeurIPS 2027 / ICML 2027

**The Core Claim:**
> APEX is the first alignment framework that (1) automatically detects and routes all preference data modalities, (2) redistributes credit hierarchically across response → segment → token granularities, and (3) applies architecture-aware optimization using the model's internal structure. Every prior alignment method is a strict special case of APEX with one or more components ablated.

**Three-Tier Architecture:**

**Tier 1: Modality Router**

Automatically detects the data type in each batch and routes to the appropriate reward path:

| Modality | Path | Key Algorithm |
|----------|------|---------------|
| Preference pairs | ORPO (ref-free) / DPO (with ref) | Log-odds ratio |
| Scalar rewards | GRPO + REINFORCE++ + PRIME | Group normalization |
| Binary labels | KTO | Kahneman-Tversky prospect theory |
| No labels | SPIN | Self-play against frozen previous checkpoint |
| Principles | Constitutional AI | CritiqueHead per principle |
| Multiple signals | PrecisionFusion | Inverse-variance weighted merge |

**Tier 2: Credit Engine**

Takes sequence-level reward `R_i` and produces per-token advantage `A(t)` through five sequential refinements:

1. **Group normalization:** `Ā_i = (R_i − μ_G) / (σ_G + ε)` [GRPO/REINFORCE++ style]
2. **SAPO entropy-based segment decomposition:** Detects segment boundaries by token-level entropy percentiles; assigns quality scores `Q_s` via learned value head (PRIME mean fallback); redistributes advantage proportionally to segment quality
3. **AEM entropy gating:** `α(t) = uncertainty(t) · α_learned` — high-entropy uncertain tokens in high-advantage segments get more credit; prevents over-crediting boilerplate tokens
4. **TUR-DPO topology decomposition:** Per-token scores for faithfulness, utility, topology_quality via linear head on hidden states; learnable `principle_weights` vector jointly trained
5. **Phase weighting:** Thinking tokens (inside `<think>...</think>`) get 0.5×; answer tokens get 1.0×

**Combined token-level advantage:**
```
A(t) = Ã_{s(t)} · α(t) · w_topology(t) · phase_weight(t)
```

**Tier 3: Architecture-Aware Optimization**

- **MTAH:** `Ā(t) = Σ_{k=0}^{K} γ^k · A(t+k)` — temporal credit extension (K=2, γ=0.95)
- **SRC:** Forward hooks capture hidden states with/without constitutional steering vector; cosine distance penalty at designated layers
- **ESA:** Routes constitutionally-unsafe tokens to safety experts via direct router gate calls
- **Constitutional gradient gate:** `gate_i = float(R_const_i ≥ τ_gate)` — zeros `Ā_gated(t)` for unsafe sequences
- **WARP:** Periodic anchor merge toward SFT checkpoint every 50 steps: `α=1−warp_anchor_mu=0.05`

**Unified APEX Loss:**
```
L_APEX = −E_t[min(ρ·Ā_gated, clip(ρ)·Ā_gated)]   ← DAPO on Tier 2 credit
        − λ_ent · H(π_θ)                             ← entropy bonus
        + β_kl · KL(π_θ ‖ π_ref)                    ← KL (zero if ref-free path)
        + λ_src · mean_l(1 − cosine_sim(h_u, h_s))  ← SRC penalty
        + α_esa · mean_l(CE(router, safety_exp))     ← ESA routing
```

where `Ā_gated(t) = gate_i · MTAH_extended(Ã_{s(t)} · α(t) · w(t) · phase_w(t))`

**Prior Methods as Special Cases of APEX:**

| Method | What it ablates | Notes |
|--------|-----------------|-------|
| PPO | Tier 2, Tier 3, all paths except scalar | Classic RLHF baseline |
| DPO | Tier 2, Tier 3; preference pairs only | Response-level loss |
| GRPO | Tier 2, Tier 3; scalar only | Group normalization only |
| KTO | Tier 2, Tier 3; binary only | Prospect theory value |
| SPIN | Tier 2, Tier 3; self-play only | No external signal |
| Constitutional AI | Tier 2, Tier 3; principles only | CAI path only |
| SAPO | Tier 2 step 2 only; no AEM/TUR-DPO; no Tier 3 | Entropy segment decomp |
| AEM | Tier 2 step 3 only | Entropy token gating |
| TUR-DPO | Tier 2 step 4 only | Topology decomposition |
| PRAXIS | Tier 1 scalar only; no Tier 2; Tier 3 SRC+ESA+MTAH | APEX precursor |
| **APEX (full)** | None | All paths, all tiers |

**Curriculum (ARIA → AURORA → APEX):**
- Steps 0–1000: ARIA (CAI path only, uniform token credit, no arch-aware)
- Steps 1000–4000: AURORA (GRPO+CAI, SAPO only in Tier 2, SRC from step 1500, WARP from step 2000)
- Steps 4000+: APEX (all components; AEM+TUR-DPO at 4000, KTO at 4500, ESA at 5000, MTAH at 5500, λ_src ramp at 6000)

**Integration strategy (compose, do not rewrite):**
All 13 techniques are imported from existing modules. APEX is an orchestration layer, not a rewrite.

**File target:** `src/alignment/apex/` — 9 files: config, modality_router, credit_engine, arch_optimizer, apex_loss, curriculum, spin_generator, value_head, trainer.

---

## 6. Inference Engine

### 6.1 KV Cache — 8 Hot-Swappable Strategies

| Strategy | Module | Description |
|----------|--------|-------------|
| DuoAttention | `duo_attention.py` | Per-head retrieval/streaming classification; JSON config auto-export |
| EVICT (H2O) | `kv_cache_eviction.py` | Attention-score-based eviction |
| KIVI | `kivi_quant.py` | INT4/INT8 quantized cache with configurable residual length |
| QUEST | `quest_attention.py` | Query-aware sparse KV access |
| Rocket KV | `rocket_kv.py` | Importance-weighted budget allocation |
| SAGE Attention | `sage_attention.py` | SageAttention kernel integration |
| TEAL | `teal_sparsity.py` | Sparsity-based token eviction |
| INT8 Sim | `kv_cache_quantization.py` | Quantization noise simulation during fine-tuning |

### 6.2 TurboQuant KV Compression

A two-stage KV cache compression system implemented natively in `src/inference/turboquant/`:

**Stage 1 — PolarQuant (MSE-optimal):**
1. Random orthogonal rotation maps K/V into a Beta(2,2)-distributed space
2. Lloyd-Max quantization (codebook built at module import, never recomputed per-input)
3. Per-channel min-max normalization + quantization
4. Stores `x_mse` (the MSE-optimal reconstruction) and `residual = x − x_mse`

**Stage 2 — QJL (1-bit Johnson-Lindenstrauss sketch of residual):**
1. Fixed sign matrix `S ∈ {±1}^{sketch_dim × dim}` (int8 storage)
2. `signs = sign(residual @ S^T)` — 1-bit projection
3. `norms = ||residual||` — stores L2 norms separately
4. Inner product estimation: `signs · query_projection × norms × √(π/2) / sketch_dim`

**Attention score = MSE term + residual term:**
```
score(q, k) = q @ x_mse^T + QJL.estimate_inner_product(q, signs, norms)
```

Achieves ~2.5 bits/element with no calibration data. Per-layer distinct seeds ensure decorrelated compression.

### 6.3 Speculative Decoding Ecosystem

The inference module grew to cover the complete 2024-2025 speculative decoding literature:
- Medusa (multi-head draft)
- EAGLE / EAGLE2 / EAGLE3 (representation-based draft)
- Self-speculative decoding
- Tree-based speculation
- Draft tree v2
- Cascade speculative
- Hydra speculative
- REST retrieval-based speculative
- Multi-draft speculative
- Speculative rejection v2

### 6.4 Generation Methods

**Sampling:** Temperature, top-p (nucleus), top-k, min-p, diverse beam search, majority voting, self-consistency, universal self-consistency, ensemble, best-of-N, MBR decoding, entropix sampler

**Constrained:** Grammar-constrained, JSON mode, format enforcer, schema-constrained, structured output (v2)

**Watermarking:** Kirchenbauer watermark, neural watermark, LLM watermark, llm_watermark

**Advanced:** Chain-of-draft, skeleton-of-thought, lookahead decoding (v2), Jacobi decoding, trie decoding, token recycling, wait-token forcer, token budget (entropy early stop v2), compute-optimal test-time scaling, chain-of-thought (v2)

### 6.5 Local Serving

```
Format:     GGUF Q4_K_M (~800MB at 1.3B scale)
Conversion: llama.cpp convert_hf_to_gguf.py → llama-quantize
Speed:      ~25–35 tok/s on M1 Pro (Metal backend)
Tools:      Ollama, llama.cpp CLI, MLX-LM
```

### 6.6 Cloud Serving

```
Server:     SGLang (OpenAI-compatible)
API:        POST /v1/chat/completions (streaming SSE + non-streaming)
Weights:    AWQ 4-bit + Marlin kernel
KV cache:   TurboQuant 3-bit keys + 4-bit values (5× compression)
```

---

## 7. Harvest Cycles: Research Integration

Starting at Cycle 124 (2026-04-20), the project adopted a structured "harvest cycle" methodology for integrating published research papers:

**Methodology:**
1. Identify paper (arXiv ID)
2. Analyze gap against existing Aurelius modules
3. Extract only the novel algorithm (not the full paper)
4. Implement natively in PyTorch with 10–16 unit tests
5. Integrate test at tiny scale (n_layers=2, d_model=64, n_heads=4)
6. Port to `/tmp/harvest/<name>`, verify, then delete
7. Commit only on green full suite

**Hard constraints for every harvest:**
- Pure native PyTorch only (no transformers, einops, flash_attn, xformers, scipy, sklearn)
- New config keys default to feature OFF
- Integration test: construct from AureliusConfig, exercise runtime path, assert regression guard

### Cycles 124–127 (April 2026)

**Source papers:**
- Kimi K2.5 (arXiv:2602.02276) — agent + alignment
- GLM-5 (arXiv:2602.15763) — model + training + longcontext
- GPT-OSS-120B (arXiv:2508.10925) — model + chat + eval + training + agent

**Cycle 124 — Agent · Alignment · Optimizers:**
- PARL: `r_PARL = λ₁·r_parallel + λ₂·r_finish + r_perf` (parallel agent reward)
- Toggle: 25–30% token reduction via phase-based budget
- GRM: LLM-as-judge multi-dimensional scoring with hybrid rule-based + generative scoring
- Agent Swarm: Orchestrator + frozen subagents; critical-path scheduling
- MuonClip: Nesterov + per-head orthogonalization (Muon Split) + RL gradient clipping
- Cross-Stage Distillation: `L_CSD = L_RL + α·KL(π_θ ‖ π_teacher_k-1)`

**Cycle 125 — Model · Training · LongContext · Inference:**
- DSA Attention: Lightning Indexer learns top-k token selection; 2-stage dense warm-up → sparse adapt
- MTP Shared: 3 shared MTP heads (accept-rate 2.76 vs 2.55 baseline)
- Async RL Infra: Decoupled inference + training; Multi-Task Rollout Orchestrator; heartbeat fault tolerance
- TITO Gateway: Token-in-Token-out; eliminates re-tokenization mismatches
- Hierarchical Context Mgr: keep-recent-k → discard-all at 80% max_len
- Reasoning Level Controller: system prompt → {low/medium/high} → (temperature, max_tokens, top_p)

**Cycle 126 — Model · Chat · Eval · Training · Agent:**
- DP-aware MoE Routing: consistent hashing → fixed DP rank; prevents cross-rank KV cache misses
- MLA-256: head_dim 192→256, head_count ×0.67; Muon Split per-head orthogonalization
- Harmony Template: Jinja2 chat template with scratchpad delimiters, tool-call format
- Swarm Bench: Evaluates agent_swarm: critical-path steps, parallelism ratio, speedup vs single-agent
- Slime Framework: Unified RL infra — rollout server, fault tolerance, task router → verifier → reward_fn
- Plugin Hook Registry: `HOOK_REGISTRY: pre/post tool_call, pre/post generation, on_error`

**Cycle 127 — Vision / Multimodal:**
- MoonViT Patch Packer: NaViT patch packing; variable resolution; spatiotemporal volume (4 frames × H/16×W/16)
- Vision Projector: Linear projection from ViT hidden dim → LLM hidden dim; 4× temporal compression pooling
- Vision Token Mixer: Early-fusion — vision tokens mixed with text at 10% ratio throughout training
- Programmatic Image Tools: crop, detect_objects, pixel_distance, blob_count — proxy ops for Zero-Vision SFT
- Zero-Vision SFT Trainer: Text-only SFT activates visual reasoning via programmatic ops
- Vision Grounding Eval: F1 with soft IoU matching; normalized edit distance for OCR

**Cycles 199–200 (April 2026):**
- Cycle 199: `tgi_backend_adapter.py`, `context_quality_scorer.py`, `video_frame_sampler.py`, `query_intent_classifier.py`, `accessibility_announcer.py` — 157 new tests
- Cycle 200: `abductive_reasoner.py`, `analogy_engine.py`, `memory_retrieval_reranker.py`, `memory_fusion.py`

---

## 8. DAIES: AMC Memory & Scaling Ledger

The DAIES (Dynamic Adaptive Intelligence with Episodic States) project ran as a parallel architectural exploration, documenting the full scaling path from 125M to 32B parameters.

### 8.1 Scaling Path

| Tier | Parameters | Context Window | Key Config |
|------|------------|----------------|------------|
| Base | 125M | — | d_model=768 |
| 1B | 1.2B | — | d_model=2048 |
| 3B | 3.3B | — | d_model=3072 |
| 7B | ~8.6B | 16K | d_model=3584, 40 heads, 40 layers |
| 14B | ~22.6B | 16K | d_model=5120, 40 heads, 53 layers |
| 32B | ~66.6B | 32K | d_model=7168, 56 heads, 81 layers |

### 8.2 AMC Memory Architecture

The AMC (Aurelian Memory Core) is a multi-tier memory system integrated into the transformer stack:

**Memory components:**
- **SurpriseGate:** Modulates memory writes based on novelty/surprise
- **BiGRU:** Bidirectional recurrent memory integration
- **Long-Term Storage (LTS):** Paged LTS with async writes, adaptive precision (BF16/FP8), prefetch router
- **Graph Consolidator:** Episodic → semantic consolidation via graph clustering
- **ForgetGate:** Importance-based retention with decay

**Memory optimizations (Iteration 2):** Paged LTS, async memory, adaptive precision, prefetch router, deduplication, KV quantization, paged attention, gradient checkpointing, BF16, CPU offload, ZeRO, LZ4 compression, FlashAttention-3, CUDA graphs, Rust PyO3 bindings

### 8.3 Agent + Skills Layer (Iteration 3)

Integrated directly into the transformer stack at every N-th block:

- **ToolFormer:** Tool-call parsing and execution
- **MCTS Planning:** BrainBridge with CriticHead + uncertainty estimation
- **ValueHead:** Returns scalar value estimate for RL
- **SkillRegistry:** 8,192 skill capacity, dynamic acquisition
- **Agent Loop:** Budget-bounded ReAct loop with AST-walker arithmetic
- **Agent-Memory Bridge:** Episodic read/write across agent steps

### 8.4 BrainBridge (Iteration 5)

Top-level cognitive layer added to all scale variants:
- MCTS planner: `n_simulations=16-32`, `max_depth=6-8` by tier
- CriticHead: Per-step value estimation
- Uncertainty estimation: MC-Dropout on critic predictions
- AgentLoopController: Budget-bounded step limit
- AgentMemoryBridge: Cross-block episodic state

### 8.5 DAIES Benchmarks (5 custom benchmarks)

| Benchmark | Metric | What It Tests |
|-----------|--------|---------------|
| CrossSessionRecall | cosine_sim, recall_accuracy | Memory persistence across session boundaries |
| SurprisePrioritization | ROC-AUC | Novel vs. familiar input discrimination |
| RelationalGraph | ARI, cluster purity, NMI | Graph consolidator cluster preservation |
| ForgetGate | Precision/recall by importance | Importance-based memory retention/decay |
| LongRangeCoherence | motif_awareness, perplexity_variance | Long-range dependency and memory read patterns |

**Status at Iteration 5:** 49 tests passing (30 tier tests + 19 integration tests). All verification checks pass (7B/14B/32B forward pass, BrainBridge standalone, memory/agent/skill units, distributed imports, FSDP, inference).

---

## 9. Agent System

### 9.1 Core Agent Loops (`src/agent/`)

- **ReAct loop:** Tool-call parsing, argument validation, budget-bounded termination; AST-walker arithmetic (no dynamic code execution for math)
- **Plan-and-execute:** Two-phase: plan generation → step-by-step execution with re-planning
- **Budget-bounded loop:** Hard token/step budget with graceful degradation
- **AbsoluteZero:** Self-play curriculum — task proposer + solver in a closed feedback loop

### 9.2 Tool & MCP Infrastructure

- `tool_call_parser.py`: Parses model-generated tool calls (JSON schema + markdown patterns)
- `tool_registry_dispatcher.py`: Routes tool calls to registered handlers
- `mcp_client.py`: Model Context Protocol client
- `plugin_hook.py`: `HOOK_REGISTRY` — pre/post hooks for tool_call, generation, error events
- Approval state machine: explicit, scoped, recorded in thread timeline

### 9.3 Planning Engine

- Workstream DAG with `TaskStatus` / `PlanStatus` StrEnum
- `get_workstream(missing_ok)` guard — explicit failure when missing
- Task scheduler: cron / interval / delayed jobs persisted to `~/.cache/aurelius/jobs.json`

### 9.4 Persona System

13 personas across 5 domains:
- GENERAL, CODING, SECURITY, THREAT_INTEL, AGENT
- 7 composable facets per persona
- Reputation system: Bayesian multi-agent trust scoring with Sybil resistance

### 9.5 Neuro-Symbolic Integration

- LLM reasoning on symbolic rule engines
- Neuro-symbolic skill registration
- Abductive reasoner + analogy engine (Cycle 200)

---

## 10. API Gateway & Serving

### 10.1 FastAPI Gateway (`gateway/`)

Production-grade OpenAI-compatible API:

```
POST /v1/chat/completions  — streaming SSE + non-streaming
GET  /v1/models            — model listing
GET  /health               — liveness (engine_loaded flag)
GET  /health/ready         — readiness (503 until engine initialized)
GET  /healthz              — legacy alias
GET  /metrics              — Prometheus scrape
WebSocket /ws              — real-time streaming chat
```

### 10.2 Security Hardening

The gateway implements production security headers:
- Content-Security-Policy (production only)
- HSTS
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Referrer-Policy
- `X-Request-ID` tracing header for end-to-end correlation

**Rate limiting:** Per-IP token bucket; in-memory (single-node) or Redis (distributed). Configurable via `AURELIUS_RATE_LIMIT` and `AURELIUS_RATE_WINDOW`.

**Host allow-listing:** `AURELIUS_ALLOWED_HOSTS` environment variable.

**Request size limits:** 1 MiB JSON body, 10 MiB streaming.

### 10.3 Prometheus Metrics

Full golden-signals instrumentation:

| Metric | Type |
|--------|------|
| `aurelius_requests_total` | counter |
| `aurelius_requests_per_second` | gauge |
| `aurelius_active_connections` | gauge |
| `aurelius_request_duration_ms` | gauge (p50/p95/p99) |
| `aurelius_uptime_seconds` | gauge |
| `aurelius_http_status_total` | counter (by code) |
| `aurelius_rate_limit_rejected_total` | counter |
| `aurelius_validation_failures_total` | counter |
| `aurelius_rate_limiter_backend` | gauge (0=memory, 1=redis) |

### 10.4 Node.js BFF (`middle/`)

The frontend never talks directly to Python. All API calls route through the TypeScript BFF:
- Auth, JWT validation
- Rate limiting
- WebSocket management
- SSE proxy
- Cron job management
- File serving

---

## 11. Frontend & Full-Stack Architecture

### 11.1 Stack

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| Rust Engine | `crates/` | Rust 2024 | Tokenization, search, vector similarity, session management, data engine |
| Python Backend | `src/`, `agent/`, `gateway/` | Python 3.12+ | Model, training, inference, alignment, API, CLI |
| Node.js BFF | `middle/` | TypeScript | Auth, rate limiting, WebSocket, SSE, cron, file serving |
| Frontend | `frontend/` | React 19 + TypeScript | Mission Control: dashboard, chat, analytics, admin |

**Data flow:** Browser → Node.js BFF (port 3001) → Python API (port 8080) → AureliusTransformer + Rust NAPI

### 11.2 Rust Components (`crates/` + `rust_memory/`)

- Tokenization via Rust NAPI (PyO3 bindings)
- Vector similarity search
- Session management
- Data engine (memory-mapped shard handling)
- `rust_memory/src/lib.rs`: Rust memory backend with Python bridge in `rust_bridge.py`

### 11.3 Resilience Layer (`src/resilience/`)

Production fault-tolerance primitives:

| Pattern | Description |
|---------|-------------|
| CircuitBreaker | CLOSED / OPEN / HALF_OPEN FSM; configurable failure threshold + recovery |
| Bulkhead | Semaphore-based concurrency cap; isolates subsystem failures |
| RetryPolicy | Exponential backoff with jitter, configurable max attempts |
| RateLimiter | Token bucket; in-memory or Redis backends |
| Pipeline | Composable chain: circuit breaker → bulkhead → retry |

---

## 12. Canonical Interface Contract

Defined in `docs/plans/2026-04-21-aurelius-canonical-interface-contract.md`, this is the behavioral specification for all host surfaces.

### 12.1 Core Nouns

**TaskThread:** Top-level unit of work — thread_id, mode, status, instruction_stack, skills, approvals, checkpoints, steps

**Mode:** Policy preset — `ask`, `code`, `debug`, `architect`, `review`, `background`, `chat`

**InstructionLayer:** Ordered precedence — system policy → user prompt → repo instructions → workspace instructions → mode instructions → skill instructions → thread memory

**Skill:** Reusable capability bundle — `skill_id`, `name`, `scope` (global/org/repo/thread), `instructions`, `scripts`, `resources`, `entrypoints`, `version`, `provenance`

**Approval:** Stateful decision — file write/delete, shell command, network request, browser navigation, MCP tool call, external process launch, model/checkpoint mutation

**Checkpoint:** Durable resumable snapshot

**Subagent:** Nested task thread for bounded subtasks

**BackgroundJob:** Detached thread — status polling, result retrieval, cancellation, resume

### 12.2 Ten Contract Principles

1. One task = one thread
2. One thread = one stable id, one timeline, one checkpoint history
3. No silent fallbacks — unsupported operations fail loudly
4. Human approval is explicit and stateful
5. Skills are installable capability bundles, not ad hoc prompt blobs
6. Modes are policy presets, not model-config booleans
7. Tool use is auditable and replayable
8. Background work must be resumable or cancelable
9. Host adapters may differ, but core thread semantics must not
10. Model family identity, tokenizer contract, and release-track policy remain separate from user-facing modes

### 12.3 Related Ecosystem

The contract design draws from and extends:
- **clawhub:** Public skill registry with moderation hooks and vector search
- **acpx:** Headless ACP client with persistent sessions and cooperative cancel
- **lobster:** Typed local-first workflow shell with JSON pipelines and approval gates
- **ironclaw:** Rust reimplementation with WASM sandboxes, persistent memory, routines

---

## 13. Security & Safety Audit

### 13.1 May 2026 Audit — 11 Critical, 8 High Remediations

**Critical fixes:**
- **Sandbox escape via `object.__subclasses__()`** blocked in `sandbox_executor.py`, `code_execution.py`, `code_eval.py`
- **SSRF:** Private/reserved IP blocklist added to `http_backend.py`; URL validation moved before `Request()` construction
- **Auth middleware:** Default changed from opt-in to `require_auth=True` (fail-closed)
- **Shell tool hardening:** `shell=True` + denylist → `shell=False` + `shlex.split()` + explicit allow-list
- **PPO trainer:** `prompt_ids` NameError fixed; off-by-one in logit gather corrected
- **Constitutional AI:** KL divergence argument order corrected (alignment signal was being silenced by reversed arguments)
- **Plugin sandbox:** Exception catch → fail-closed `SandboxResult(success=False)`
- **CI gates:** `continue-on-error: true` removed from all security scan steps

### 13.2 Security Cycle 139-sec (April 2026)

**AUR-SEC-2026-0001 through 0027:**
- `weights_only=True` on all `torch.load()` calls
- Path-traversal hardening in `FileConversationStore`
- ReDoS-bounded regexes
- Canary pipeline, safe archive extractor, HMAC auth middleware

**Pre-commit hook:** Forbids `weights_only=False` outside tests. CI step runs pre-commit checks automatically.

### 13.3 Ongoing Safety Infrastructure

- Topology safety: persistent-homology invariants on hidden-state geometry
- Superposition geometry: polysemanticity detection (sparse autoencoder)
- 24 adversarial defense modules
- Jailbreak detector
- PII scanner
- Harm taxonomy: 9 categories
- Constitutional committee: multiple constitutional principles evaluated per completion

---

## 14. Evaluation & Benchmarking

### 14.1 Standard Benchmarks (v1 Targets)

| Benchmark | Method | Expected at 1.3B |
|-----------|--------|------------------|
| MMLU | 5-shot | 42–48% |
| HellaSwag | 10-shot | 65–72% |
| ARC-Challenge | 25-shot | 45–55% |
| TruthfulQA | 0-shot | 35–45% |
| GSM8K | 8-shot | 20–30% |
| HumanEval | 0-shot | 25–35% |
| MBPP | 3-shot | 25–35% |

### 14.2 APEX Paper Evaluation (Planned)

| Benchmark | What It Tests |
|-----------|---------------|
| AlpacaEval 2.0 | LC win rate vs. GPT-4-turbo |
| MT-Bench | Multi-turn instruction following |
| TruthfulQA | Factual accuracy |
| HarmBench | Safety (ASR metric) |
| Constitutional eval | Custom 8-principle rubric via CritiqueHead |

### 14.3 Evaluation Infrastructure (`src/eval/`)

The eval module contains **180+ files** covering the complete 2024-2025 evaluation literature:

**Benchmarks:** MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval, MBPP, HumanEval+, GPQA, LiveCodeBench, SWEBench Lite, TAUBench, WebarEna, OSWorld, ArenaHard, AlpacaEval, MT-Bench, RULER (long-context)

**Generation metrics:** ROUGE-L, BLEU, BERTScore, METEOR, EM, F1, semantic similarity

**Code evaluation:** Sandboxed execution, pass@k, HumanEval runner

**Safety evaluation:** HarmBench, jailbreak probes (tree-of-attacks, many-shot, crescendo), red-team benchmark, behavioral audit taxonomy

**Interpretability:** Causal tracing (activation patching + ROME rank-1 editing), linear probes, attention pattern visualization, attention rollout, gradient attribution, integrated gradients, logit lens, tuned lens, sparse autoencoder, sparse circuits, representation similarity

**LLM-as-judge:** G-Eval, pairwise eval, judge consistency, judge diversity, judge margin, verdict stability, ELO rating, shade arena, VIBE reviewer

**Reward evaluation:** Reward correlation, reward rank correlation, reward ensemble metrics, verifier metrics

### 14.4 DAIES Benchmarks (Custom)

5 proprietary benchmarks for memory system evaluation (CrossSessionRecall, SurprisePrioritization, RelationalGraph, ForgetGate, LongRangeCoherence) — all passing at Iteration 5.

---

## 15. Infrastructure & DevOps

### 15.1 Docker & Container

- Multi-stage `Dockerfile` (non-root user, pinned base-image digests)
- `docker-compose.yml` with GPU support
- `.dockerignore` for clean builds

### 15.2 CI/CD (GitHub Actions)

Full CI matrix:
- Python 3.12/3.13 matrix
- Full pytest run
- Ruff lint + format
- Bandit security scan
- pip-audit
- Pre-commit checks (including `weights_only=False` forbiddance)

**Trigger branches:** `cycle/*`, `sec/*`, `feat/*`, `deploy/*`

### 15.3 Kubernetes (`k8s/`)

- Readiness/liveness probes wired to `/health/ready` and `/health`
- Container resource limits

### 15.4 CLI (`aurelius_cli/`)

36 CLI modules covering:
- `aurelius pipeline` — fluent ETL data transformation (filter/map/sort/head/tail/dedup)
- `aurelius schedule cron|interval|once` — job scheduling
- `aurelius metrics demo` — synthetic SRE workload reporting
- `aurelius chat` — interactive chat session

### 15.5 Observability (`src/observability/`)

| Module | Purpose |
|--------|---------|
| `AgentTelemetry` | High-level facade: audit + metrics + tracing in one call |
| `AuditLogger` | Structured audit trail with retention |
| `EventBus` | In-process async event routing |
| `MetricsCollector` | Counters, histograms, gauges with labels |
| `TraceContext` | W3C-compatible distributed trace propagation |

---

## 16. Research Paper Pipeline

Three papers are planned or in design:

### 16.1 APEX Paper (Primary — NeurIPS/ICML 2027 Target)

**Title:** "APEX: Unified Alignment via Adaptive Preference Routing and Hierarchical Credit Assignment"

**Falsifiable claim:** Every prior alignment method is a special case of APEX. Each row of the ablation table corresponds to a reproducible experiment on Aurelius 1.395B.

**Paper outline:**
- §1 Introduction — the fragmentation problem in RLHF
- §2 Background — RLHF, DPO, GRPO, PPO, SAPO, TUR-DPO, AEM, ORPO, SimPO, CPO, KTO, CAI, SPIN, PRAXIS
- §3 Method — Modality Router (Tier 1), Credit Engine (Tier 2), Architecture-Aware Optimization (Tier 3), unified loss, curriculum
- §4 Experimental Setup — Aurelius 1.395B, datasets, evaluation, $350 cloud budget
- §5 Results — main table vs. all prior methods, ablation table, curriculum learning curves
- §6 Analysis — credit visualization heatmaps, ESA expert activation patterns, SRC steering correlation
- §7 Discussion — when credit engine matters, reference-free tradeoffs, limitations
- §8 Conclusion

**Compute budget:** ARIA (1000 steps × 8 completions = 8K passes), AURORA (3000 steps = 24K passes), APEX (6000+ steps = 48K passes). Estimated $200–250 of $350 budget; remainder for ablations.

### 16.2 AMC Paper (Follow-on)

**Focus:** Aurelian Memory Core — the multi-tier episodic + semantic + graph-based memory architecture and its 5 DAIES benchmarks.

### 16.3 Paper Ideas (Documented in `docs/`)

Four paper idea documents were produced:
- `paper_ideas_alignment.md` — 5 papers analyzed (DPO, CAI, TruthfulQA, RARR, GPT-4 System Card) with concrete implementation ideas
- `paper_ideas_efficiency.md` — efficiency and inference optimization ideas
- `paper_ideas_memory_moe.md` — memory + MoE architecture ideas
- `paper_ideas_reasoning.md` — reasoning and chain-of-thought ideas

---

## 17. Chronological Milestone Log

| Date | Milestone |
|------|-----------|
| 2026-04-06 | v1 design document approved — 1.3B dense decoder architecture finalized |
| 2026-04-07 | Heavens Gate integration designed — 4 module groups: model core, data, alignment, inference |
| 2026-04-07 | Training loop fix designed — 4 concrete breaks identified and patched |
| 2026-04-08 | Learned optimizer fix handoff — first autonomous agent handoff |
| 2026-04-18 | Cycle 139-sec — security gate: closed AUR-SEC-2026-0001 through 0027 |
| 2026-04-20 | Harvest Cycles 124–127 designed — Kimi K2.5, GLM-5, GPT-OSS-120B integration |
| 2026-04-21 | Canonical Interface Contract published — 10 principles, 9 canonical nouns |
| 2026-04-24 | Cycle 199 — 157 new tests |
| 2026-04-25 | Cycle 200 — abductive reasoner, analogy engine, memory retrieval reranker |
| 2026-04-27 | Autonomous improvement cycles logged (`.aurelius-cycles.log`) |
| 2026-05-02 | Docker, Cargo.toml, Rust toolchain established |
| 2026-05-03 | Cycle 140-sec — KV-cache eviction shape fix, GradScaler hardening, pre-commit hook |
| 2026-05-09 | AURORA + MOSAIC design — 3-stage alignment curriculum first drafted |
| 2026-05-09 | PRAXIS implementation plan finalized — 9 tasks, full TDD, ready for agent execution |
| 2026-05-11 | Gateway, agent, CLI systems finalized; 775+ tests passing |
| 2026-05-12 | DAIES Iteration 5 complete — 7B/14B/32B models, BrainBridge, 49 tests |
| 2026-05-13 | Hardening: security headers, rate limiting, SSRF blocklist, production readiness |
| 2026-05-18 | Code review + rescan — combined review document |
| 2026-05-19 | Gateway, deployment, frontend finalization |
| 2026-05-20 | APEX design document approved — grand-unified alignment superseding PRAXIS + MOSAIC v2 |
| 2026-05-22 | This retrospective generated |

---

## 18. Summary Statistics

| Metric | Count |
|--------|-------|
| Total Python source files (excl. venv) | 4,933 |
| Source files in `src/` | 2,125 |
| Test files in `tests/` | 2,379 |
| Design documents (plans) | 19 |
| Alignment modules | 160+ |
| Model architecture variants | 200+ |
| Inference strategies | 80+ |
| Training/optimization methods | 370+ |
| Evaluation modules | 180+ |
| Harvest cycles completed | 200+ |
| Cloud budget target | ~$350 |
| Test status (last recorded) | 775 passing, 2 skipped |
| Architecture-aware alignment signals (PRAXIS) | 6 signals |
| Tier 2 credit engine steps (APEX) | 5 steps |
| Prior methods as APEX special cases | 14 |
| DAIES scaling tiers | 6 (125M → 32B) |
| Custom DAIES benchmarks | 5 |
| Security issues remediated | 27+ (AUR-SEC-2026-0001 through 0027+) |
| Languages in The Stack v2 (training data) | 619 |
| Paper targets | 3 (APEX, AMC, TBD) |
| Research paper target conferences | NeurIPS 2027 / ICML 2027 |

---

## Key Technical Decisions — A Synthesis

**1. Plain tuple forward contract**
The decision to return `(loss, logits, present_key_values)` as a plain tuple (not a HuggingFace `ModelOutput` object) was the single most consequential early decision. It required an explicit training loop fix and an explicit unpacking convention throughout every module. The benefit: no implicit dependency on HuggingFace semantics.

**2. Native PyTorch for everything**
No flash-attn, no bitsandbytes, no HuggingFace Transformers. This means the codebase is fully readable, fully debuggable, and fully portable. The cost is more code and more testing. The benefit is that every algorithm lives in the repository and is testable with random weights on CPU.

**3. Architecture-aware alignment as a research contribution**
The insight that alignment methods should exploit the model's internal structure (which layers carry safety, which experts handle unsafe content, what the temporal credit horizon should be) is the core research contribution threading through PRAXIS → AURORA → APEX. No published paper does all three.

**4. Hierarchical credit as the unifying principle**
The Credit Engine in APEX is the observation that all prior methods assign credit at a single granularity (response, segment, or token) using a single reward type. Composing SAPO + AEM + TUR-DPO into a single hierarchical A(t) is the key technical novelty.

**5. Bayesian signal fusion throughout**
PrecisionFusion (inverse-variance weighting) appears in PRAXIS, AURORA, and APEX as the canonical way to merge reward signals of different reliability. The explicit uncertainty estimation via MC-Dropout means the training signal is inherently noise-aware.

**6. Constitutional gradient gating vs. output filtering**
The constitutional gate zeros the gradient (not the output) for unsafe completions. This is the difference between blocking unsafe text at serving time (existing practice) and teaching the model at training time that task performance does not justify unsafe outputs. The APEX claim is that this is more sample-efficient and more robust.

**7. Self-improvement as a first-class design goal**
MOSAIC and AbsoluteZero are the data-free endpoint — after initial SFT, the model can improve indefinitely without human labels. This was a stated goal from Day 1 ("fully autonomous, continuously self-improving personal LLM") and is achieved via the SPIN + online DPO + AbsoluteZero curriculum in MOSAIC.

---

*End of Aurelius Project Retrospective — 2026-05-22*
