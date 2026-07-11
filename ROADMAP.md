# Aurelius — Project Roadmap

> Complete history from inception through current state. Evidence-gated, receipt-backed.
> Every claim references a measured run, a falsifier gate, or an explicit "designed, not yet built" flag.
> Last updated: 2026-07-11

---

## Timeline Overview

```
v0 (2024-2025)     Infrastructure & Learning
v1 (2025-Q1)       1.395B From-Scratch Backbone — smoke checkpoint
v2 (2025-Q2→2026)  Qwen3 Testbed — shipped optimum
v3 (2026-04→07)    Capability Measurement Arc — receipts that v2 IS the optimum
v4 (2026-07→now)   CUA + Agentic + Verifier System — ACTIVE
v5 (gated)         From-Scratch Continued Pretraining — gated behind v4
```

---

## v0 — Pre-Aurelius: Infrastructure & Learning (2024–2025)

**What was built:**
- `ps-it-toolkit` — PowerShell IT automation toolkit (public, S3nna13)
- `windows-admin-reference` — cross-platform admin command reference (public)
- Homelab: Dell OptiPlex, MSI laptop, M1 Pro MBP 32GB
- Deep study of the transformer architecture (Vaswani et al. 2017 through GPT-4)
- Karpathy GPT-from-scratch curriculum completed
- 220+ cross-platform command reference

**What was learned:**
- Full-stack engineering from PowerShell automation through Python ML
- Enterprise IT infrastructure (Intune, Entra ID, Graph PowerShell, Autopilot)
- The transformer architecture end-to-end: attention, training dynamics, scaling laws

**Artifacts:** public GitHub repos (S3nna13), homelab running Qwen3 MoE via MLX + LM Studio

---

## v1 — From-Scratch Backbone: Aurelius-1.3B (2025-Q1)

**Goal:** Build an LLM from absolute scratch — no HuggingFace Transformers, no flash-attn, no bitsandbytes.

**Architecture:**
| Parameter | Value |
|-----------|-------|
| Type | Decoder-only causal LM |
| Parameters | 1.395B |
| Layers | 24 transformer blocks |
| Hidden dim | 2,048 |
| Attention | Grouped-Query Attention (16 Q heads, 8 KV heads) |
| Head dim | 128 |
| FFN | SwiGLU, d_ff = 5,632 |
| Normalization | Pre-norm RMSNorm |
| Positional encoding | RoPE (θ = 500,000) + YaRN context extension |
| Vocabulary | 50,257 tokens |
| Embeddings | Tied input/output |
| Optimizer | Muon (Newton-Schulz 8+2 steps + Nesterov + RMS rescaling) |

**What was built:**
- Full transformer core in pure PyTorch (`src/model/`) — 200+ modules
- Rust data engine (`crates/`) — tokenization, search, vector similarity
- Node.js BFF (`middle/`) — auth, rate limiting, WebSocket
- React 19 frontend — Mission Control dashboard
- Training pipeline: Muon optimizer, ZClip gradient clipping, BAdam, curriculum
- 8 hot-swappable KV cache strategies (KIVI, DuoAttention, EVICT, QUEST, Rocket KV, SAGE, TEAL, INT8)
- PRAXIS/MOSAIC v2 alignment — 6-signal architecture-aware alignment
- 13 personas across 5 domains, 7 composable facets
- Composer agent — repo-level coding with diff engine, checkpoint rollback
- TruthSurface / claims-ledger — measurement backbone
- Tapered FFN — cosine-scheduled per-layer width (Bayat et al. 2026)
- Full production observability + resilience stack

**Designed (not yet built):**
- **RSVA** — Recursive Self-Verifying Architecture: generation ↔ verification fixed-point loop
- **HMC** — Holographic Mechanism Compression: mechanisms stored as interference patterns
- **FEPG** — Ungameable verifier to replace GRPO-style reward hacking

**Training state:** Smoke checkpoint only (`step-0000002`, ~512 tokens). Full pretraining pending — compute-bound.

**Key decision:** The 1.3B backbone architecture was correct, but pretraining from scratch at this scale was ruled out by cost analysis. The project pivoted to using open-weight models as testbeds while the native backbone remains a gated v5 option.

---

## v2 — Qwen3 Testbed: The Shipped Optimum (2025-Q2 → 2026)

**Goal:** Validate the alignment/eval/verifier tooling on real trained models while the native backbone path was compute-gated. Build the verifier-native inference system.

**Model family (6 private repos on HF/Zephyrs33):**
| Repo | Base | Type | Status |
|------|------|------|--------|
| `aurelius-14b` | Qwen3-14B | Full finetune | Complete |
| `aurelius-14b-sft` | Qwen3-14B | SFT LoRA (14dl) | Complete |
| `aurelius-14b-dpo` | Qwen3-14B | DPO LoRA | Complete |
| `aurelius-14b-fly` | Qwen3-14B | LoRA | Complete |
| `Aurelius-Qwen3-8B-RLVR` | Qwen3-8B | GRPO/RLVR LoRA, verification-native | Complete |
| `aurelius-v2` | Qwen3-Coder-30B-A3B | GGUF, verifier-native | **DEPLOYED** |

**What shipped in v2:**
- **Best-of-N / maj@N inference** — the one lever that consistently moved the needle
- **Verifier-native inference** — model self-verification as a selection mechanism
- **AMC (Agentic Memory Compression)** — hierarchical memory system
- **MOSAIC v2 alignment** — PRAXIS loss (DAPO + KL penalty + constitutional gate)
- **Full alignment suite** — REINFORCE++, SAPO, TUR-DPO, AEM, DPO, GRPO, CPO, ORPO, PPO, SimPO, SPIN, KTO, constitutional AI
- GGUF Q4_K_M export for Apple Silicon (25-35 tok/s)

**v2 capability numbers (measured, 2026-07-11):**
| Benchmark | Greedy | maj@8 | Oracle | Selection Gap |
|-----------|--------|-------|--------|---------------|
| MATH-L5 | 65.7 | 73.9 | 81.3 | ~7.4pp |
| HumanEval | — | — | 91.5 (30B-A3B) | — |
| MBPP | — | — | 72.0 (30B-A3B) | — |

**Key findings:**
- v2 deliverable (30B-A3B + best-of-N/maj@N + repair) would prove to be the optimum at this compute class
- The verifier/selection system was the cashable lever, not architecture
- Zero-shot LLM self-verification was NULL (over-approval + abstention, not discrimination failure)

---

## v3 — Capability Measurement Arc (2026-04 → 2026-07)

**Goal:** Measure EVERY lever systematically. Find a step-change improvement over v2, or prove v2 IS the optimum given current compute.

### Capacity Ladder (measured)

| Model | Params | HE | MBPP | Verdict |
|-------|--------|-----|------|---------|
| Qwen3-8B | 8B | 84.1 | — | Baseline |
| Qwen3-14B | 14B | 88.4 | — | Marginal |
| **Qwen3-Coder-30B-A3B** | **3.3B active / 30B total** | **91.5** | **72.0** | **DEPLOY BASE** |
| Seed-OSS-36B | 36B dense | 79.9* | 76.0 | Uneconomic (~10× inference cost) |

*Seed-OSS-36B HE internally contradictory (think-truncation/template suspects); MBPP suggestive-best but ~10× inference cost for at-best-tie quality. Dense larger = uneconomic at this compute class.

> **Verdict: Capacity exhausted at Colab-class compute.** 30B-A3B MoE is the ceiling.

### Training-Side Levers (all measured)

| Lever | Method | Result | Verdict |
|-------|--------|--------|---------|
| SFT | Self-distill on 30B | +0.6pp | Polish only, not a lift |
| SFT | GLM-5.2 teacher traces | Pending run | Possible lift attempt |
| RLVR fold-in | LoRA GRPO on 30B (4-cycle) | FLAT | No gain from LoRA-dose RLVR |
| RLVR fold-in | LoRA GRPO on 30B (8-cycle) | FLAT (definitive) | Dose-extension flat |
| RLVR fold-in | Qwen3-4B code (8-cycle) | FLAT | Cross-model confirmation |
| OPD | On-policy distill | +1.2pp n.s. | First positive sign, but not significant |
| OPD | 4× dose extension (600 steps) | FLAT | Noise-positive confirmed |

> **Verdict: Training side closed 3 ways — SFT (polish only), RLVR-fold-in (flat), OPD (flat).**
> The safety receipt: 8 cycles of training on own verifier-selected outputs = ZERO regression + ZERO selection collapse. The loop is safe scaffolding.

### Inference-Side Levers (all measured)

| Lever | Result |
|--------|--------|
| Zero-shot LLM verifier | NULL — over-approval + abstention, no net signal beyond majority |
| Aggregation rules sweep (12 rules) | Ceiling ≈ maj; best rule veto+greedy = +2 (McNemar p≈0.63, not significant) |
| **Best-of-N / maj@N** | **THE one cashable lever** — inference-side selection from v2 shipping |
| Learned math verifier (probe) | NULL — verifier@8 61.0 loses to maj@8 70.7 held-out |
| Verifier miss analysis | Mechanism = low-precision over-approval, not broad discrimination failure |

> **Verdict: Inference-side verification = the one cashable lever, but cheap verifiers can't close the ~7.4pp MATH selection gap.** Only heavier trained options remain (GenRM / PURE-min-form PRM / RLVR-rerank) — gated behind capacity compute.

### Architecture Levers (ruled out)

| Lever | Result |
|-------|--------|
| Tapered FFN | NEUTRAL at proxy scale (Δ+0.26% within noise, does NOT reproduce paper's win) — PARKED |
| Per-token routing genus | FALSIFIED 3× — only from-scratch-revivable, prior LOW |
| RSVA / HMC / FEPG | Assessed vacuous — self-coined scaffolding, zero arXiv grounding |

### v3 Conclusion (measured, 2026-07-11)

> **The v2 deliverable (30B-A3B + best-of-N/maj@N) was already the optimum.**
> The v3 arc = the receipts proving it.
>
> v3 closes as a full negative-result program with one deploy upgrade (30B base receipts)
> and hardened experimental infrastructure as its durable products.

**Durable products from v3:**
1. Capacity ladder PROVEN: 30B-A3B is the ceiling at Colab-class compute
2. Training-side NULLS PROVEN: SFT/RLVR/OPD flat at LoRA dose
3. Inference-side ceiling PROVEN: aggregation ≈ maj, cheap verifiers don't close the gap
4. Experimental infrastructure: Drive-resumable notebooks, paired-flip verdicts, computed READs, dose-tagged artifacts
5. Safety receipts: RLVR loop is safe scaffolding, ZERO regression risk confirmed

**Remaining open levers (gated):**
| ID | Lever | Status |
|----|-------|--------|
| G1 | Trained math verifier (GenRM / PURE-min-form PRM / RLVR-rerank) | Gated — needs compute |
| G2 | Full-param RLVR on 30B-A3B via ZeRO-offload | Gated — needs A100/H100 |
| G3 | FTPO (spec not yet located) | Gated — undefined |

---

## v4 — CUA + Agentic + Verifier System (2026-07 → ACTIVE)

**Thesis:** The verifier/RLVR/best-of-N thesis GENERALIZES — OS-state diff IS a ground-truth verifier the same way unit-tests are, so the whole system ports to computer-use without a closed teacher.

### Active Workstreams

| ID | Workstream | Status |
|----|-----------|--------|
| **R1** | Alignment EvalLab release gates | Built (14 tests pass, CLI end-to-end) |
| **R2** | QAT before GGUF packaging | Gate set — Kimi INT4-QAT = existence proof the ~3-4pp 4-bit tax is removable |
| **R3** | Flight-recorder-lite in model cards | Part of release gate |
| **G1** | Trained math verifier (GenRM nb built) | Gated — needs compute, QVal-preflight-gated |
| **A1** | CUA-execution-verified RLVR | OS-state diff = ground-truth verifier |
| **A2** | AAR (autonomous auto-research loop) | Design phase |
| **A3** | MSGA (Multi-Scale Governed Agency) | Design phase |
| **A4** | ADAC/AWRF reason-rich alignment data | Generator built (firewall-gated) |

### v4.x Release Track (highest-EV near-term)

| ID | Work | EV Rationale | Cost |
|----|------|-------------|------|
| R1 | Alignment EvalLab release gate run | Blocks public-visibility flip on FAIL | Low |
| R2 | QAT-before-GGUF | Directly upgrades v2 GGUF quality (~3-4pp tax removal) | Low-Med |
| G1 | Trained math verifier | Targets the ~7.4pp measured gap; falsifier-gated | Medium |

### v4 Capability Lanes

| Lane | Description | Status |
|------|-------------|--------|
| **A** | Secure execution + per-test progress selector | HIGHEST EV — generalizes verifier to CUA domain |
| **B** | Frontier map (research landscape) | Runs first, informs Lane A |
| **C** | $750 full-param 8B GRPO | Gated — only if fresh benchmarks justify |

### RLVR Improvements Available (2026 literature)

Three mechanisms ready to integrate for better RL training (see `docs/RLVR_IMPROVEMENTS.md`):

| Mechanism | Paper | Benefit | Implementation |
|-----------|-------|---------|----------------|
| **BV-Blend** | 2606.28707 | Fixes zero-variance group stalling | EMA stats per cluster, blend into advantage (200 lines) |
| **RSI-S** | 2606.31575 | +2-3pp on AIME/AMC, filters noisy tokens | Token-level filtering in GRPO forward (100 lines) |
| **Layer-aware RL** | 2607.01232 | 4-5x VRAM reduction, 85% of full-param gains | Train middle layers only (150 lines) |

Drop-in improvements ready for Lane C or any RLVR work.

---

## v5 — From-Scratch Continued Pretraining (GATED)

**Gate conditions:**
1. v4 CUA + verifier system proven AND exhausted
2. Capacity lever fully spent
3. Fresh benchmarks justify the compute

**Plan:**
- Continued pretraining on 30B-A3B (Composer2-validated)
- HMC-J as arch primitive (last)
- Only after all verification-native levers exhausted

> **HARD RULE:** Do NOT re-propose from-scratch-14B. It is a measured DOWNGRADE from the shipped 30B-A3B by scaling laws AND a ruled-out path.

---

## Soundness Map (the full discipline)

### What was measured and worked
- **Capacity scaling** — bigger IS better, up to 30B-A3B MoE at this compute class
- **Best-of-N / maj@N** — the cashable inference lever
- **RLVR safety** — no regression, no selection collapse at 8-cycle dose
- **Alignment gates** — EvalLab release gates block deployment on FAIL
- **Firewall discipline** — CleanTeacher fails CLOSED; closed-model data never touches release artifacts

### What was measured and ruled out
- **SFT as a capability lift** — flat (polish only)
- **RLVR fold-in at LoRA dose** — flat (3 independent runs)
- **OPD at affordable dose** — flat (dose-checked)
- **Tapered FFN** — neutral-null at proxy scale
- **Per-token routing genus** — falsified 3×
- **RSVA / HMC / FEPG** — assessed vacuous
- **Zero-shot LLM verifier** — NULL (over-approval, not discrimination)
- **Cheap learned verifiers** — can't beat majority at this sample size
- **From-scratch-14B** — measured downgrade vs shipped 30B-A3B

### What remains open (gated)
- **Trained math verifier** (GenRM / PURE-min-form PRM) — gated by compute
- **Full-param RLVR on 30B** — gated by compute (ZeRO-offload proven feasible)
- **CUA-execution-verified RLVR** — v4 active
- **QAT GGUF** — release-track, Kimi proves removable
- **FTPO** — undefined, locate spec before scoping

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-Q1 | Pivot from 1.3B pretraining to open-weight testbeds | Cost analysis: from-scratch pretraining not affordable |
| 2026-04 | Ship v2 = 30B-A3B + best-of-N | Measured optimum at Colab-class compute |
| 2026-06-30 | Park tapered FFN | Neutral-null at proxy scale — don't build v3 around architecture |
| 2026-07-01 | Redirect v3 to verifier/selection system | v3 = verifier system not architecture (confirmed by 8+ independent doc passes) |
| 2026-07-02 | Wire GLM-5.2 as clean teacher | MIT license, frontier-competitive, release-eligible |
| 2026-07-06 | Build Alignment EvalLab as release gates | Blocks deployment on FAIL; closes capability-only-release gap |
| 2026-07-08 | Close RLVR fold-in (definitive flat) | 8-cycle dose extension = flat; 3 independent confirmations |
| 2026-07-09 | ZeRO-offload 30B proven feasible | Colab G4 = plain LoRA, 61.5GB peak, no DeepSpeed needed |
| 2026-07-11 | v3 complete: v2 IS the optimum | All levers measured; remaining = gated-by-compute only |
| 2026-07-11 | Reject from-scratch-14B (permanent) | Measured downgrade, ruled-out path — HARD RULE |

---

## Repository Map

| Repo | Purpose |
|------|---------|
| `S3nna13/Aurelius` | Main repo — v1 backbone code, v3 eval harnesses, Composer, TruthSurface |
| `S3nna13/Aurelius-v2` | v2 shipped — Qwen3-Coder-30B-A3B GGUF, verifier-native |
| `S3nna13/Borealis` | Multi-model family + 200+ skills + POLARIS governance |
| `S3nna13/mosaic` | MOSAIC unified framework — Moses/Setus/Aigis + Aurelius capabilities |
| `Zephyrs33` (HF) | 6 private model repos — Qwen3-based experimental checkpoints |

---

> **Compass:** v3 training = RLVR-centric. v3 serving = verifier-gated best-of-N.
> v4 = generalize the verifier/RLVR thesis to CUA. v5 = from-scratch only when everything else is spent.
>
> *"No mechanism is promoted unless it improves held-out pass@1 / oracle@K under an ACDT falsifier contract."*
