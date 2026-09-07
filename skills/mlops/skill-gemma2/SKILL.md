---
name: skill-gemma2
description: >
  Reference skill for Gemma 2 Google DeepMind 2024, arXiv:2408.00118. Covers the Gemma 2 family 2B / 9B / 27B, key architecture changes (interleaved local-global attention, GQA, logit soft-capping, dual RMSNorm), knowledge-distillation training for 2B and 9B variants, benchmark results versus Gemma 1 / LLaMA 2 / Mistral / LLaMA 3 / Qwen 1.5, Chatbot Arena Elo scores, multi-turn human evaluation, safety and assurance evaluations (CBRN, offensive security, self-proliferation, memorisation). Use when reproducing Gemma 2 architecture or training changes, designing distillation schedules, looking up benchmark numbers, writing safety evaluations, or comparing Gemma 2 to other open-weight models.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [gemma, google-deepmind, open-source, llm, transformer, distillation, rmsnorm, local-attention, gqa, safety, benchmarks, faq-eval]
    related_skills: [skill-llama2, skill-deepseekv2, skill-qwen2]
---

# Gemma 2: Improving Open Language Models

> **Gemma Team, Google DeepMind** (2024) — arXiv:2408.00118v3 · Jun 2024 · 21 pages
> Core contributors (equal \*): Morgane Riviere\*, Shreya Pathak\*, Pier Giuseppe Sessa\*, Cassidy Hardin\*, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem. Lead: Armand Joulin. Technical leads: Kathleen Kenealy, Robert Dadashi, Alek Andreev.

---

## 1. Paper Overview & Metadata

| Field          | Value                                                          |
|----------------|----------------------------------------------------------------|
| **Title**      | Gemma 2: Improving Open Language Models at a Practical Size    |
| **Authors**    | Gemma Team, Google DeepMind (see full roster above)            |
| **arXiv ID**   | 2408.00118v3 (originally Jun 2024; updated Oct 2, 2024)       |
| **Institution**| Google DeepMind                                                |
| **Pages**      | 21                                                            |
| **License**    | Open (permissive) — https://kaggle.com/models/google/gemma     |
| **GitHub**     | https://github.com/google-deepmind/gemma                        |

---

## 2. Core Thesis

Gemma 2 is a new generation of the Gemma open-model family that **significantly advances the state-of-the-art for lightweight (2B–27B parameter) language models** without solely increasing token volume. Its core thesis rests on three pillars:

1. **Knowledge Distillation over sheer token volume** — for the 2B and 9B variants, replace next-token prediction with cross-entropy against teacher-model output distributions; train with >50× the compute-optimal token budget. This yields dramatic quality uplifts even at fixed token counts (cf. 10% absolute gains on 9B).

2. **Simplified but powerful architectural modifications to Transformers** — interleaved local sliding-window and global attention layers (Beltagy et al., 2020a, Longformer), Grouped-Query Attention (Ainslie et al., 2023, GQA), logit soft-capping, and dual RMSNorm (pre-norm + post-norm). Each change is minimal individually but collectively they make deep, narrow networks much better.

3. **Rigorous safety and responsible deployment** — three-pillar framework (train-time mitigations, robust transparency evaluations, and the Responsible Generative AI Toolkit). Gemma 2 models actually score *better* than GPT-4o on held-out safety benchmarks, and their violation rates on child-safety content drop significantly vs. Gemma 1.1.

The quantifiable claim: Gemma 2 models are **the best in their parameter class** and are even competitive with models **2–3× larger** (e.g., Gemma 2 27B surpasses LLaMA-3 70B on Chatbot Arena Elo), delivering this at practical, cost-accessible scales.

---

## 3. Model Variants

| Variant | Non-embedding params | Total params | Regularisation | Pre-train tokens |
|---------|:----:|:--------:|:------:|:--------:|
| **Gemma 2 2B IT** | 2.03B | 2.59B | GQA, distil. (from 7B) | 2T |
| **Gemma 2 9B IT** | 8.32B | 9.24B | GQA, distil. (from 27B) | 8T |
| **Gemma 2 27B PT** | 26.05B | 27.23B | GQA, trained from scratch | 13T |

> *IT = Instruction Tuned; PT = Pre-Trained base.* No multimodal model released in this paper; models are English-majority text (not specialised multilingual).

---

## 4. Architecture Improvements

All changes relative to Gemma 1:

### 4.1 Decoder-only Transformer baseline
Same decoder-only foundation as Gemma 1/Gemini — context length 8 192 tokens, RoPE rotary-position embeddings, GeGLU approximated non-linearity.

### 4.2 Interleaved Local / Global Attention
Alternate every other layer: one deep layer uses **local sliding-window** attention (4096-token span, Longformer/Beltagy et al.), the next uses **global** attention (full 8 192-token span). Local attention reduces attention compute quadratically vs. naive global; alternating preserves span coverage.

### 4.3 Grouped-Query Attention (GQA)
Replace full MHA with GQA **(num_groups = 2)** — 8 KV heads share the same key/value projections for 16 query heads at 9B. Ablations show near-identical accuracy vs. MHA while reducing KV parameter count and improving inference speed.

### 4.4 Logit Soft-Capping
Apply `tanh` clipping to each attention layer's logits and again at the final layer:
```
logits ← soft_cap · tanh(logits / soft_cap)
```
- self-attention layers: `soft_cap = 50.0`
- final output layer: `soft_cap = 30.0`
This is inspired by Bello et al. (2016) and provides implicit entropy regularisation.

### 4.5 RMSNorm — Pre AND Post Norm
Apply **RMSNorm** (Zhang & Sennrich, 2019) to *both* the input and output of every transformer sub-layer, every attention layer, and every feed-forward layer — a departure from the pre-norm only pattern of Gemma 1. Post-norm insertion stabilises deeper networks (depth-for-width is favoured; see §5 below).

### 4.6 Wide vs. Deep — Favour Depth
Ablations show a **deeper 9B is better than a wider 9B** at identical parameter counts (52.0 vs 50.8 avg benchmark score). Gemma 2 consistently chooses more layers, fewer heads, and thus narrower per-layer dimensions: 2B→26 layers, 9B→42 layers, 27B→46 layers.

### 4.7 Vocabulary
Shared 256 128-token SentencePiece BPE (same as Gemma 1 / Gemini). Digits split, whitespace preserved, byte-level fallback. Tied embedding weights.

### Full Architecture Table

| Parameter             | 2B      | 9B      | 27B     |
|-----------------------|---------|---------|---------|
| `d_model`             | 2 304   | 3 584   | 4 608   |
| Layers                | 26      | 42      | 46      |
| Non-linearity         | GeGLU (approx.) | GeGLU | GeGLU |
| FF dim                | 18 432  | 28 672  | 73 728  |
| Head type             | GQA     | GQA     | GQA     |
| Num heads             | 8       | 16      | 32      |
| Num KV heads          | 4       | 8       | 16      |
| Head size             | 256     | 256     | 128     |
| Global att. span      | 8 192   | 8 192   | 8 192   |
| Sliding window        | 4 096   | 4 096   | 4 096   |
| Vocab size            | 256 128 | 256 128 | 256 128 |
| Tied embedding        | yes     | yes     | yes     |

---

## 5. Training Pipeline

### 5.1 Pre-training

| Model | TPU config        | Chips | Data shards | Model shards |
|-------|-------------------|-------|-------------|-------------|
| 2B    | TPUv5e, 2×16×16   | 512   | 512         | 1           |
| 9B    | TPUv4, 8×16×32   | 4 096 | 1 024       | 4           |
| 27B   | TPUv5p, 8×24×32  | 6 144 | 768         | 8           |

Optimiser sharded to ZeRO-3 levels; all-reduce over data-centre network (Pathways Barham et al. 2022). GSPMD partitioner; MegaScale XLA compiler.

- **Data sources**: web documents, code, science articles — primarily English.
- **Filtering**: same pipeline as Gemma 1 — remove Personal Identifiable Information (PII), unsafe content; decontaminate evaluation sets.
- **Carbon**: ~1 247.61 metric tons CO₂ equivalent; Google data centres are carbon-neutral.

### 5.2 Knowledge Distillation
Only applied to **2B and 9B** — they are the distillation targets, trained with a 27B teacher:
```
min Σ_x   − P_T(x | x_c) · log P_S(x | x_c)
```
Distillation run for >50× the compute-optimal token count (Chinchilla regime). The result: distillation beats next-token prediction at the same token budget (e.g., 2B @ 500B tokens: distilled→67.7 avg score vs. scratch→60.3).

**Perplexity ablation** — distillation reduces token-perplexity across all model sizes significantly, and the gain holds at larger student sizes.

### 5.3 Post-training (SFT → RLHF → Merge)
1. **SFT** — behavioural cloning on synthetic and human prompts; responses mainly synthetically generated by the teacher (27B). Distillation on student distribution included.
2. **RLHF** — same algorithm as Gemma 1.1 but with reward model **10× larger** than the policy; reward model oriented toward multi-turn conversations.
3. **Model merge** — Werner-style weight-averaging of multiple SFT/RLHF checkpoints (Ramé et al., 2024, WARPa).
4. **Data filtering** — remove PII, unsafe outputs, self-identification errors, duplicates; include subsets promoting in-context attribution, hedging, and refusal for hallucination reduction.

### 5.4 Safety Architecture
SFT and RLHF trained specifically to minimise harm (see §8).
Formatting uses `<start_of_turn>user` / `<start_of_turn>model` / `<end_of_turn>` / `<eos>` — explicitly EOS-ate end-of-turn to enable clean multi-turn continuation.

---

## 6. Benchmark Results

### 6.1 Pre-training Evaluations — 27B vs. LLaMA-3 70B & Qwen 1.5 32B
Gemma 2 27B beats Qwen 1.5 32B (same-size class) on all five HuggingFace benchmarks and comes very close to LLaMA-3 70B despite being **2.5× smaller** and trained on **2/3 the data**:

| Benchmark | Gemma 2 27B | LLaMA-3 70B | Qwen 1.5 32B |
|-----------|:-----------:|:-----------:|:------------:|
| MMLU      | 75.2        | 79.2        | 74.3         |
| GSM8K     | 74.0        | 76.9        | 61.1         |
| ARC-c     | 71.4        | 68.8        | 63.6         |
| HellaSwag | 86.4        | 88.0        | 85.0         |
| Winogrande| 83.7        | 85.3        | 81.5         |

### 6.2 Full Benchmark Suite — All Variants (2B → 9B → 27B; Table 13)
Gemma 2 9B improves vs. Gemma 1 7B by up to **+10 pp** on some benchmarks; Gemma 2 27B achieves the highest scores across all open-weight models of comparable/>2× scale.

| Benchmark        | G1 2B | G2 2B | Mistral 7B | LLaMA-3 8B | G1 7B | G2 9B | **G2 27B** |
|------------------|------:|------:|:----------:|:----------:|------:|------:|:------:|
| MMLU (5-shot)    | 42.3  | 52.2  | 62.5       | 66.6       | 64.4  | 71.3  | **75.2** |
| ARC-C (25-shot)  | 48.5  | 55.7  | 60.5       | 59.2       | 61.1  | 68.4  | **71.4** |
| GSM8K (5-shot)   | 15.1  | 24.3  | 39.6       | 45.7       | 51.8  | 68.6  | **74.0** |
| BBH (3-shot CoT) | 35.2  | 41.9  | 56.0       | 61.1       | 59.0  | 68.2  | **74.9** |
| HellaSwag (10)   | 71.7  | 72.9  | 83.0       | 82.0       | 82.3  | 81.9  | **86.4** |
| HumanEval        | 22.0  | 20.1  | 26.2       | —           | 32.3  | 40.2  | **51.8** |
| **Avg (8)**      | 44.0  | 50.0  | 61.0       | 61.9       | 62.4  | 70.2  | **74.4** |
| Avg (all)        | 44.2  | 48.7  | 55.6       | —           | 57.9  | 64.9  | **69.4** |

### 6.3 Chatbot Arena (Human-blind Elo) — Table 14
| Model | Elo  | 95% CI |
|-------|-----:|--------|
| gemma-2-27b-it    | **1 218** | +4 / -3 |
| llama-3-70b-instruct | 1 206 | +2 / -2 |
| gemma-2-9b-it     | **1 187** | +3 / -5 |
| gpt-4-0314        | 1 186 | +2 / -3 |
| gemma-2-2b-it     | 1 126 | +10/-10 |
| gpt-3.5-turbo-0613| 1 116 | +3 / -4 |

> **Gemma 2 27B IT is ranked higher than LLaMA-3 70B and competitive with GPT-4-0314**, despite being ~13× smaller. This is the flagship claim of the paper.

### 6.4 Human Evaluations — Instruction Following & Safety (Table 15)
| Model | Instruction Following % | Safety win / tie / loss rate |
|-------|:----------------------:|:----------------------------:|
| Gemma 1.1 IT 7B | 24.3% | 37.4% / 10.8% / 51.8% |
| Gemma 2 IT 2B   | 26.5% | **53% / 9% / 38%**   |
| Gemma 2 IT 9B   | 34.1% | **48.2% / 19.2% / 28.3%** |
| Gemma 2 IT 27B  | 37.7% | **49.6% / 10.8% / 39.6%** |

Gemma 2 models **produce safer, more appropriate responses than GPT-4o** on held-out safety prompts.

---

## 7. Ablations

| Ablation | Finding |
|----------|---------|
| **Distillation vs. scratch (2B, 500B tokens)** | Avg score 67.7 vs. 60.3 (+7.4 pp) |
| **Perplexity at increasing student sizes** | Gains persist as student grows; smaller students benefit most |
| **GQA vs MHA (9B, 4 benchmarks)** | Avg 50.8 vs 50.3 — negligible; GQA chosen for speed |
| **Wide 9B vs Deep 9B** | 52.0 deep vs. 50.8 wide — favour deeper networks |
| **Sliding window 4096→1024 (at inference time)** | Perplexity rises modestly 1.63→1.64 — inference speed gain without catastrophic loss |
| **Formatting robustness (MMLU std dev)** | Gemma 2 2B=2.1, 9B=0.9, 27B=1.0 — Mistral 7B worst at 6.9; Gemma 1 7B=0.7; Gemma 2 preserves robustness |

---

## 8. Safety Evaluation

### 8.1 Framework (three pillars)
1. **Train-time mitigation** — safety-focused SFT + RLHF; evaluation against 6 harm categories (child safety, PII, hate speech, dangerous content, sexually explicit, counter-consensus medical)
2. **Transparent external evaluations** — standard academic benchmarks (RealToxicity, CoS-Pairs, BBQ, Winogender, TruthfulQA, ToxiGen)
3. **Responsible Generative AI Toolkit** — LLM Comparator, agile safety classifiers, prompt-debugging, interpretation tools

### 8.2 External Benchmark Results (Table 18)
| Benchmark | G1.1 IT 2.5B | G1.1 IT 7B | G2 IT 2.6B | G2 IT 9B | **G2 IT 27B** |
|-----------|:-------------:|:----------:|:----------:|:---------:|:------------:|
| RealToxicity avgTox | 7.03 | 8.04 | 8.16 | 8.25 | **8.84** (lower is worse) |
| CrowS-Pairs top-1 % | 45.89 | 49.67 | **37.67** | **37.47** | **36.67** (↓ better) |
| RealToxicity avgTox | 29.64 | 38.75 | 48.32 | **39.30** | **38.42** |
| BBQ Ambig (4-shot)  | 58.97 | 86.06 | 83.20 | 88.58 | 85.99 |
| BBQ Disambig        | 53.90 | 85.08 | 69.31 | 82.67 | 86.94 |
| Winogender top-1    | 50.14 | 57.64 | 52.91 | **79.17** | 77.22 |
| TruthfulQA MC2 Acc  | 44.24 | 45.34 | 43.72 | 50.27 | **51.60** |

> Key results: Gemma 2 IT achieves **lower bias / higher truthfulness / lower toxicity** than Gemma 1.1 on nearly all axes despite being much smaller (2B–2.6B pairing showed TruthfulQA 43.72 vs 44.24 reality).

### 8.3 Assurance Evaluations (extreme-risk domain)
| Capability   | Gemma 2 27B | Gemma 1.0 Ultra | LLaMA-3 FacepalM |
|-------------|-------------|----------------|-----------------|
| InterCode CTF (score) | 34/76 (45%) | 28/76 (37%) | 12/76 (16%)  |
| Self-proliferation success | 49% | 36%  milestones | low |
| CBRN knowledge | **Low** | — | — |
| Code vuln detection | 63% PrimeVul | 54% Gemini Ultra | — |

### 8.4 Memorization & Privacy
- Exact memorisation rate: **< 0.1%** across all Gemma 2 flavours — significantly lower than Gemma 1 (log-y-axis)
- Approximate memorisation: near-exact rates; virtually no lift vs. exact-mem approximation
- Personal data emission: **0.00026%** lower-severity PII; zero high-severity data
- Training data used Google Cloud Sensitive Data Protection for filtering

---

## 9. Comparison to Prior & Competing Models

### 9.1 vs. Gemma 1
| Dimension           | Gemma 1 7B      | Gemma 2 9B        | Change                        |
|---------------------|:---------------:|:-----------------:|-------------------------------|
| Pre-Norm only       | Pre-norm only   | Pre + Post-norm   | Training stability improvement |
| Attention scheme    | Full self-att.  | Interleaved local/global | Better long-context efficiency |
| GQA                 | No              | Yes               | Fewer KV params, faster inference |
| Logit capping       | No              | Yes               | Implicit entropy regularisation |
| Distillation        | No              | Yes (2B, 9B)       | +7pp+ on 2B; 9B massively up |
| Training tokens     | not disclosed  | 8T (9B)           | Denser per token quality      |
| MMLU                | 64.4            | **71.3**          | +6.9 pp                       |
| HellaSwag           | 82.3            | 81.9 (-0.4 pp)    | Slightly below                 |
| HumanEval           | 32.3            | **40.2**          | +7.9 pp                       |

### 9.2 vs. LLaMA 2 (Meta, 2023)
| Model              | MMLU | GSM8K | HumanEval | BBH        | Params |
|--------------------|:----:|:-----:|:---------:|:----------:|:------:|
| LLaMA 2 7B Chat    | —    | —     | ~12       | ~45        | 7B     |
| LLaMA 2 70B Chat   | 78.5 | 72   | —          | —          | 70B    |
| Gemma 2 9B IT     | 71.3 | 68.6  | 40.2      | 68.2       | 9B     |
| Gemma 2 27B IT    | 75.2 | 74.0  | 51.8      | 74.9       | 27B    |

> Gemma 2 9B comfortably outperforms LLaMA 2 70B Chat on multiple benches despite being an order-of-magnitude smaller; Gemma 2 27B is clearly state-of-the-art in that parameter class. Caveat: direct comparisons limited by different evaluation protocols (ephemeral).

### 9.3 vs. Mistral 7B (01.ai, 2023)
| Model            | MMLU | GSM8K | HellaSwag | HumanEval | Avg (8) |
|------------------|:----:|:-----:|:---------:|:---------:|:-------:|
| Mistral 7B (IT)  | 62.5 | 39.6  | 83.0      | 26.2      | 61.0    |
| Gemma 2 2B (IT)  | 52.2 | 24.3  | 72.9      | 20.1      | 50.0    |
| Gemma 2 9B (IT)  | 71.3 | 68.6  | 81.9      | 40.2      | 70.2    |

> Gemma 2 9B is +9 pp vs. Mistral 7B on overall average — a striking uplift at similar scale from architectural changes + distillation.

### 9.4 vs. LLaMA 3 (Meta, 2024)
| Model             | MMLU | GSM8K | HumanEval | Params | Elo              |
|-------------------|:----:|:-----:|:---------:|:------:|:----------------:|
| LLaMA 3 8B IT     | 66.6 | 45.7  | —         | 8B     | ~1 151           |
| Gemma 2 9B IT     | 71.3 | 68.6  | 40.2      | 9B     | **1 187 (+36)**  |
| LLaMA 3 70B IT    | 79.2 | 76.9  | —         | 70B    | 1 206            |
| Gemma 2 27B IT    | 75.2 | 74.0  | 51.8      | 27B    | **1 218 (+12)**  |

> Gemma 2 9B outranks LLaMA 3 8B on Chatbot Arena Elo by +36 pts — a large gain for only 1B extra params. Gemma 2 27B outranks LLaMA 3 70B by +12 pts, suggesting distillation + architectural choices over sheer scale does meaningful work.
>
> *Limitation: "spiral-alert-aware attention mechanisms" is not a term used in the paper; the paper uses standard GQA + local/global attention, Soft-capped logits, and post-training safety mitigations.*

---

## 10. Key Technical Decisions & Rationale

| Decision | Why |
|----------|-----|
| **Distillation on 2B & 9B only** | Maintains a from-scratch 27B baseline while pushing 2B/9B to be faster-to-train small models |
| **GQA with num_groups=2** | Ablations showed within-noise-level accuracy loss vs MHA; KV memory and inference speed improved |
| **Deep (depth-favoured) over wide** | 28-layer depth vs. fewer layers gave consistent +1 pp on 4-bench averages |
| **Post-norm + RMSNorm** | "Born again" (BABY) Pre+Post norm helps gradient flow in deep networks; avoids training instability at >26 layers |
| **Soft-capped logits** | Implicitly restricts output entropy; simulates temperature annealing inside forward passes without extra training cost |
| **Local-global attention alternation** | Approximate full global context coverage with only half the global-attention compute; sliding window halves local-attention compute further |
| **RLHF reward model 10× larger** | ParlAI/Anthropic work shows reward model scale directly stabilises RLHF training; splits the compute budget to reward model training instead of policy-size inflations |

---

## 11. Limitations

1. **English-majority** — models are not tuned for true multilingual performance; training data is primarily English.
2. **8 192 token context** — restrictive for long-document tasks; no RoPE extrapolation tricks enabled at inference.
3. **Safety benchmark coverage** — while comprehensive, cannot conceivably cover all real-world deployment scenarios; users must conduct their own safety evaluations.
4. **Risk landscape** — the paper concedes the release is not risk-free; opening models at large scale, even well-intentioned, creates adversarial exposure.
5. **Self-proliferation / offensive security** — Gemma 2 27B shows some Capabilities (e.g., InterCode CTF 45%); this is a self-reported ceiling not a floor; frontier models can cause harm in agentic settings.
6. **Interpretability gap** — safety mitigations in §8 are procedure descriptions, not mechanistic explanations (e.g., no "spiral-alert" terminology or mechanism).
7. **Evaluation standardisation** — many ablation runs were evaluated using slightly different prompts to LLaMA-3 baselines (+3–4 pp artefacts); direct cross-paper comparisons need caution.

---

## 12. Reproduction Guide

### Hardware
- 9B: H100-class GPU or A100 cluster; 8B model fits in ~16 GiB of GPU VRAM at 4-bit, ~32 GiB at 8-bit.
- 27B: multi-GPU + offloading; ~52 GiB at 4-bit, ~108 GiB at 8-bit.

### Pre-training (conceptual)
```
1. Build training corpus: web docs + code + science articles; filter PII/toxic content.
2. Tokenise with SentencePiece BPE, vocab 256 128; decontaminate eval sets.
3. Shard data replicated; shard model (ZeRO-stage 3); train with GSPMD partitioner.
4. Add RMSNorm (pre+post); alternating local/global attention layers; GQA with num_groups=2.
5. Apply logit soft-caps: self-att=50, final=30.
6. For 2B & 9B: train with knowledge distillation ≥ 50× Chinchilla-optimal token count.
```

### Post-training
```
1. SFT: behavioural cloning on synthetic (teacher-generated) + human prompts.
2. Distillation on student distribution (Agarwal et al., 2024).
3. RLHF: reward model (10× policy size, multi-turn oriented); PPO / RPO-style update.
4. Merge: averaged checkpoint ensembles (Ramé et al. 2024).
5. Safety SFT/RLHF: RLHF optimal hyperparameters target minimal child-safety/counter-medical harm.
```

### Known Pitfalls
- **Distillation ablation mismatch** — Table 7 uses 200M/400M/1B student sizes with a 7B fixed teacher; replicate at your target scale for accurate signal.
- **Local window memory** — Seq-wise sliding window still attends to 4096 past tokens; long sequences still O(n × 4096) compute — not free.
- **Soft-cap tail clipping** — setting `soft_cap` too high (no clipping) or too low (< 5) causes training instability; 50 (self-att) / 30 (final) are tuned hyper-parameters.
- **RLHF reward model size** — reward models much larger than policy (10× param) are computationally expensive; please ensure adequate compute for this step.
- **Benchmark parity** — LLaMA-3's reported BBH/AGIEval numbers used slightly different eval prompts (+3–4 pp offset); always re-run benchmarks on your own harness when yielding research conclusions.

---

## 13. Checklist — Using This Skill

Use this skill whenever:
- [ ] You are discussing or reproducing Gemma 2 architecture (2B / 9B / 27B)
- [ ] You are comparing Gemma 2 to Gemma 1, LLaMA 2, LLaMA 3, Qwen, Mistral
- [ ] You need to design knowledge-distillation-based training schedules
- [ ] You need logit-soft-capping, RMSNorm pre/post, GQA, or local-global interleaved attention
- [ ] You need benchmark numbers for MMLU / HellaSwag / HumanEval / MATH / GSM8K
- [ ] You are writing safety evaluations, impact assessments, or Responsible AI guidance
- [ ] You are using the Chatbot Arena Elo runner or human-evaluation templates

Do NOT use this skill for:
- Gemma 1 (see `skill-gemma1`)
- Gemma 3 / Gemini lineages
- Quantisation techniques (use `skill-quant`)
- Fine-tuning peft / LoRA pipelines (use `skill-peft`)
