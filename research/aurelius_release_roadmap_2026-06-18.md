# Aurelius — Release Roadmap (v1 → v1.x → v2)
### 2026-06-18 · how the model ships now, and how the "aspirational" ambitions earn their way in later

> Companion to `aurelius_decision_memo_2026-06-18.md` and `aurelius_selection_arc_writeup_2026-06-17.md`.
> Honest-first. The point of this doc: turn a pile of ideas into a *gated* plan, so ambition is reached by
> promotion through evidence, not by scope creep.

---

## 0. The promotion gate

Anything moves from "aspiration" to a real iteration only when it clears **all three**:

1. **Grounded** — rests on a *validated* result, not a named module. ("Gated DeltaNet" = real published mechanism; "Creativity Engine" = a name.)
2. **Justified** — a concrete need pulls it in (eval plateau, serving cost), not "it'd be cool."
3. **Affordable** — fits a compute/data budget you actually have or can fund.

This *is* the Aurelius method (falsify/validate each mechanism before integrating). The selection-arc nulls
exist because we held to it; the aspirational docs are aspirational because they integrate unvalidated
mechanisms on paper.

---

## 1. Assets in hand (2026-06-18 audit)

**Clean data already on disk** (`aurelius/data/`, 100% deterministic-synthetic — zero external-model outputs):
- `aurelius_reasoning_sft_v20/sft/train.jsonl` — **105,430** SFT examples `{prompt, response, system_prompt}` (+11,808 val). v18/v19 similar; v20 newest/largest (notes previously said v17).
- `…/preferences.jsonl` — **16,749** DPO pairs `{prompt, chosen, rejected}`.
- `…/pretrain_docs.jsonl` — **23,448** continued-pretrain docs (~21M tokens total corpus).
- Caveat: responses are template-generated → excellent for **format/persona/coverage**, NOT a reasoning-quality
  ceiling. Quality comes from verified teacher traces (below).

**Release-track scripts (built + smoke-validated, branch `spike/first-light-readiness`):**
- `make_traces.py` — verified rejection-sampled traces from a CLEAN teacher (DeepSeek-R1-Distill, MIT). Quality layer.
- `sft_train.py` — gentle LoRA SFT, loss-masked, before/after gym eval. Consumes v20 directly.
- `dpo_train.py` — DPO on the 16k pairs (reference = adapters-disabled). Stage 2.
- `continued_pretrain.py` — domain-adaptive CPT on the 23k docs. Optional Tier 1.
- `layerdelta.py` (+ frontier/generalization) — in-distribution structured-substitution compression.

**Serving knowledge base** (research, not yet code):
- `AI Plans/aurelius-release-inference-cost-roadmap-2026-06-17.md` — local-first quantized stack + hard quality
  gates; explicitly *don't* put layer-skip on the critical path (matches our nulls).
- `AI Plans/aurelius-inference-training-research-2026-06-18/` — 91 curated sources + anchor papers (GQA, Medusa,
  EAGLE-2, vLLM/PagedAttention, Sarathi-Serve, KVQuant) + the "certified computation-substitution stack" ordering.

---

## 2. The iteration ladder

### Aurelius v1 — ships now, clean, uses only assets in hand
**Recipe:** clean open base (VibeThinker-3B, MIT; or Qwen3-8B, Apache)
→ SFT (v20 105k scaffold **+** R1 verified traces for quality, `sft_train.py`)
→ DPO (16k pairs, `dpo_train.py`)
→ LayerDelta in-distribution compression
→ local-first quantized serving (MLX/GGUF Q8/Q4) behind **verifier gates + dense fallback**.
**Gate to ship:** before/after gym Δ ≥ 0 and no general-reasoning regression. **This is the floor — real today.**

### Aurelius v1.x — incremental, grounded, each step justified by an eval gap
- Curated **clean** corpus (the disciplined version of the 50GB plan — licensing-checked), `continued_pretrain.py`.
- Bigger base (3B → 8B) — highest-leverage single upgrade.
- Build out the serving stack from the inference research (budget ledger, prefix/response cache, structured-output
  constraints, spec-decode *with acceptance telemetry*).

**v1.x experiment queue (tooling built — run after v1 ships):**
| # | experiment | tool | the question |
|---|---|---|---|
| 1 | **base ablation** | `sweep_v1x.py` / `aurelius_v1x_sweep_colab.ipynb` (`--bases`) | VibeThinker-3B vs Qwen3-8B vs Coder-7B + our SFT — which wins? |
| 2 | **trace scaling** | same (`--trace_sizes 60,150,300`) | does pass-rate keep climbing with #verified traces? where's the plateau? |
| 3 | teacher ablation | `make_traces.py --teacher` | R1-Distill vs QwQ-32B vs Qwen-Coder — best *verified* traces? |
| 4 | real DPO pairs | (small builder, TODO) | verified-correct vs verified-wrong from rejection sampling → real preference signal |
| 5 | LayerDelta on release model | `layerdelta.py` | compress the SFT'd model on its serving distribution → measured decode savings |

Sweep harness emits a ranked held-out-pass-rate table + `sweep_results.json`; the winning base/trace-budget feeds
straight back into the v1 notebook config. #1 + #2 are the highest-leverage and share one job.

### Aurelius v2 — the ambitious from-scratch model, GATED on a research positive
Triggered **only if** the research track (skip-native OBL-063, or LayerDelta-frontier) yields a *validated,
replicated* mechanism worth owning. Then a from-scratch model that bakes in **that validated mechanism + grounded
arch pieces + the clean data foundation** — with real compute (~$150k+/cluster; a from-scratch 4T-token pretrain is
frontier-lab scale and only justified here).

---

## 3. The "aspirational" docs, tagged by gate

| Item | Verdict | Promotes as | Leave behind |
|---|---|---|---|
| **50GB dataset plan** | 🟢 promotable | v1.x data layer (licensing-checked, built when a model needs it) | the unmounted drive / un-fetched state; don't build data before a confirmed need |
| **10B BLT architecture** | 🟡 half-promotable | v2 arch: GQA, **MTP**, **BLT** byte-latent, **Gated DeltaNet** — each validated independently | the mechanism zoo (Theory-of-Mind/Creativity/NLC/LOML/UIDS/RIV/EA/FEPG) — unvalidated = scope dilution |
| **From-scratch 10B pretrain** | 🟡 conditional | v2, only on a research win + real compute | doing it *now* (the $150k cluster trap) |
| **50k-axioms / consciousness / horizon-2027** | 🔴 not an architecture | inspiration / option pool | treating as a build backlog — no validated mechanism inside |
| **CCS bootstrap traces (48, synthetic)** | 🔴 wrong purpose | route-controller schema dev only | using for SFT or paper claims |

---

## 4. Immediate next actions (no new data needed)

1. **Run v1 SFT** on Colab Pro: `sft_train.py --base WeiboAI/VibeThinker-3B --data <v20 train.jsonl>,<R1 traces>` → read Δ.
2. If Δ ≥ 0: **DPO** (`dpo_train.py` from the SFT adapter) → re-eval (confirm no regression).
3. **LayerDelta** the result on its serving distribution → measure realized decode savings.
4. Only if eval shows a *knowledge* gap: **continued-pretrain** on the 23k docs.

Floor = a clean, specialized, verifier-backed small model from data already on disk. Ambition = reached through
this, plus the research gate — never around it.
