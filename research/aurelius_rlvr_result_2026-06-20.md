# Aurelius — RLVR Result Writeup
### 2026-06-20 · the verification-native win, honest-first

> **⚠ CORRECTION — 2026-06-25 (supersedes the original headline below).** A direct pass@k re-measurement
> (base vs this adapter, `docs/training/eval_code_bench.py --passk`, Colab A100) shows **RLVR ≈ base — within
> noise on every axis tested.** The original "+1.3pp, beat the base, reproducibly" headline is **withdrawn**.
> Details immediately below; the 2026-06-20 analysis is retained beneath it for the record.

---

## ⚠ Correction (2026-06-25): RLVR ≈ base, not a win

The 2026-06-20 headline overstated a noise-level result. Direct measurement:

- **HumanEval (164):** base 138/164 (84.1%) vs RLVR 140/164 (85.4%) → **+2 problems = +1.2pp, within noise**
  (SE ≈ 4.6). The earlier "reproduction" (140/164 at 150 *and* 400 steps) was the **same recipe producing the same
  number** — not independent replication of an *effect*.
- **MBPP (200) — exactly the robustness check this writeup queued as Next #2:** base **125/200 = RLVR 125/200**
  (62.5%); oracle@16 140 vs 139. **No gain** — and MBPP was *in* the training mix, so this is on-distribution.
- **Thinking-on (HumanEval n=80):** base 43/80 = RLVR 43/80. Identical (rules out the "eval suppressed CoT" excuse).

**Honest status: this RLVR adapter is inert vs base.** Diagnosis: **`lr 1e-6` (~10–100× too low for LoRA GRPO)** +
KL-anchor → the policy barely moved (the adapter is a near-no-op; it does change *some* HumanEval outputs, just
net-neutral), and the narrow F2/F3 gym distribution does not transfer to general code.

**What still stands:** the negatives (routing dead, SFT capped) and the strategic conclusion this doc already reached
— **the lever is a stronger base/teacher, not more RL on this 8B.** Only the specific "RLVR beat the base" claim is
withdrawn. Do **not** release this adapter as "beats base." Full data: memory topic file `aurelius-two-line-divergence`,
2026-06-25 entries; reproduce with `flip_analysis.py` on two `--dump` files.

---

## Original writeup (2026-06-20) — retained for the record; headline withdrawn (see Correction above)

> ~~RLVR against verifiers is the first and only lever that beat a strong base on held-out general code,
> reproducibly — the result to release/publish.~~  *(withdrawn 2026-06-25 — within noise; MBPP + thinking-on disconfirm.)*

---

## The one-line result
**Qwen3-8B + RLVR (GRPO vs execution verifiers, clean data, no teacher) → 85.4% HumanEval vs 84.1% base
(+1.3pp, reproduced across two independent runs).**

## The arc that led here (why this is the result that survived)
1. **Dynamic compute routing — falsified.** Three powered nulls (per-layer gain, per-task selector at k=2 and k=4).
2. **Structured pruning / LayerDelta — partial.** A few layers ≥ dense; LayerDelta substitutes layers in-distribution
   but does not generalize → compression, not a general gain.
3. **Supervised fine-tuning — could not beat the base.** Gym-trace SFT was *flat* (saturated 3B; 82.3% on 8B ≈ base);
   a *weaker* teacher (R1-7B) made it **worse** (76.2%) — distilling down-levels a strong base.
4. **RLVR — the win.** Reward = ground-truth correctness, so there is **no teacher ceiling**. It improved the
   training reward (+7%) and **transferred** to held-out HumanEval: **85.4% > 84.1%, reproduced** at 150 and 400
   steps (both 140/164).

## What we established about RLVR (the science)
- **It works and transfers** — in-distribution reward gains carried to held-out general code (SFT did not).
- **The gain is reproducible** (two independent runs → identical 140/164) → the +1.3pp is real, not a 2-problem draw.
- **It plateaus at ~85.4% on this 8B.** Tested from every angle:
  - more steps (150→400): no change;
  - harder tasks at cf_rating ≤1500 and ≤1100 (code_contests): both 84.8%, *below* the gym+MBPP result.
  The **learnable band matters** (the model must pass ~20–50% for gradient) — but even well-targeted hard tasks
  did not exceed the base-capability ceiling.
- **Implication:** to raise the ceiling further, change the *base* (the proven lever earlier: VibeThinker-3B 33% →
  Qwen3-8B 48→60% on the gym), not the RL.

## The defensible contribution
A clean, honest demonstration of the **verification-native recipe**: *clean verified data → verifier-reward RL → a
model measurably and reproducibly better than its strong base, with no teacher dependence.* Modest in magnitude,
rigorous in method (independent re-scoring, held-out eval, reproduction), and the **only** positive transfer in the
entire program. The negatives (routing dead, SFT capped) sharpen rather than weaken it.

## Engineering banked (reusable)
`grpo_train.py` (GRPO + verifier rewards, LoRA, grad-ckpt, chunked log-probs for 152k-vocab, expandable-segments,
checkpointing, `--think` for reasoning bases), `eval_code_bench.py` (HumanEval/MBPP pass@1 + stdin/stdout executor),
the trace + SFT + DPO pipeline, and the model card (`research/aurelius_rlvr_MODEL_CARD.md`).

## Next (when compute allows)
1. **Bigger base** (Qwen3-14B/32B) + the same RLVR — the lever that moves the ceiling.
2. **Confirm robustness** of +1.3pp on a second held-out bench (MBPP-held-out).
3. Merge LoRA + serve (local-first quantized, verifier-gated) for an actual release.
