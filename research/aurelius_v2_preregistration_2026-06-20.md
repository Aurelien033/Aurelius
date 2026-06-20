# Aurelius v2 — Preregistration
### 2026-06-20 · a fully-improved BROAD model (code + reasoning + math), built by stacking what v1 proved

> v1 result: Qwen3-8B + RLVR = 85.4% > 84.1% base (reproduced). v2 stacks the proven levers (bigger base +
> warm-start SFT + RLVR) and widens to **code + reasoning + math** with multi-domain verifiers. Honest-first:
> every stage has a baseline + a gate; we keep only what beats it.

---

## 0. Goal & success criterion
A clean, Apache/MIT model that is **measurably and reproducibly better than its (bigger) base across a broad
battery** — code **and** math **and** reasoning — built from clean data + verifier-reward RL, no Claude outputs.

**v2 succeeds iff:** the final model beats its *own base* by a clear margin on a **held-out battery**
(HumanEval + MBPP-held-out + LiveCodeBench + GSM8K + MATH + a reasoning bench), reproduced across two seeds, with
no domain regressing below base. Target: a *multi-point* gain (v1 was +1.3pp on one bench; v2 aims broader + bigger).

## 1. The base (the #1 ceiling lever)
- **Primary: Qwen3-14B** (Apache, thinking) — the proven pattern is "bigger base raises the ceiling," and 14B fits
  a single A100-80GB (or 2×40GB) for LoRA-RLVR on RunPod/Lambda.
- **Stretch: Qwen3-32B / QwQ-32B** if budget allows (H100s) — the biggest jump, ~2× the cost.
- Decided per the eval after Phase 1; don't pre-commit to 32B before 14B shows the lift.

## 2. Multi-domain verifiers (the heart of a BROAD verification-native model)
RLVR needs a ground-truth checker per domain. Reward = pass/fail (or graded), advantage group-relative.
| domain | tasks (train) | verifier | status |
|---|---|---|---|
| **code (function)** | MBPP, repair gym | execute vs unit tests (`run_program`) | ✅ built |
| **code (competitive)** | code_contests (calibrated rating) | stdin/stdout (`run_io_tests`) | ✅ built |
| **math** | GSM8K, MATH (train) | extract final answer, compare (numeric/symbolic) | ⚙ `rlvr.py:MathReward` exists → wire it |
| **reasoning** | logic / multiple-choice w/ answers (e.g. the gym, ARC, or synthetic) | answer-match verifier | ⚙ build |
**Eval held-out from all training sources** (HumanEval, MBPP-test, LiveCodeBench, GSM8K-test, MATH-test, GPQA/MMLU-sub).

## 3. The staged pipeline (stack the levers)
- **Stage 0 — base + broad smoke.** Load Qwen3-14B; measure the base battery (the bar to beat).
- **Stage 1 — strong-teacher SFT warm-start.** Distill a teacher *stronger than the base* (GLM-5.2 / Qwen2.5-Coder-32B
  / DeepSeek-R1 **via API** for speed) on **verified** code+math+reasoning traces → raise the floor. *(Path B done
  right — the stronger-teacher test v1 never finished.)* Gate: SFT ≥ base on the battery (no regression).
- **Stage 2 — multi-domain RLVR** (`grpo_train.py`, on the SFT'd model). `--data` mixes code+math+reasoning, each
  task **difficulty-calibrated to the 14B's learnable band** (the v1 lesson: ~20–50% pass-rate = gradient). More
  steps than v1. Gate: beats SFT *and* base on the battery, reproduced.
- **Stage 3 — DPO (optional).** Real preference pairs (verified-correct vs verified-wrong from rejection sampling).
- **Stage 4 — compress + serve.** LayerDelta on the serving distribution + local-first quantized + verifier-gated
  → a deployable release + model card.

## 4. Compute plan (matched to your resources)
- **Free (Colab Pro + Kaggle T4): all DEV + smokes** — build & validate the new pieces (math reward, reasoning
  verifier, broad eval) on 0.5–1.5B models; cost $0.
- **RunPod/Lambda (A100-80GB, ~$1.5–2.5/h): the 14B runs** — Stage 1 SFT (~few h) + Stage 2 RLVR (~10–20 h).
  Estimated **~$40–100** for a full v2 cycle (14B). 32B ≈ 2–3×.
- **API (~$2–10): the strong teacher** (Stage 1 traces) — far cheaper/faster than local 4-bit.
- Checkpoint everything (`--save_every`) — rented sessions + spot instances die.

## 5. Build queue (free, on Colab/Kaggle — do BEFORE renting GPUs)
1. **Math RLVR**: `load_math_tasks` (GSM8K/MATH) + math verifier (port `rlvr.py:MathReward`: extract `\boxed{}`/last
   number, compare; add sympy for MATH symbolic) → `grpo_train --data math`.
2. **Reasoning RLVR**: a verifiable reasoning task loader + answer-match verifier (multiple-choice / short-answer).
3. **Broad eval harness**: extend `eval_code_bench` → `eval_battery` (HumanEval, MBPP, LiveCodeBench, GSM8K, MATH,
   GPQA/MMLU-sub) with one report.
4. **Curriculum / difficulty calibration**: a quick "base pass-rate per task" pass to pick the learnable band per
   domain (avoids the v1 too-easy/too-hard miss).
5. **Strong-teacher broad SFT data**: `make_traces` over code+math+reasoning via API teacher.
All five smoke on a 1.5B for $0; then the real runs go to RunPod.

## 6. Falsifiers / honest gates (the discipline that made v1 trustworthy)
- Each stage must **beat the prior stage's model on the held-out battery**, reproduced (2 seeds), no domain
  regressing below base. A stage that doesn't is dropped, not shipped.
- **Independent re-score** a sample of completions per stage (catch verifier/harness bugs — as throughout v1).
- If 14B + full stack doesn't clearly beat the 14B base → the honest result is "RLVR gain is base-capability-bounded
  at this scale too" → the lever is 32B, not more pipeline. (Same conclusion shape as v1, one tier up.)

## 7. Risks (named up front)
- **Sparse reward** on math/reasoning if tasks miss the learnable band → calibrate (Build #4).
- **Domain interference** in multi-domain RLVR (code gains hurt math?) → monitor per-domain reward; weight the mix.
- **Reward hacking** (math answer-format gaming, code trivial-pass) → certificate-gated rewards (the batch-3 kernel).
- **Compute overrun** → 14B first; 32B only if 14B proves the lift; checkpoint + spot-resume.

## 8. The one-line v2 thesis
**A bigger clean base, warm-started by a stronger teacher, then improved by multi-domain RLVR against real
verifiers — the v1 recipe, scaled and widened — to a broad model that beats its base on code, math, AND reasoning,
proven honestly.**
