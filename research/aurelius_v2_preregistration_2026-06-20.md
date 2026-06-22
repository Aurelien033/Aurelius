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

**Statistical-honesty note (from `aurelius-rlvr-humaneval-benchmark-improvement-research-2026-06-20.md`):** +2
problems on n=164 is *not* separable from base on a single unpaired test — **~150/164 is the threshold to clear 95%
confidence** vs the 138 base. So the real HumanEval target is **~150/164 (91.5%)**, reached via signal-density +
selection, *and* per-task paired analysis (which flips fewer problems are needed). Don't claim a win that's a 2-problem draw.

## 0.5. STEP ZERO — the pass@k capability map (run before renting any GPU)
The single most decisive cheap experiment (now built: `eval_code_bench.py --passk`): on the *failing* problems,
compare **pass@1(greedy)** vs **oracle@K(sampled)**. It forks the entire v2 strategy:
- **Big selection_gap** (model *can* sample correct answers, just doesn't pick them) → the cheap, inference-time
  levers win: **best-of-N with the execution verifier, reranking, distill-from-winners** — possibly no bigger base needed.
- **Small gap** (model rarely samples them) → it's a true **capability ceiling** → the lever *is* the bigger base.
**Do this first.** It's free (eval-only) and it tells you whether to spend $0 (selection) or $100 (14B capability).

## 0.7. Strategic frame & cost-efficient sequencing (from `aurelius-opus-qwen-max-parity-answer-2026-06-20.md`)
**The goal is a verifier-native SYSTEM, not just a better checkpoint.** Raw frontier parity from an 8–14B is not
realistic (from-scratch = $0.14M–$32M, months–years; confirmed). What *is* realistic is **Tier-2 workflow / in-domain
parity** — as useful as a frontier model on tasks where *correctness is checkable* — reached through search +
verification + repair + a data flywheel around a strong base. So v2 ships a *system*, and the checkpoint is one part.

**Sequence the levers cheap→expensive (spend $0 before $100s):**
1. **Truth surface** (free) — benchmark battery + failure ledger + **pass@k map** (done/building). *We are here.*
2. **Search + repair wrapper** (free, inference-time) — **best-of-N with the execution verifier**, one-step **repair mode**, generated-test reranker. Cashes in any selection gap *with no training*.
3. **Verified data flywheel** (cheap) — store candidates/failures/repairs/tests + chosen-vs-near-miss pairs → a self-generated verified dataset. *Verified backing:* **Self-Verified Distillation** `2605.26132` (Qwen3 self-generates + self-verifies its training data, no teacher) + the **`2601.00828`** caveat (self-correction has limits — weaker models self-correct better; measure compound-rate, stop when Δ<0.2pp/cycle).
4. **Distill search into the model** (cheap GPU) — verified-winner SFT + CID-DPO (winner vs near-miss) → the model does *directly* what search found. *Refinement (register-don't-adopt extract, Layer-4 CPS 2026-06-22):* distill the **decision points**, not just final answers — data shape `(partial_state, winner_action, loser_action, evidence)` from search branches, via the existing `token_credit_assignment.py` / `process_reward.py`. Teaches *how not to pick the dead branch*, not just what a good answer looks like.
5. **Smarter RLVR** (cheap–mid GPU) — learnability-band + positive-advantage + execution-grounded credit + repair reward (the §4.5 menu).
6. **Bigger base — LAST** ($, RunPod) — Qwen3-14B/32B, *only if* steps 2–5 still plateau. Don't pay for capability until the cheap selection/flywheel levers are exhausted.

This reorders §1–§5 below: the **bigger base moves to the end**, after the free system levers. Multi-domain (code+math+reasoning) applies at every step.

## 1. The base (raised LAST, only if cheap levers plateau)
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
  **Distillation recipe — DLCoT** `2503.16385` (verified, 2026-06-22): segment teacher traces (restate/approach/execute/verify/answer),
  keep only correct-final traces, but **RETAIN the wrong→correct self-correction pairs (don't prune the error)** and weight correction
  tokens 2× / verification 1.5× — teaches the teacher's *error-recovery strategy*, not just its answers (why R1-distilled models punch above the answers alone).
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

## 4.5. RLVR signal-density menu (folded from the 2026-06-20 improvement research — attack the plateau MECHANISM)
The v1 plateau is **sparse terminal reward + equal-reward groups (no gradient) + base ceiling**. Beyond a bigger
base, these directly densify the RL signal (try in Phase 2, cheapest first):
- **Learnability-band adaptive sampler / difficulty thermostat** — keep feeding tasks at ~20–50% pass-rate (auto-tuned). Fixes the v1 too-easy/too-hard miss. *Verified backing (direct arXiv, 2026-06-21):* **SC-SDPO** `2605.27765` (pass-rate-weighted self-distillation, weight `[p(1−p)]^½` = the sweet-spot; **+3.2/+4.3 on Qwen3-8B**) — the closest match to our base; **DIVA-GRPO** `2603.01106` (difficulty-adaptive variant advantage); also VADE `2511.18902`, D³S `2509.22115`, SAGE `2602.03143`. **+GDRO-GRPO** `2601.19280` (verified: no-regret adaptive difficulty-grouping + rollout reallocation = automated curriculum, ~+10% pass@8) and **The Art of Efficient Reasoning** `2602.20945` (verified, Qwen3 0.6–30B: keep ≥20–30% positive reward density or exploration collapses). **+CurES** `2510.01037` (verified: gradient-analysis / Bayesian-posterior curriculum — prompt selection + rollout allocation, beats GRPO +3.3/+4.82 at 1.5B/7B).
- **Positive-advantage / winner-only GRPO** — test the "negative updates damage the model" hypothesis (only reinforce passes).
- **Execution-grounded credit assignment** — reward where the code first diverges from passing, not just terminal.
  Concrete shaped reward (improvement-suite, 2026-06-20): `pass_fail + 0.05·syntactically_valid + 0.05·correct_entrypoint
  + 0.05·no_timeout + 0.05·passes_public_smoke − 0.10·format_violation`. **Gate: activate ONLY when the failure ledger
  shows local bugs dominate (>10% of remaining fails are base-fail/RLVR-win), and KILL if the shaped reward rises but
  strict pass@1 doesn't** (= reward hacking, shaping too loose).
- **Process-reward credit design (verified, 2026-06-22)** — **PURE** `2504.15275` (NeurIPS'25: **min-form** credit assignment, NOT sum — sum-form "collapses training even at the beginning"; min-form matches verifiable-reward perf in **30% of steps** + alleviates reward hacking by bounding the value range) + **PAPO** `2603.26535` (**decoupled advantage normalization**: outcome normed across ALL responses, process normed among CORRECT only). Drop-in upgrade to `process_reward_model.py` (~3 days); directly attacks the reward-hacking collapse §4.6 flags.
- **Generated/differential-test verifier expansion** — synthesize extra tests so the verifier is stricter (anti reward-hacking / trivial-pass). *Verified hygiene (direct arXiv, 2026-06-22):* **PAR** `2502.18770` (bound all RLVR rewards to [0,1]; rapid-growth-then-converge schedule) + **Hack-Verifiable** `2605.20744` (embed detectable hack-loopholes in training tasks → measure exploitation, retrain if >5%; small models hack more). *+Adaptive test budget:* **CodeRM** `2501.01054` (verified — "Dynamic Scaling of Unit Tests": scale the verifier's unit-test count by candidate difficulty/uncertainty; CodeRM-8B test generator, gains on HumanEval+).
- **Multi-turn repair RL** — let the model see the failing test output and fix (turns a 0 into signal). *Verified:* **SPOC** `2506.06923` (single-pass interleaved solve+verify self-correction; +8.8pp MATH500 / +10pp AMC23 on Llama-3.1-8B) — the in-pass variant for the math/reasoning domain. *Refinements (Layer-4 CPS extract, 2026-06-22):* **critique-effectiveness training** — score a critique by *causal* answer-improvement (`corrected_score − original_score`), not fluency (upgrade to `self_refine.py`); **solver/verifier disagreement harvesting** — feed the flywheel boundary cases (solver-confident/verifier-fails = overconfidence; solver-wrong/verifier-catches = repair) instead of random successes.
- **pass@k→pass@1 distillation** — if step-zero shows a selection gap, distill verified winners back into greedy. *Verified backing:* **Self-Verified Distillation** `2605.26132` (Qwen3, model is its own verified-data pipeline, NO external teacher — eliminates the teacher ceiling on math/science/code).
- **Domain-conditioned length reward (two-stage)** — *Verified:* **The Art of Efficient Reasoning** `2602.20945` (Qwen3-validated): per-domain length shaping `correctness·exp(−|tok−tok*|/τ)` with τ differing for math/code/prose (stops math-brevity leaking into prose) + two-stage RLVR (Stage-1 ~30% = length adaptation, Stage-2 ~70% = reasoning refinement). Low-complexity, directly fits Qwen3-8B.

## 4.6. Training-stability / collapse-prevention (cheap, ~0-param guards — instrument BEFORE the next big spend)
The v2 co-train FAILED via **catastrophic forgetting** (dense 73%→22%); RLVR also risks entropy/reward/mode collapse. From the collapse-prevention doc (2026-06-22; mechanisms are standard hygiene — *its M2101–M2110 IDs = register-don't-adopt*). All hook into **REAL modules** (`src/alignment/dr_grpo.py` = the active GRPO trainer, `src/training/adaptive_sft.py`, `src/eval/verifier_metrics.py`):
- ⭐ **Capability-retention eval every N steps** — fixed held-out battery (general/code/math/instruction); regression past threshold → **rollback**. This is *exactly* what would have caught the 73%→22% co-train collapse WHILE it ran. Highest-value, 0-param.
- **Entropy monitor + adaptive control** — track policy entropy + advantage-variance in `dr_grpo.statistics()`; adaptive clip + token-level entropy reg (NOT a fixed coefficient). Anti entropy/mode collapse. *Verified formulas (direct arXiv, 2026-06-22):* **STEER** `2510.10150` (token-level reweight `adv[t] *= (1+λ·entropy_change[t])`, theory-grounded) + **DSDR** `2602.19895` (dual-scale: global trajectory-diversity bonus + local token-entropy reg on CORRECT trajectories only, with a correctness-preserving guarantee).
- **Sustain entropy via problem synthesis** — **SvS** `2508.14029` (verified): synthesize varied problem instances from correct solutions → sustains diversity, **+18.3/22.8% Pass@32 AIME24/25**. Pairs with the learnability-band sampler.
- **EMA checkpoint + rollback** — val degrades K steps → roll back to EMA weights, not latest.
- **Adaptive grad-norm clipping** (vs fixed max_norm) + fp16 loss-scale guard (reduce-not-abort on NaN).
- **Mix-entropy monitor** on data loaders (stop the mix collapsing to the easiest/most-rewarded source).
- **KILL the false comforts** (doc, correct + matches our co-train lesson): "KL penalty prevents forgetting" (it does NOT — we forgot anyway with a KL/LoRA constraint), "RM accuracy is enough", "fixed entropy coeff is fine", "static held-out prevents overfitting".
- ⚠ Most of the doc's ~25 cited papers are UNVERIFIED (the "Anti-Ouroboros" claim has NO arXiv ID — don't rely on it); use a specific paper's numbers only after a hub-check. PAR `2502.18770` (bounded reward) is already in §4.5.

## 5. Build queue (free, on Colab/Kaggle — do BEFORE renting GPUs)
0. **Pass@k capability map** (§0.5) — `eval_code_bench --passk` on the v1 RLVR model. **Run first** — it decides selection-vs-capability.
1. **Math RLVR**: `load_math_tasks` (GSM8K/MATH) + math verifier (port `rlvr.py:MathReward`: extract `\boxed{}`/last
   number, compare; add sympy for MATH symbolic) → `grpo_train --data math`.
2. **Reasoning RLVR**: a verifiable reasoning task loader + answer-match verifier (multiple-choice / short-answer).
3. **Broad eval harness**: extend `eval_code_bench` → `eval_battery` (HumanEval, MBPP, LiveCodeBench, GSM8K, MATH,
   GPQA/MMLU-sub) with one report.
4. **Curriculum / difficulty calibration**: a quick "base pass-rate per task" pass to pick the learnable band per
   domain (avoids the v1 too-easy/too-hard miss).
5. **Strong-teacher broad SFT data**: `make_traces` over code+math+reasoning via API teacher.
All five smoke on a 1.5B for $0; then the real runs go to RunPod.

**Existing assets (AUDITED 2026-06-21 — the build is mostly WIRING, not writing):** the v5 mechanism modules are
already in the repo **and unit-tested** — `curriculum_rl.py`, `best_of_n_reranker.py`, `self_refine.py`,
`cot_verifier.py` / `reasoning/step_verifier.py`, `contrastive_cot.py`, `data_flywheel.py`, `process_reward.py` /
`process_reward_model.py`, `token_credit_assignment.py` / `token_level_rl.py`, `self_play.py`, `src/multiagent/`
(**182 targeted tests PASS**, 2026-06-21). So §4.5's "smarter RLVR" levers are largely *wiring these to the
verifier-native loop*, not building from scratch — the prereg build queue above under-counted existing assets.
⚠ **But unit-tests-pass ≠ end-to-end-pipeline-works:** the integration (verifier→token-credit→RL update;
curriculum→GRPO rollout w/ variance gate; PRM→RL; the flywheel's canonical pipeline) is the *actual* remaining
work — per the v5-deep-research-update's own "Status:" lines. Smoke each wiring before relying on it. Caveat: heavy
`build/lib`+`src`+`aurelius/` duplication — use the `src/` copies (canonical); `aurelius/alignment/process_reward_model.py` is a 1-line stub.

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
