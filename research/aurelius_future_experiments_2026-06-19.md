# Aurelius — Future Experiment Menu
### 2026-06-19 · what to put in the next models, grounded in what the arc actually proved

> Curated, not a wishlist (the Hermes 18-mechanism scope-trap is the failure mode to avoid). Each item ties to
> a *specific* finding, with an honest prior and cost. Prioritized.

---

## 0. What the arc established (the priors these experiments inherit)

1. **Dynamic per-input layer routing is dead on a frozen base** (3 powered nulls: g_gain, per-task selector k=2 & k=4). Don't revisit at the per-layer-marginal rung.
2. **LayerDelta substitution works IN-DISTRIBUTION** (matches dense at less compute, scales to ~k=10) but **does not transfer out-of-family** → it's *compression*, not a general gain.
3. **Verified-trace SFT specializes, it does not generalize** (Qwen3-8B: +11.5pp on held-out repair, ≈flat on HumanEval). And **light SFT cannot broadly beat a strong 8B base** — imitation is capped by the teacher and by the base's existing coverage.
4. **The biggest accuracy lever was the base** (VibeThinker-3B 33% → Qwen3-8B 48% → +SFT 60% on repair). Using better open bases > marginal SFT.
5. **Narrow evals mislead** (the n=60 mirage; gym-specialization looked like a win). Rigor = full sets + held-out + breadth.

**The defensible identity that emerges:** Aurelius = **verification-native models** — *verified data → verifier-reward optimization → verifier-gated serving.* Every lever below either (a) exceeds the teacher via the verifier, (b) compresses, or (c) deploys verification. That's the coherent thesis the arc supports.

---

## 0.5. Existing assets (device + GitHub sweep, 2026-06-19) — much is already built

A repo sweep found **real, unit-test-passing** implementations of the core machinery these experiments need (the repo's history is sprawl-heavy, so these were verified: `pytest tests/alignment/...` = **69/69 pass**):

| asset | file | relevance |
|---|---|---|
| **RLVR** (verifiable-reward RL, n_samples=8) | `src/alignment/rlvr.py` | **E-1** |
| **GRPO v3** (DeepSeekMath, group-rel adv, PPO-clip+KL) | `src/alignment/grpo_v3.py`, `grpo_advanced.py` | **E-1** |
| **CoT / process verifier** (per-step + chain validity, `ProcessRewardModel`) | `src/alignment/cot_verifier.py` | **E-7** |
| Value head (GAE), reward models (ensemble/hierarchical/distill/multi-obj) | `src/model/value_head.py`, `src/alignment/reward_*.py` | E-1/E-7 |
| Rejection sampling, synthetic math, preference collection | `src/data/rejection_sampling_data.py`, `synthetic_math.py`, `synthetic_preference.py` | E-1/E-3/E-5 |
| Early-exit (dynamic depth) | `src/model/early_exit{,_v2}.py` | E-2 / routing |
| Absolute-Zero self-play RL | `src/alignment/absolute_zero.py` | E-1 advanced |
| Process-verify / judge-audit DESIGN specs | `~/Desktop/AI Plans/N9-PROCESS-VERIFY-SPEC.md`, `N7-JUDGE-AUDIT-SPEC.md` | E-7 |

**⚠ Critical caveat (the verify-before-trust rule):** these pass *unit* tests on tiny tensors and were built against the from-scratch `AureliusTransformer`, **never run as a real RL job and never wired to a HF base (Qwen3-8B) or the real code verifiers.** So the algorithms exist and are sound; the *real* work is **integration + a real validated run**, not implementation. This lowers E-1's cost from "build" to "wire + validate" — but the validation is exactly where this repo's scaffolding has historically been hollow, so treat it as a real (de-risked) experiment, not a free win.

---

## Tier 1 — the headline future levers

### E-1. RLVR (verifier-reward RL) — *the one lever that can BEAT the base*
- **Why now:** SFT imitation is capped by the teacher (R1) and the base's coverage (finding #3). **RL with the verifier as reward lets the model exceed both** — generate, the verifier scores, optimize. This is the proven recipe for verifiable tasks (R1-Zero/GRPO), and Aurelius *already has the verifiers* (gym F2/F3, MBPP/HumanEval execution). This is the natural, highest-value continuation.
- **Method:** GRPO/RLOO from the SFT'd model. Per prompt: sample G=8 completions → verifier reward {1,0} → group-relative advantage → policy-grad. On the gym + MBPP (verifiable). KL-anchor to the SFT model.
- **Prior:** HIGH. **Cost:** LOWER than first thought — `src/alignment/grpo_v3.py` + `rlvr.py` already exist and pass unit tests (69/69). Work = **integrate** (wire the existing GRPO loop to the gym/MBPP/HumanEval *execution* verifiers + a HF base like Qwen3-8B + the SFT'd policy) + one real validated run — *not* implement from scratch. ⚠ Unit-tested only, built for the from-scratch `AureliusTransformer`; the HF-base + real-verifier integration is the actual work, and where to apply the verify-before-trust discipline.
- **Success:** beats SFT *and* base on the verifiable evals, and — the prize — transfers to HumanEval (RL-discovered solutions generalize better than imitated ones).
- **Folded in from the 2026-06-19 triage:** (a) **certificate-gated rollouts** (Batch-3/4DEE "no certificate, no training") — a trace counts only with provenance + no-reward-hacking + dense-comparison checks; this is the **reward-hacking guard** RLVR needs, wired into the reward fn. (b) **CID-DPO objective** — shape the reward toward the *cheapest verifier-passing* answer (correctness-per-compute), not just pass/fail. (c) adopt the **`state→action→check→verdict→commit/repair/fallback` grammar** as the rollout/data schema.

### E-2. LayerDelta compression on the shipped model — *the arc's one positive, deployed*
- **Why now:** the only clear positive (in-distribution structured substitution = dense quality at less compute). Turns Aurelius into a real **efficiency** story regardless of the accuracy outcome.
- **Method:** `layerdelta.py`/`_frontier` on the shipped 8B over its serving distribution; substitute k layers with rank-r surrogates; measure pass@1 retention + real decode speedup.
- **Prior:** MOD-HIGH (validated at 1.5B; confirm at 8B). **Cost:** low (hours).
- **Success:** substitute ~4–8 of 36 layers, match quality, ~15–30% faster decode → a deployable efficient release.

---

## Tier 2 — cheap, sharp, do-alongside

### E-3. Reasoning-trace SFT (distill the *reasoning*, not just answers)
- **Why:** v1/v2 SFT trained on verified CODE only; specialization didn't transfer (finding #3). Distilling R1's *reasoning process* (think+code), eval'd *with* thinking, may transfer to novel problems.
- **Method:** store full R1 traces in `make_code_traces`; SFT thinking-aware; eval `--think 1` big budget. **Prior:** MODERATE. **Cost:** low (reuse pipeline).
- **Batch-3 variants worth folding in:** **CFT-SFT** (counterfactual failure-trace) — mine fail-vs-pass deltas into explicit *repair* supervision (pairs naturally with RLVR's negative rollouts); **CID-DPO** (cheap-invariant DPO) — the principled upgrade of "real DPO pairs": prefer the cheapest correct answer.

### E-4. Best-of-N self-verification at inference (a deployment quality knob)
- **Why:** the arc *saw* verifier-assisted multi-try beat single-pass (k2 best-of-top5 ≥ dense). At deploy: sample N, verify (or self-verify), return the passing/best one — quality with **no retraining**.
- **Method:** inference wrapper: N samples → verifier/scorer → best. Measure pass@1 vs single-pass at N× compute. **Prior:** HIGH. **Cost:** low (inference-only). **Caveat:** N× compute + needs a verifier/scorer.

### E-5. Broaden the eval battery (rigor infrastructure)
- **Why:** narrow evals misled us twice (finding #5). **Method:** extend `eval_code_bench` to MBPP-held-out + BigCodeBench + a GSM8K math verifier → a trustworthy battery every future model runs. **Prior:** N/A. **Cost:** low. Highest-ROI *meta* work — it protects every other result.

---

## Tier 3 — bigger bets / closure

### E-6. Skip-native pretraining (the thesis-closure bet) — BUILT, never run
- The one remaining dynamic-routing revival path (OBL-063, launch-ready). **Prior: LOW** (3 nulls). Do once, deliberately, for closure or a low-odds revival — not urgent.

### E-7. Process reward model / step-verifier (the N7/N9 specs)
- Train a verifier that scores reasoning *steps*, not just final answers → enables process-RL and sharper best-of-N. **`src/alignment/cot_verifier.py` already implements this** (`ProcessRewardModel`, per-step + chain validity, passes tests) and `N9-PROCESS-VERIFY-SPEC.md` is the design. So again: integrate + get step-labeled data, not build. **Prior:** MODERATE. **Cost:** real (the step-labeled data is the bottleneck). The advanced form of the verification-native thesis.
- **Training-data side (Batch-3 VCSS):** generate the step-labeled data as *verifier-causal micro-steps* (each step carries a check) instead of long-CoT mimicry — that's both the PRM training set and a cleaner SFT target.

---

### E-8. Adaptive-compute / AMPLIFY probe — scout-disagreement-gated recursion (NEW, 2026-06-19)
- The `research/anticipatory-scout-transformer` packet: lightweight scouts run ahead, committee **disagreement** drives **per-token loop depth** (the AMPLIFY rung, not SKIP). High-quality, falsifiable (σ_max kill-tests, bootstrap-CI gates, adversarially reviewed). Smokes ran locally: A0 sound, guards correct, **Round E roofline already shows the FLOP→speedup/attention-coupling wall**. **Prior: GUARDED** — its Stage-0/1 crux ("cheap signal predicts where compute helps") is the same genus our SKIP nulls answered NO for; different signal+lever so not pre-falsified. **Cost: low to falsify** (Stage 0/1 = a few GPU-h on a tiny LM, decisive). **Rule:** Stage 0/1 must clear its CI-lower-bound gate on real data, then beat Round E's systems gate — else stop. Behind v2→RLVR. Full analysis: `anticipatory_scout_analysis_2026-06-19.md`.

---

## Explicitly SKIPPED / closed (do not build) — from the 2026-06-19 triage
- **Route-conditioned SFT family — RCSFT, JRM (joint route masks), TLA (token-level arbitration)** (SFT Batch-3 P1/P4/P5). These re-pose dynamic layer routing at a different granularity, but we **already ran the exhaustive k=2 pair matrix + the per-task selector at k=2 *and* k=4 — all powered nulls.** Re-falsifying costs 20–80 GPU-h for no new information. If routing is ever revived it's via **E-6 skip-native**, not route-conditioned SFT.
- **The full 4D-Elasticity × Expansiveness framework** (VECS/NEVS/elastic-dims/Collapsometer/StabilityShield/MUD-LDP-CCAE-…). 15+-mechanism scope-dilution built on the falsified ACDT SKIP/AMPLIFY thesis; the series itself files it as a "future reservoir." Kept only its 2 grounded kernels (certificate-gated data → E-1; falsifier discipline → already standard). Promote a piece only when it earns its own falsifier + need + budget.

## The sequencing call
**E-2 + E-5 now** (cheap, sure, protective: an efficiency win + a trustworthy eval battery), **E-1 (RLVR) as the next real bet** (the only lever that can beat the base — the headline future model), now with certificate-gated rollouts + the CID-DPO objective folded into its reward design. E-3/E-4 fold in cheaply (CFT-SFT/CID-DPO/VCSS live here). E-6/E-7 are deliberate, later. The through-line: stop trying to out-imitate strong bases with SFT; **compound the verifier** — into the reward (E-1), the decode budget (E-4), and the loss (E-7). That's the version of Aurelius the evidence actually supports.
