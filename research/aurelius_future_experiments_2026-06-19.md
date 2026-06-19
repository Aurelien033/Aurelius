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

## Tier 1 — the headline future levers

### E-1. RLVR (verifier-reward RL) — *the one lever that can BEAT the base*
- **Why now:** SFT imitation is capped by the teacher (R1) and the base's coverage (finding #3). **RL with the verifier as reward lets the model exceed both** — generate, the verifier scores, optimize. This is the proven recipe for verifiable tasks (R1-Zero/GRPO), and Aurelius *already has the verifiers* (gym F2/F3, MBPP/HumanEval execution). This is the natural, highest-value continuation.
- **Method:** GRPO/RLOO from the SFT'd model. Per prompt: sample G=8 completions → verifier reward {1,0} → group-relative advantage → policy-grad. On the gym + MBPP (verifiable). KL-anchor to the SFT model.
- **Prior:** HIGH. **Cost:** moderate (G× sampling × steps; hours–day on A100). Build `grpo_train.py` (reuse the verifiers).
- **Success:** beats SFT *and* base on the verifiable evals, and — the prize — transfers to HumanEval (RL-discovered solutions generalize better than imitated ones).

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
- Train a verifier that scores reasoning *steps*, not just final answers → enables process-RL and sharper best-of-N. **Prior:** MODERATE. **Cost:** real (needs step-labeled data). The advanced form of the verification-native thesis.

---

## The sequencing call
**E-2 + E-5 now** (cheap, sure, protective: an efficiency win + a trustworthy eval battery), **E-1 (RLVR) as the next real bet** (the only lever that can beat the base — the headline future model). E-3/E-4 fold in cheaply. E-6/E-7 are deliberate, later. The through-line: stop trying to out-imitate strong bases with SFT; **compound the verifier** — into the reward (E-1), the decode budget (E-4), and the loss (E-7). That's the version of Aurelius the evidence actually supports.
