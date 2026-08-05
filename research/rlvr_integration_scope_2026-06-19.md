# RLVR Integration Scope (E-1) — using grpo_v3.py + rlvr.py
### 2026-06-19 · exactly what it takes to run real verifier-reward RL on Qwen3-8B

> Read of `src/alignment/grpo_v3.py` (303 ln) and `src/alignment/rlvr.py` (329 ln). Both pass unit tests; both
> were built for the from-scratch `AureliusTransformer`, never run on a HF model or a real verifier. This scopes
> the gap honestly.

---

## 1. What already exists (reuse as-is)

**`grpo_v3.py` — the loss core (correct, tested, keep):**
- `GroupRewardNormalizer.normalize(rewards (B*G,))` → group-relative advantages.
- `GRPOLoss(log_probs, old_log_probs, ref_log_probs, advantages)` → asymmetric PPO-clip + low-variance KL. **Sequence-level** (one log-prob per completion). Cleanly separates *old* (behaviour) and *ref* (KL anchor).
- `GRPOTrainer.grpo_step(...)` → normalize→loss→backward→step.

**`rlvr.py` — the loop skeleton + reward classes (reuse the structure):**
- `RLVRTrainer.train_step(prompt_ids, prompt_text, ground_truth)` — the full rollout→reward→logprob→update loop.
- `_recompute_log_probs` — teacher-forced per-token log-prob extraction (the right idea).
- Reward classes: `VerifiableReward` (base), `MathReward`, `FormatReward`, `CompositeReward`, `MemoryGroundingReward`.

**Net:** the *algorithm* is done. `grpo_v3`'s loss is the one to use (more correct than `rlvr.py`'s, which conflates old≈ref).

---

## 2. The integration gaps (the real work, in priority order)

| # | gap | why it blocks | fix |
|---|---|---|---|
| **1** | **HF model interface** | `_, logits, _ = model(ids)` (AureliusTransformer 3-tuple) **breaks** on Qwen3-8B (returns `.logits` object) | thin adapter: `logits = model(input_ids=ids).logits` everywhere |
| **2** | **generation is a hand-rolled token loop** (`sample_completions`) — no KV cache, 1 fwd/token | impractically slow for 8B × 256 tok × G=8 × many prompts | replace with `model.generate(do_sample=True, temperature, num_return_sequences=G, use_cache=True)` |
| **3** | **no code reward** — only Math/Format | the whole point is *code* correctness | new `CodeReward(VerifiableReward)` wrapping the gym `R.VERIFIERS` and `eval_code_bench.run_program` (MBPP/HumanEval) → {1.0 pass, 0.0 fail}, optional +format partials |
| **4** | **log-prob extraction at 8B scale** | 3 fwd passes (policy-grad, ref, old) on B*G seqs of an 8B → big activation memory | LoRA (only adapter trainable) + gradient checkpointing + small G / micro-batch the group |
| **5** | **reference model = 2nd 8B in memory** | OOM | reuse the `dpo_train.py` trick: ref = policy with `model.disable_adapter()` (no second copy) |
| **6** | **old vs ref** | `rlvr.py` uses ratio vs *ref* (conflates behaviour+anchor); textbook GRPO needs *old* = generation-time policy | on-policy single update: `old_log_probs = policy_log_probs.detach()` captured at rollout; `ref` = disable_adapter. Use `grpo_v3.GRPOLoss` which takes both. |
| **7** | **throughput / batching** | one prompt/step is slow; group = G completions of *one* prompt | batch several prompts × G per rollout; (later) vLLM for generation if HF `generate` is the bottleneck |
| **8** | **reward hacking / stability** | RL finds degenerate wins (length, format gaming) | length penalty, KL coeff (start 0.04), grad-clip, reward = pass-rate only first, add shaping later |

---

## 3. Recommended design — `docs/training/grpo_train.py` (new, ~300 ln)

A thin orchestrator that **reuses the tested loss** and fixes the 8 gaps:

```
load Qwen3-8B (+ SFT adapter as the start) as LoRA policy; ref = same, disable_adapter
for each step:
  sample G completions/prompt via model.generate (gap 2), for a micro-batch of P prompts  -> P*G seqs
  reward = CodeReward(gym/MBPP verifier) per completion (gap 3)            -> (P*G,)
  logp     = teacher-forced fwd on policy (grad, grad-ckpt)               -> seq log-prob (P*G,)  (gap 1,4)
  old_logp = logp.detach()                                                (gap 6)
  ref_logp = teacher-forced fwd with disable_adapter (no grad)            (gap 1,5)
  adv = GroupRewardNormalizer(G).normalize(reward)                        # reuse grpo_v3
  loss = GRPOLoss(cfg)(logp, old_logp, ref_logp, adv)                     # reuse grpo_v3
  loss.backward(); clip; opt.step()
  log mean_reward (the curve that must go up)
periodically: eval pass@1 on held-out gym + HumanEval (eval_code_bench)
```

Reuse: `GRPOLoss`, `GroupRewardNormalizer` (grpo_v3, tested), `VerifiableReward` base (rlvr), `run_program`/`R.VERIFIERS` (eval harness), the LoRA+grad-ckpt+disable_adapter patterns (sft_train/dpo_train). **New code is mostly the HF rollout + CodeReward + the loop.**

---

## 4. Effort, risk, smoke

- **Effort:** ~1–2 focused days. The loss/normalizer are done+tested; ~300 ln of orchestrator + a code reward.
- **Top risks:** (a) **throughput** — HF `generate` for G×P×steps is the bottleneck; mitigate with batching, then vLLM if needed; (b) **memory** — 8B × 3 passes; mitigate LoRA+ckpt+small G; (c) **reward hacking** — watch for length/format degeneracy; (d) the usual: it's never been run for real, so **verify the reward curve actually rises on a smoke before trusting**.
- **Smoke (cheap, decisive):** Qwen2.5-1.5B, G=4, ~16 gym prompts, CodeReward = gym verifier, 50 steps → **mean_reward must increase**. If it does, scale to Qwen3-8B + gym+MBPP.
- **Data:** start where verifiers exist — the gym (F2/F3) + MBPP (execution). HumanEval stays held-out for eval only.

---

## 5. Why this is the bet worth making
SFT (v1/v2) is capped — it can only imitate the teacher and can't beat a strong base (the arc's finding). **RLVR is the one lever that can exceed both**, because the reward is *ground-truth correctness*, not a teacher's tokens. The algorithm is already implemented and tested; the gap is integration to a real model + real verifiers — bounded, ~2 days, and decisively testable by one smoke (does mean_reward rise?). This is the highest-value next build for a genuinely-better Aurelius.
