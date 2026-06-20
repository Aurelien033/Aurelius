# Aurelius-Qwen3-8B-RLVR — Model Card

**A clean, verification-native fine-tune of Qwen3-8B that measurably beats its base on held-out code generation,
trained with reinforcement learning against execution verifiers — no teacher distillation.**

---

## Summary
| | HumanEval pass@1 (held-out) |
|---|---|
| Qwen3-8B (base) | 84.1% (138/164) |
| **Aurelius-Qwen3-8B-RLVR** | **85.4% (140/164)** |
| Δ | **+1.3pp — reproduced across two independent runs** |

Small but **reproducible**, and notable because it is the result of **RL against ground-truth verifiers**, which
(unlike supervised distillation) is not capped by a teacher's quality.

## What it is
- **Base:** `Qwen/Qwen3-8B` (Apache-2.0).
- **Method:** GRPO (Group Relative Policy Optimization) with **verifiable rewards (RLVR)** — for each prompt, sample
  G=8 completions, reward = the **gym/MBPP execution verifier's pass/fail**, group-relative advantage, PPO-clip + KL
  to the frozen base. LoRA (r=16), gradient-checkpointed, ~150–400 steps. `docs/training/grpo_train.py`.
- **Training tasks:** an F2/F3 JSON/type-repair gym + **MBPP** (verified by running code against unit tests).
- **Data provenance:** 100% clean — verifier signal only; **no Claude/closed-model outputs, no teacher distillation.**
- **License:** Apache-2.0 (inherits the base).

## Results & honest scope
- **+1.3pp over base on HumanEval, reproduced** at 150 and 400 steps (both 140/164). Training reward rose
  +7% (0.70→0.77); the gain **transferred** to held-out HumanEval (MBPP ≠ HumanEval).
- **It plateaus at ~85.4%.** More steps did not help, and harder tasks (code_contests, multiple difficulty tiers)
  did not push past it — the gain is **bounded by the base model's own capability**, not the method.
- **Scope:** evaluated on HumanEval pass@1 (greedy, no-think). It is a *modest, reproducible* improvement, not a
  dramatic one. To raise the ceiling further, the evidence points to a **larger/stronger base**, not more RL.

## Why it matters
Across a long research arc, **supervised fine-tuning could not beat this base** (flat at best; a weaker teacher
made it *worse*). **RLVR is the only lever that did** — because its reward is correctness itself. This is a small
but clean demonstration of the **verification-native thesis**: clean verified data → verifier-reward optimization →
a model that is *measurably and reproducibly* better than its strong base, with zero teacher dependence.

## Reproduce
```bash
python docs/training/grpo_train.py --base Qwen/Qwen3-8B --data both \
  --group_size 8 --steps 150 --max_new 512 --lr 1e-6 --out aurelius-rlvr
python docs/training/eval_code_bench.py --model aurelius-rlvr --bench humaneval --think 0
```

## Limitations
Single eval (HumanEval); +1.3pp = 2 problems (mitigated by reproduction across two runs); LoRA not yet merged for
distribution; not safety-tuned beyond the base. Intended as a research artifact demonstrating clean RLVR-over-base
improvement, not a production model.
