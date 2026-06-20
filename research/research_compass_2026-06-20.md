# Aurelius — Research Compass
### 2026-06-20 · where side-research actually moves the needle (and what NOT to re-chase)

> Grounded in the full arc. The through-line: **every good Aurelius research question is "where can we add a
> verifier, densify the reward, or capture selection?"** Stay verification-native — that's the only thing that worked.

---

## 🟢 LIVE questions — ranked by value × tractability

1. **Selection vs capability → distill-the-search.** Does the model already *sample* correct answers it doesn't
   *pick* (greedy)? If yes, best-of-N + verifier → distill winners back to greedy = a **single-pass gain for ~free**
   (no bigger base). The cheapest big win. *(Wednesday's `bestofn_repair.py` starts this.)* Read: test-time-scaling
   / best-of-N-with-verifier, pass@k→pass@1 distillation.
2. **RLVR signal density.** The v1 plateau = sparse terminal reward + equal-reward groups (no gradient). Which fix
   helps most: **winner-only / positive-advantage GRPO** (does penalizing failures hurt?), **learnability-band
   curriculum** (auto-keep tasks at ~20–50% pass), **execution-grounded credit** (reward where code first diverges),
   **repair reward** (see the error, fix it)? Read: GRPO variants, WAPO, process rewards.
3. **Verified data flywheel.** best-of-N + repair → store winners (SFT) + winner-vs-near-miss (DPO) → distill →
   repeat. *How much does it compound before saturating?* This is the self-improvement engine.
4. **Multi-domain verifiers** (for the broad model). Math = answer-match (`rlvr.py:MathReward` exists; add sympy for
   MATH). Reasoning = short-answer/multiple-choice match. **The verifier design is the crux** — a sloppy verifier =
   reward hacking. Read: RLVR on math (DeepSeekMath/R1), verifiable-reward design.
5. **Reward-hacking defenses.** Generated/differential test expansion, anti-trivial-pass, certificate-gated rewards
   (provenance + no-hack + dense-comparison). Protects every RLVR run as you densify reward.
6. **Bigger base = the raw-capability ceiling** — proven lever (VibeThinker 33→ Qwen3-8B 48→60). But **last**:
   exhaust 1–5 (cheap) before paying for 14B/32B.

## 🔴 DON'T re-chase (settled — we paid for these)
- **Dynamic per-input layer routing** — 3 powered nulls (g_gain, k=2 & k=4 selectors). Dead on a frozen base.
- **From-scratch pretraining / raw frontier (Opus/Qwen-Max) parity** — $0.14M–$32M, months–years. Infeasible; and
  parity-in-domain comes from the *system*, not the checkpoint.
- **The 4D-Elasticity mechanism-zoo / integrating unvalidated modules** — scope dilution; promote a piece only after
  it has its *own* falsifier + need + budget.
- **SFT from a teacher *weaker* than the base** — it drags the base *down* (v2 R1-7B: 84→76%). Teacher must be stronger.
- **Re-running the gym+MBPP RLVR plateau** — 85.4% is the ceiling there; needs harder *learnable-band* tasks, not more steps.

## 📚 Worth reading (grounded, verification-native)
- **GRPO / RLVR:** DeepSeekMath `2402.03300`, DeepSeek-R1 / R1-Zero `2501.12948`.
- **Process reward / step verification:** "Let's Verify Step by Step" `2305.20050` (→ our `cot_verifier.py`).
- **Test-time scaling / best-of-N + verifier:** the inference-scaling line (Snell et al.), self-consistency.
- **RLVR-improvement specifics:** `~/Desktop/AI Plans/aurelius-rlvr-humaneval-benchmark-improvement-research-2026-06-20.md` (already folded into the v2 prereg).
- **Hard code benchmarks:** LiveCodeBench `2403.07974`, code_contests/AlphaCode `2203.07814`, EvalPlus.
- **The serving stack:** `~/Desktop/AI Plans/aurelius-inference-training-research-2026-06-18/` (91 sources, certified-substitution).

## 🎯 Verified v2-RLVR papers (HF paper_search, 2026-06-20) — each maps to a failure WE observed
*(The auto "continuous-run" arXiv scanner is buggy — scrambled title/abstract pairs — so it's NOT a source; these came from real search.)*
| our finding / lever | paper | what it adds |
|---|---|---|
| **pass/fail reward is sparse** (we only reward terminal pass) | **VeRPO** `2601.03525` | verifiable *dense* execution rewards for code (partial-success + global) ⭐ |
| **too-hard tasks → 0 reward → no gradient** (our code_contests miss) | **Scaf-GRPO** `2510.19807` | scaffolding/in-prompt hints overcome the "learning cliff" |
| **negative updates may hurt** (winner-only hypothesis) | **DISPO** `2602.00983` | decouple importance-clipping for correct vs incorrect responses |
| **equal-reward groups give zero gradient** (we hit this directly) | **DCPO** `2509.02333` (22▲) | dynamic clipping → more nonzero advantage; smooth adv standardization |
| **scalar group advantage is coarse** | **LamPO** `2605.21235` | pairwise-decomposed advantages (Qwen3-tested) |
| **broad multi-domain reward** (v2 code+math+reasoning) | **RGR-GRPO** `2511.12344` / **VPR** `2605.10325` | rubric / process dense rewards from oracles |
| **reward hacking / trivial pass** | **Posterior-GRPO** `2508.05170` | reasoning-quality reward reduces hacking in code RL |
| **plateau under more steps** | **Scaling-Up-RL** `2507.12507` | prolonged RL: controlled KL + periodic reference resets |
| token-level credit | **TEMPO** `2509.18314` (Qwen3-1.7B/4B) ; aggregation bias: **Balanced Aggregation** `2605.04077` |
**Top to implement in v2 (attack the exact plateau):** VeRPO (dense code reward) + Scaf-GRPO (hard-task scaffolding) + DISPO/DCPO (fix the zero-gradient equal-reward groups).

## The one filter for any new idea
Before building: **(1) is it grounded in a result, (2) does a concrete need pull it in, (3) does it fit the budget?**
And: **does it add a verifier, densify reward, or capture selection?** If not, it's probably a scope-trap.
