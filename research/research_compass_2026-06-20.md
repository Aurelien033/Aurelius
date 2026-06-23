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

## 🚀 Paths to exceed +3–7pp (the "+20%" ambition — how to NOT stay capped)
The **+3–7pp bound is specific to *post-training a FIXED 8B, single-pass pass@1*** — and it's a ceiling because **RLVR = ELICITATION, not ACQUISITION** (it sharpens/reweights pass@k mass into pass@1; it does NOT add capability the base lacks). So no post-training *algorithm* on a frozen 8B breaks it much — the capped thing is the base's reachable set, not the optimizer. **+20% IS reachable — by leaving that regime.** Three real levers, EV-ordered:
1. **CAPACITY (proven).** 8B→10B (grow: SOLAR depth-upscale / sparse-upcycle, roadmap §3) or native Qwen3-14B = **+10–15pp from capacity**, *then* RLVR on top. The v7 mechanism-zoo doc itself conceded "+20% = 14B-equivalent" → so just use the bigger base. Most reliable route.
2. **INFERENCE-TIME + FLYWHEEL (compounding, verification-native).** best-of-N + repair + verifier (+effective pp) → **distill the search back into pass@1** → repeat. This ADDS the search's discoveries to the model = the one genuinely-novel "make single-pass beat the base ceiling" lever; it COMPOUNDS over cycles. **Wednesday's pass@k diagnostic MEASURES this headroom** (= LIVE #1).
3. **FROM-SCRATCH arch+data (the real R&D bet).** SuperBPE (+4% verified) + BLT + GatedDeltaNet + *clean curated data* + skip-native = a model BUILT better, not post-trained. Gated ($150k+/months, roadmap §3 grounded-arch pool), but where "genuinely beat the inherited base" lives.

⚠ **"+20%" is benchmark-dependent:** HumanEval is at 84% (only 16pp to the ceiling → +20% near-impossible THERE). Aim the ambition at **LiveCodeBench (~60%) / MATH (~50%) / competition reasoning**, where there's real headroom.
**Why this is here:** the mechanism-zoo stream (v5/v6/v7, M1741–M2006) chases +20% via *unvalidated post-training stacking on a frozen 8B* = the ONE regime that's capped (= same ambition, WRONG lever). High-EV research = **flywheel (cheap, compounding) → capacity (proven) → from-scratch (ambitious)** — not mechanism-coining.

## 🔴 DON'T re-chase (settled — we paid for these)
- **Dynamic per-input layer routing** — 3 powered nulls (g_gain, k=2 & k=4 selectors). Dead on a frozen base.
- **From-scratch pretraining / raw frontier (Opus/Qwen-Max) parity** — $0.14M–$32M, months–years. Infeasible; and
  parity-in-domain comes from the *system*, not the checkpoint.
- **The 4D-Elasticity mechanism-zoo / integrating unvalidated modules** — scope dilution; promote a piece only after
  it has its *own* falsifier + need + budget.
- **SFT from a teacher *weaker* than the base** — it drags the base *down* (v2 R1-7B: 84→76%). Teacher must be stronger.
- **Re-running the gym+MBPP RLVR plateau** — 85.4% is the ceiling there; needs harder *learnable-band* tasks, not more steps.
- **AMC / composer / token-efficient-recall (v3+ memory-agent docs, 2026-06-21)** — aspirational long-context/memory-agent architecture. **`CTAR` (per-token *per-layer* cross-tier routing) = the SAME dynamic per-token/per-layer routing genus that's been a powered null 3×.** Tag **v3+ aspirational**, keep OFF the v2 capability critical path (the compass's mechanism-zoo trap). Grounded sliver = local KV-cache budget math, already covered by the inference-cost roadmap. NOTE (positive, 2026-06-21): the *v5-deep-research-update* is the disciplined exception — its named v5 modules ARE in the repo + unit-tested (182 pass); that's wiring work, not a scope-trap.
- **Layer-4 CPS / Counterexample-Proof-Search (M2049–M2066 mechanism-zoo, 2026-06-22)** — register-don't-adopt: ~10 self-coined mechanisms, NO external citations, optimistic stacked +10pp. It RE-DERIVES the verification-native plan (certificates/adversarial-tests/synthetic-gym already covered) and its kill-decisions match ours (blind self-correction, generic debate, latent-too-risky) = convergence signal, not new info. **3 refinements EXTRACTED → prereg:** stepwise decision-point preference distillation (M2052), critique-effectiveness training (M2059), solver/verifier disagreement harvesting (M2053). REJECT: the wholesale CPS program (scope-dilution), the M-IDs (Line-B re-mints per §145.7 alias rule), and heavy inference-time tree/MCTS search (compute + report-separately).
- **Collapse-prevention (M2101–M2110) + evidence-portfolio (M2002–M2031) docs (2026-06-22):** **collapse-prevention = USEFUL-EXTRACT** (targets our REAL failure — co-train 73%→22% forgetting; hooks into real `dr_grpo.py`/`adaptive_sft.py`; cheap stability guards **FOLDED → prereg §4.6**: ⭐capability-retention-eval / entropy-monitor / **SvS `2508.14029`** / EMA-rollback / adaptive-grad-clip; KILL false comforts like "KL prevents forgetting"). **evidence-portfolio = register-don't-adopt** (re-ranks already-verified frontier levers + winner-only inside a Phase-0–5 / CPS-scoring / impossibility-landscape framework = analysis-theater); ONE new lever EXTRACTED → §4.5: **CodeRM `2501.01054`** (dynamic unit-test scaling, verified). Both docs' M-IDs = Line-B mints (don't adopt); the collapse doc's ~25 papers are mostly UNVERIFIED (hub-check before using specific numbers; "Anti-Ouroboros" has no ID).
- **Multi-turn / conversational collapse (40-rounds doc: IG-KV / Sheaf-Consistency CSV-SC / drift-corrector / error-cascade / M2111–M2119 + M2101-R…M2110-R re-mints, 2026-06-22)** — register-don't-adopt: the single-turn collapse guards are ALREADY in prereg §4.6 (this doc just re-mints them + adds UNAUDITED code stubs); the genuinely-new mechanisms are all MULTI-TURN / episodic-state = **v3+/agentic** (AMC/composer bucket, NOT the v2 single-turn capability path). "Sheaf consistency" for conversation = novelty-for-novelty. EXTRACTED → §4.6: **STEER `2510.10150`** + **DSDR `2602.19895`** (verified single-turn entropy-control formulas).
- **Layer-6 frontier-deepening (E601–E610, 2026-06-22) = top-3 SURVIVED (6/6 IDs verified), broad-10 = v3+.** ⭐**E601 PRM min-form** (PURE `2504.15275` NeurIPS'25 + PAPO `2603.26535`: min-form credit + decoupled-norm prevents reward-hacking collapse — sum-form "collapses training at the start") → FOLDED prereg §4.5 (`process_reward_model.py` upgrade, ~3 days). **CurES `2510.01037`** (gradient-analysis curriculum, beats GRPO +3.3/+4.82) → §4.5 learnability-band. **TREC `2509.25380`** (data-placement via AdamW EMA) → cleanup-spec CPT-ordering note. **E603 Arbitrage `2512.05033` + SpecKV `2605.02888`** (advantage-aware spec-decode ~2×) → serving roadmap (NOT capability). REGISTER-DON'T-ADOPT the broad 10 (E604–610: world-models / meta-learning / emergence-forecasting / circuit-steering / dynamic-routing = v3+ speculative); E-IDs = Line-B.
- **v5/v6/v7 mechanism-zoo STREAM (2026-06-23, ~9 docs, M1741–M2006 ≈27 self-coined mechanisms) = REGISTER-DON'T-ADOPT the WHOLE STREAM** (fence once, don't re-triage each). Tells: **ZERO arXiv citations** in 997 lines (vs survivors' 6–9/9 verified); **+20% headline = the optimistic-stacking fallacy** ("+24% compound, each mechanism makes others 5–15% better" — no evidence; our verified bound is +3–7pp@8B, no +20pp on 8B). **Refutes itself:** its own caveat admits "105% = matching the 14B baseline, not exceeding" → the 27-mechanism / 6–12-month program just reaches **Qwen3-14B-equivalent = a 1-week native fine-tune**, so it's backwards (27 unvalidated mechanisms vs one download). Real ideas already covered (M2001 failure-induction = failure-ledger; M2006 = self_play.py; M2002 ≈ learnability-band) or v3+ scope-traps (M2003 verifier-head steering / M2004 meta-cog routing / M2005 SAE-CoT). **Plan SATURATED — 13th external doc, still register-don't-adopt; the productive move is EXECUTING Wednesday's pass@k diagnostic, not more mechanism-coining.**
  - **v7-CITATIONS-APPENDIX checked 2026-06-23: 9/9 IDs are REAL papers BUT post-hoc loose/keyword mappings** (e.g. an HCI-governance paper `2603.21735` cited for "cognitive load"; a safety-CoT-eval `2603.05706` cited for "activation steering") — they do NOT ground the self-coined mechanisms = **citation-laundering; stream STAYS fenced.** Also note: the appendix even cites `2601.05280`, which *refutes* its own self-play optimism. **4 real SLIVERS extracted → prereg:** SePT `2510.18814` (reward-free self-train) + ⭐self-improving-limits `2601.05280` (flywheel degenerates w/o fresh external signal) → §0.7 flywheel; EWC-DR `2603.18596` (anti-forgetting) → §4.6; Overthinking `2604.10739` (test-time-compute caveat) → §4.5. Real-but-not-folded: step-SAE `2603.03031` (v3+ interp), Hi-CoT `2604.00130` (mild prompting).
- **VIMPO + R-SWA experiment-spec (2026-06-25) = 5th SURVIVOR (both IDs verified real + accurate; a disciplined 2-mechanism spec, NOT the zoo).** **VIMPO `2606.20008`** (critic-free Value-Implicit Policy Optimization — policy-implied value from KL-RL optimality, beats group-relative credit w/o a critic) → FOLDED §4.5 as a GRPO A/B candidate (`dr_grpo.py`; one of several GRPO-beaters, not a unique unlock). **R-SWA `2606.23050`** ("Unlimited OCR Works", bounded-KV sliding-window attention) → serving roadmap §2.6c; ⚠ OCR method — training-free SWA on full-attention Qwen3 risks quality → falsifier-gated, serving-not-capability. The VIMPO+R-SWA "coupling" is ORTHOGONAL (training vs serving) — evaluate separately, no synergy.

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

**RLVR *efficiency* (verified):** **two-rollout GRPO** `2510.00977` (32▲, "Your GRPO Is Secretly DPO") — G=2 matches G=8 at ~4× less rollout compute → test `--group_size 2`. Also SFPO `2510.04072`, SPPO `2604.08865`, iGRPO `2602.09000`. **Debunked:** the v3 doc's "BPPO 6× GRPO speedup" is a misattribution — real BPPO `2302.11312` is *offline-RL control* (D4RL/MuJoCo), unrelated to LLM GRPO.

**Brainstorm arXiv IDs — FULL direct-arXiv verification (2026-06-21; SUPERSEDES the earlier "phantom" guess).** 18 IDs from the v5/capability-beyond-base docs checked: **15 REAL + correctly labeled, 3 bad.** The `continuous-run` scanner is MORE reliable than assumed — its scramble bug hits ~1-in-6, not "most" (correct the over-skewed prior).
- **RLVR / learnability (verified, directly relevant):** **SC-SDPO** `2605.27765` ("Restoring the Sweet Spot," pass-rate-weighted self-distill = learnability sweet-spot, **+3.2/+4.3 Qwen3-8B** — *my earlier 'phantom' call was WRONG*); **DIVA-GRPO** `2603.01106` (difficulty-adaptive); **Self-Verified Distillation** `2605.26132` (Qwen3 self-flywheel, no teacher); **SPOC** `2506.06923` (single-pass solve+verify, +8.8pp MATH500/+10pp AMC23). Also the model-free-search-surfaced **VADE** `2511.18902`, **D³S** `2509.22115`, **SAGE** `2602.03143`, **SRPO** `2604.02288`.
- **Serving / efficiency (verified):** Speculative Thinking `2504.12329`; Domino `2605.29707` / Draft-OPD `2605.29343` (spec-decode ~5×); MiniMax Sparse Attn `2606.13392`; DynamicPTQ `2606.12487` / HARP `2605.29843` / LFQ `2605.29756` (≤10B-deploy quant); CIRF `2605.28292` / MemoSight `2604.14889`; Adaptive Minds/LoRA-as-tools `2510.15416`; self-correction paradox `2601.00828`.
- ⚠ **BAD (3):** **RUBAS** `2606.04051` is real but = *agent-safety* rubrics NOT code-reward (mislabeled application); **AlphaQ** `2606.04971` → a fairness-agents paper (WRONG id); **TWLA** `2606.13024` → "CausalMoE" (WRONG id). Hub-check a specific ID before building, but the scanner isn't "mostly junk."

**Frontier deep-dive verified (2026-06-22, `aurelius_frontier_research_deep_dive`): 9/9 load-bearing IDs REAL + accurate** (best-sourced doc yet). **ADOPT** (fit fine-tune+RLVR, folded into prereg §4.5/Stage-1): **Art of Efficient Reasoning** `2602.20945` (Qwen3-validated reward design + positive-reward-density), **GDRO-GRPO** `2601.19280` (adaptive learnability-band), **DLCoT** `2503.16385` (teacher-distill: keep error→correction pairs), **PAR** `2502.18770` + **Hack-Verifiable** `2605.20744` (reward hygiene), **R³** `2402.05808` (reverse curriculum). ⚠ **SKIP (wrong fit):** procedural-warmup `2601.21725` + curriculum-dynamics `2601.21698` = PRETRAINING-stage (we don't pretrain); MoE-routing-distill (ExpertFlow `2410.17954` + DeepFusion `2602.14301`/VAA) = unpublished gamble, Phase-3, DeepFusion is federated-edge repurposed; Latent-GRPO `2604.27998` = arch-change; sleep-time `2504.13171` = AMC-dependent serving. ⚠ **CAVEATS:** doc's "211GB corpus" = the license-toxic Yggdrasil scrape; base "Qwen3.5-9B" = a VLM (vision dead-weight — our text-native Qwen3-8B→14B is cleaner); its "+8–12pp" stacking is optimistic vs our +3–7pp@8B bound.

## The one filter for any new idea
Before building: **(1) is it grounded in a result, (2) does a concrete need pull it in, (3) does it fit the budget?**
And: **does it add a verifier, densify reward, or capture selection?** If not, it's probably a scope-trap.
