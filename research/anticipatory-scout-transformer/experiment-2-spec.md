# Experiment 2 (Stage 2) — Does *loop*-disagreement predict where *more loops* help?

**The construct-valid test. The build gate.**
*(v2 — rewritten after adversarial review `wyvommwcf`, which judged v1 unable to
gate the build. Supersedes v1.)*
Companion to [research notes §11](anticipatory-transformer-research-notes.md) and
[Experiment 1 spec](experiment-1-spec.md).

---

## 0. What a GREEN earns — and the four conditions for it to transfer

Stage 1 tested the premise on a proxy axis. Stage 2 tests it on the **real axis**
(iterations of one tied block). But the review established that a naive Stage 2
**still won't transfer to the build** unless four conditions hold. A GREEN that
meets all four = green-light to build (with the R1 gate). A GREEN that doesn't =
green-light only to *prototype the anisotropic operator*, nothing more.

**Transfer conditions (all required):**
1. The signal survives on the **anisotropic** operator at the build's stability
   regime — not the isotropic stand-in (see §1, §4-tension).
2. It beats current-NLL **and** CALM-entropy on the **natural-language** arm, not
   only synthetic tasks.
3. The iso-cost policy simulation **charges the scout's own compute** and still
   wins on **measured wall-clock** (R1 quantified).
4. **Untrained truncated** scouts predict **converged** benefit (R7); else trained-
   scout cost is folded into the economic inequality and the result is at best YELLOW.

> **Strip rule unchanged:** no MoE/prefetch/halting-policy. But anisotropy and the
> ε regime are now **in scope** — they are exactly what makes the test valid.

---

## 1. The prototype (now anisotropic, with a swept ε)

- **Weight-tied recurrent block**, applied `k = 1..K`. `K` justified from the
  task's reasoning-step count; state the BPTT/implicit-diff scheme — and **verify
  late-loop readouts are not systematically under-trained** (truncated BPTT depresses
  `B(k)` at large `k` → false-RED).
- **Subtractive/residual update** `value_read = V_actual − α·V_explained`, run at
  the build's intended α/damping regime (over-relaxation/oscillation must be tested,
  not assumed away).
- **Stability — HEADLINE ARM IS ANISOTROPIC** (per [notes §10.4]): an SPD-metric
  operator, singular values **≤1−ε off-manifold**, **≈1 along a fixed symmetric
  metric**. The isotropic σ≤1−ε version is **diagnostic only** and can never produce
  a GREEN (see §4). **Sweep ε ∈ {0.02, 0.05, 0.1, 0.2, 0.4}; never toggle it.**
- **Skip path + reconstruction loss** so contraction doesn't erase information.
- **Training objective pinned & reported both ways:** B(t) needs a per-loop readout,
  which depends on the loss. **Deep supervision** manufactures monotone-help
  (false-GREEN); **final-loop-only** makes intermediate readouts garbage (false-RED).
  Report B(t) under **both**; require robustness across them. Fit the readout on a
  **document-disjoint** split.
- **≥3 seeds** on the headline arm (BPTT + spectral-norm + subtractive are unstable;
  one run can't set the verdict). Disagreeing seeds → INCONCLUSIVE.
- **Size:** few-M to tens-of-M params.

---

## 2. Testbed — synthetic primary, **natural-language now a gating arm**

- **PRIMARY (synthetic/algorithmic):** multi-step arithmetic, parity, ListOps,
  sorting, graph reachability, bracket matching — iteration provably helps,
  ground-truth reducible-vs-irreducible labels for free.
- **NATURAL LANGUAGE — now GATING, not "realism check":** a small char/byte LM with
  its **own pre-registered tail-metric threshold**. GREEN-to-build requires the
  cheap signal to beat current-NLL and CALM-entropy **on NL too**. If NL is
  underpowered at prototype budget, say so and **cap synthetic-only at YELLOW**
  ("signal exists where iteration provably helps; NL transfer unproven").
- **Precondition gate (concrete, numeric):**
  - (a) more loops monotonically reduce **average** loss; (b) per-token loop-benefit
    has real **variance**; (c) the **joint** loop converges — estimate the joint
    operator norm by **power-iteration on the full N·d Jacobian-vector product** (or
    block-Gershgorin per R2), report spectral radius/Lipschitz with a CI, require
    **joint ρ̂ ≤ 1−ε at the longest tested context** and the **joint** residual
    monotone-decreasing on a probe batch; (d) converged loss within a margin of a
    **matched-param non-tied depth-K baseline** (catches over-damped collapse cheaply).

---

## 3. Definitions

- **Compute axis = loop count `k`;** `B(t) = NLL(loop k) − NLL(loop k+1)`, marginal,
  **signed**.
- **Surprise — S1 is the PRIMARY gate; S2 confirmatory** (S1 is deterministic →
  cheaper, native to the design, dodges the ensemble confound and the build-cost
  multiplier):
  - **S1 — residual magnitude:** `‖residual‖` at the scout loop `k_scout`.
  - **S2 — truncated-trajectory committee disagreement** (relabelled — it is NOT
    "epistemic JSD"; at 1–2 loops it is *transient* variance, and the Fort-2019
    between-mode justification does **not** hold until earned by the §4.2 result):
    JSD across `m≥2` separately-initialized models (+dropout) at `k_scout`.
- **Difficulty `D`:** generator ground truth (synthetic) / reference model (NL).
  Diagnostic only.

---

## 4. Killing the residual tautology (scalar control is NOT enough)

`B` and `S1` are both deterministic functions of the **same hidden state through the
same readout**; partialling out a 1-D projection (current-NLL) leaves the
d-dimensional shared confound intact. The following are **hard gates**, not screens:

1. **Pin the control index** (was undefined/gameable): primary nuisance control =
   **NLL at `k_scout`** (what a live policy actually has); stricter second control =
   **NLL at loop `k`** (left endpoint of `B`). The signal must survive **both**.
2. **Truncated-loop-NLL control (C8):** add **loop-`k_scout` readout cross-entropy to
   gold** as a second mandatory nuisance control for **both** S1 and S2 (if readouts
   are deep-supervised, the committee disagrees exactly where the loop-`k` readout is
   far from gold → spurious).
3. **Future-loop decorrelation is PRIMARY:** S1(`k_scout`) must predict `B` at loops
   **far** from the scout (e.g. between `K−2` and `K`).
4. **Mechanical-channel removal:** S1's predictive power must survive after
   residualizing out the **state-step norm** `‖v_expl(k+1)−v_expl(k)‖`. If S1 predicts
   `B` only through step magnitude, that's the tautology.
5. **Mechanical null:** shuffle which token's residual pairs with which token's
   far-future benefit within a stratum; the gate must clear this null.
6. **Cross-readout control:** S1 must also predict `B` under a **held-out readout**
   (lens fit on a disjoint split), so single-readout geometry can't pass.
7. **Same-state negative control:** S1 from a *different* token matched on current-NLL
   must **fail** to predict `B`. (The `+1`-shift control is an **alignment** check
   only — it does not cover the tautology.)

---

## 5. Statistics & baselines (Stage-1 machinery, adapted)

- Cluster bootstrap on **independent instances/documents**; **GREEN = CI lower bound
  > a pre-registered numeric threshold** (§7); tail metric primary, global ρ coarse.
- **Binding tautology test is tail-based (C6):** the *orthogonalized*-S tail metric
  (allocate on S after regressing out current-NLL) with CI lower bound > 0 vs the
  current-NLL allocator — OR S beats current-NLL on the tail **within current-NLL-
  matched strata**. "Partial Spearman > threshold" is screening only.
- **FLOP/capacity-match the S2 head-to-head (C7):** compare S2 not just to single-model
  entropy but to an **m-model ensemble** mean-prediction entropy. If ensemble-entropy
  predicts benefit as well as committee-JSD, S2 is just a better model, not a better
  signal → YELLOW.
- **Do NOT disattenuate S1** (it's deterministic; reliability=1 trivially). Bootstrap
  S1 over documents + training seed; keep T-pass disattenuation for S2 only.
- ρ(S,B) only gate; ρ(S,D) diagnostic; aleatoric + positive controls with generator
  ground truth; multiplicity control; **pre-register S1 as primary** to avoid winner's-
  curse over {S1,S2}.

---

## 6. Ablations (≥3 seeds each, matched training budget, seed-level CIs)

- **Subtractive vs vanilla tied loop** — at the build's α regime. *If GREEN is S2-only
  and the residual update doesn't improve predictability, the build proceeds S2-only
  and S1/subtractive is dropped — record this explicitly (§7).*
- **Anisotropic vs isotropic** — the headline; isotropic is the diagnostic floor.
- **ε × reconstruction-weight sweep** — one grid (see §8).
- **Committee m=1 (dropout) vs m=2–3 (between-mode)** — does Fort-2019 diversity matter.

An ablation flips a design decision only if its effect CI excludes zero **and**
exceeds seed spread.

---

## 7. Decision rule (the build gate — numbers pre-registered)

**Pre-register before running:** (1) the minimum tail-metric advantage over **both**
baselines, derived from the iso-cost headroom the economic inequality needs, plus a ρ
floor; (2) the resampling unit (independent generated instances) and a **power calc
(≥80%)** to detect that margin; (3) a Stage-1-style **inconclusive band**: a null that
fails the minimum-resource bar (stated tokens, K range, committee size, **converged
precondition**, ≥3 seeds) is **INCONCLUSIVE, not RED**.

- **GREEN → build** requires ALL: (a) precondition met incl. joint convergence and the
  non-tied-baseline margin; (a0) ≥3 concordant seeds; (b) a **cheap** signal predicts
  per-token loop-benefit, CI lower bound > threshold, **within strata**, beating
  current-NLL **and** CALM/ensemble-entropy, **on the anisotropic operator** and **on
  the NL arm**; (c) survives **all** §4 tautology gates; (d) untrained truncated scouts
  predict **converged** benefit; (e) **reducible-vs-irreducible separation** — mean
  S(reducible-hard) > mean S(aleatoric), non-overlapping CIs, with a ceiling on the
  irreducible fraction of top-x%-by-S tokens. **PLUS §7a.**
- **§7a — single iso-wall-clock gate (folds R1 + economic inequality):** one end-to-end
  simulation that **charges the scout's own compute**, with 2–4 discrete tiers (each a
  dense GEMM + pre-captured CUDA graph, committee fused into one batched GEMM +
  streaming JS), reporting **median+tail tokens/sec** of tiered+scout vs a matched
  dense baseline on the **same hardware**. Pass line: e.g. **≥1.15× tokens/sec AND
  sustained GPU util ≥ floor**, with `scout_time+sync < saved_main_time` **measured,
  not modeled**. FLOP counts never satisfy this.
- **Conditional caps:**
  - Trained without the compute-budget regularizer → **conditional GREEN, R6
    unobserved.** Required pre-build gate: one arm with an **annealed FLOPs/ponder cost
    + stop-gradient on the surprise path**; verify S retains non-trivial variance and
    still predicts benefit (else R6 cold-start collapse will kill it at build).
  - Synthetic-only → cap at **YELLOW**.
- **YELLOW = "do more Stage-2 work," never "build with a TODO."** §4.2 failure
  (untrained truncated scout fails) is **RED-for-the-cheap-scout**; it authorizes at
  most a scoped trained-scout re-test (disjoint train split, held-out, must beat
  current-NLL/truncated-loop-NLL), itself **at best YELLOW** (adds unbudgeted cost).
- **RED → core unsupported:** more loops don't help, loop won't converge, or no cheap
  signal beats current-NLL **across the full ε range and the anisotropic arm.** A
  B-variance collapse that appears as ε grows but vanishes as ε→0 is an
  **instrument/stability-regime failure, NOT a real RED.**

---

## 8. The contraction-vs-benefit tension (the crux — resolved)

**Can a loop be provably-convergent (σ≤1−ε) AND have meaningful per-token loop-benefit
variance?** **Not under isotropy — they genuinely fight; yes under the build's
anisotropic operator.** Under isotropic contraction, residual decays like (1−ε)^k so
marginal benefit decays geometrically: a "safe" ε flattens benefit to ≈0 (false-RED on
an artifact), while surviving benefit means ε was too small to be the build's regime
(GREEN won't transfer). Anisotropy escapes by contracting **off-manifold** (convergence)
while staying compliant **along-manifold** (a usable, slowly-varying benefit axis); the
isotropic skip+reconstruction stand-in cannot decouple these.

**→ Therefore:** the headline arm **must** be anisotropic, ε is **swept** (GREEN =
∃ε satisfying both convergence and benefit-variance; RED only if **no** ε does), and the
GREEN-qualifying run must sit in the build's stability regime. **One pre-registered
ε × reconstruction-weight grid on the anisotropic operator** retires the tension, the
false-RED risk, the transfer gap, and the degenerate-fixed-point precondition together.

---

## 9. What Stage 2 still does NOT prove

Experts/prefetch; **breathing/elastic attention** (a per-loop-changing map can push the
*joint* spectral radius >1 even when fixed-map blocks contract — add a monotone-shrinking
breathing arm to the convergence precondition, or name it a pre-build gate); full R6
cold-start dynamics at scale; throughput at scale. These remain gated on this Stage-2
GREEN + the §7a wall-clock gate.
