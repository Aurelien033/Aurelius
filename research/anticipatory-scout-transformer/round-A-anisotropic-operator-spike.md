# Round A — Anisotropic Equilibrium Operator Spike

**The crux test. Runs BEFORE the validation ladder.**
*(v2 — rewritten after adversarial review `w6ej09ksi`, which found the v1 stability
certificate mathematically wrong (eigenvalues where a non-normal operator needs
singular values) and showed the decisive test is a CPU pre-spike runnable in hours.
Supersedes v1.)*

Answers the one question the whole program reduces to ([notes §12](anticipatory-transformer-research-notes.md)):

> **Can an anisotropic equilibrium operator be simultaneously *stable*, *trainable*,
> and *benefit-capable*?**

RED here = stop the program. Structured in **two phases** so the cheapest, most-likely-
to-kill test runs first.

| Phase | Cost | What | Can it RED alone? |
|---|---|---|---|
| **Phase 0 — analytic pre-spike** | **hours, CPU, no learning** | pure linear algebra on small planted matrices | **YES** |
| **Phase 1 — minimal learned confirmation** | 1–2 days, 1 GPU | T1 planted + synthetic-benefit + emergent-anisotropy gate | only if Phase 0 survives |

"Run A first" is justified by Phase 0 being **hours**, not days.

---

## 0. Why this is Round 0

Novelty (anisotropic stiffness, [§10.2]) and *all* HIGH theoretical risk (R4/R5:
marginal stability, non-normal transient blow-up, gradient conditioning →∞) are the
**same component**. Phase 0 can kill the program for the cost of an afternoon, before any
GPU, learning machinery, or the signal ladder (which tests an independent premise and
runs in parallel).

---

## 1. The object under test (aligned to the build, §10.4)

**Iteration** (subtractive form): `z_{k+1} = z_k − D·r(z_k)`, `r(z)=z−G(z)`,
Jacobian `J = I − D·J_r`. Seek fixed point `z*=F(z*)`.

- **`D` — strictly SPD** anisotropic preconditioner. Parameterize `D = LLᵀ + ηI`
  (η>0) with `L` lower-triangular, **softplus-positive diagonal** (plain `LLᵀ` is only
  PSD — a v1 bug). Defined in a **fixed symmetric metric** (state it).
- **HEADLINE ARM = the §10.4 design, NOT eig≈1.** §10.4 abandoned the
  along-manifold-eigenvalue≈1 route. So the headline operator enforces **strict
  `σ_max(J) ≤ 1−ε` everywhere**, and delivers along-manifold *compliance* via an
  **additive skip path + reconstruction loss** — benefit-richness is measured on the
  **skip-augmented** map. The raw `σ∥≈1` operator is a **labelled negative control
  only.** The swept knob is **ε (singular-value margin)** — matching §12's "sweep ε,
  never toggle."
- **Non-normality is INJECTED, not hoped for (else the toy is benign and false-GREENs).**
  `J_r` carries a parameter **γ** = strictly-triangular non-symmetric coupling
  `P∥→P⊥` (and/or Jordan shear in `P⊥`). **Sweep γ.** Calibrate the γ-range to a real
  attention block's measured non-normality (departure-from-normality / numerical
  abscissa of a small attention Jacobian); document the calibration.
- **Manifold `P∥`** (rank `m`): planted in Phase 0 / T1; estimated **independently** of
  the anisotropy claim elsewhere (top-m subspace of reconstruction targets, or planted
  subspace transported through a known map) — never read off the same operator it's
  meant to verify. Report **principal angles** to the independent estimate.

---

## 2. Phase 0 — analytic pre-spike (hours, CPU, no learning — can RED on its own)

Build `J = I − D·J_r` with planted `P∥` (rank m), strictly-SPD `D` at the §10.4 margin,
and **non-normal** `J_r` carrying γ. On a **1-D grid of ε × a few γ points** (sub-second
numpy each), compute on the **full coupled J** (see §5 for the correct estimators):

- (a) **transient amplification** `κ = sup_{k≤K} ‖J^k‖₂` over the few-loop budget k=1..5;
- (b) **Kreiss constant** lower bound and **numerical abscissa** `ω(J)=λ_max((J+Jᵀ)/2)`;
- (c) **`cond₂(I−J)=σ_max/σ_min(I−J)`**, and verify the `~1/(margin)` growth law;
- (d) **fixed-point existence/uniqueness margin** + the monotonicity/contraction
  sufficient condition (§5).

**Phase-0 decision (declared in hours):** if, for *every* benefit-permitting ε, either
(i) `κ` grows unbounded with γ in k≤5, **or** (ii) `cond₂(I−J)` blows up past the fp32
floor → **program RED-STABILITY / RED-GRADIENT.** Phase 1 never runs. Otherwise, carry
the surviving (ε, γ) operating point(s) into Phase 1.

---

## 2.1 Worked analytic example (exact, by hand — instrument check + first findings)

*(Done without a machine; reproduce/scale via [round_A0_prespike.py](round_A0_prespike.py).)*

- **Case 1 — the eigenvalue-vs-singular-value gap.** `J=[[0.9,3],[0,0.9]]`: ρ(J)=0.9
  (eigenvalue test "passes") but **σ_max=3.25**, numerical abscissa=2.4, transient
  ‖Jᵏ‖₂ ≈ **7.4× at k=3**, peak ~11.6× at k≈9. → the v1 eigenvalue certificate
  **false-GREENs** an operator that amplifies ~7× in the scout regime. Corrected
  certificate (σ_max on full J) catches it.
- **Case 2 — §10.4 fix vs coupling budget.** Enforcing `σ_max≤1−ε` with along-factor
  0.9, ε=0.05 → surviving coupling **g ≲ 0.098** (down from 3). Fix removes blow-up;
  cost = tight non-normal-coupling (expressivity) budget.
- **Case 3 — contraction-vs-benefit (tol=0.01).** Iteration spread (benefit) vs margin ε
  with ε_off=0.2: ε=0.02 → 11× spread, cond(I−J)≈10; ε=0.05 → 4.4×, ≈4; ε=0.10 → 2.1×,
  ≈2; ε=0.20 → ~1×. **Non-empty viable window ε≈0.05–0.1** (2–4× spread, cond 2–4,
  practical iters).

**Analytic verdict = conditional GREEN, not RED:** the strict-margin design removes
transient growth (σ_max<1 ⇒ no amplification) AND keeps cond(I−J)≈ε_off/ε≈2–10 (R5
tame). Open question handed to Phase 1: can attention+LoRA do useful adaptive work
within the small coupling budget, in the ε≈0.05–0.1 window? (→ T3.)

## 3. Pre-registered thresholds (frozen before any sweep; gated on CI lower bounds)

*(Numbers illustrative — pin them, don't choose post hoc.)*

- **STABLE (converges):** unique fixed point within tol across **R≥20** inits **AND**
  existence certificate holds **AND** `σ_max(full J) ≤ 1−ε` at the operating point
  **AND** transient `κ = sup_{k≤K}‖J^k‖₂ ≤ 2.0` at the **largest calibrated γ** **AND**
  residual reaches its floor **within the K-loop budget** (not k→∞).
- **TRAINABLE:** `σ_min(I−J)` ≥ floor s.t. `cond₂(I−J) ≤ 1e3` (fp32) **AND** zero NaNs
  **AND** grad-norm 99th-pct ≤ 10× median **AND** implicit-vs-unrolled cosine ≥ 0.95
  **at k=1,2,3** (the truncated regime scouts live in — not just near z*).
- **BENEFIT-CAPABLE** (renamed from "benefit-rich"; necessary-only): iteration-count
  **CV ≥ 0.3** on heterogeneous difficulty **AND** ≈0 (< cross-seed std) on a
  **homogeneous-difficulty control** **AND** median `loss(k)−loss(k+1) > 0` beyond k=1
  **AND** along-manifold work fraction ≥ 0.5 **AND** along-manifold-step-vs-loss-reduction
  Spearman > 0 (drift ≠ improvement).
- **Region width:** GREEN = viable over **≥1 decade of ε**; YELLOW = viable only within
  <0.5 decade (knife-edge) or only with phantom gradients; RED = empty **after grid
  refinement.**

---

## 4. Phase 1 — minimal learned confirmation (only if Phase 0 survives; 1–2 days, 1 GPU)

- **T1 (planted) + synthetic benefit:** plant a *distribution* of per-token off-manifold
  noise norms so iterations-to-tolerance has variance **by construction**; verify
  Battery 3 recovers iteration-count variance, along-manifold work fraction, and
  drift-vs-improvement separation. **Must recover the planted `P∥` within a
  pre-registered principal angle** before any verdict is trusted.
- **Sweep collapses to a phase LINE over ε** at one safe ε_off, one nominal
  reconstruction weight, one small m — then **two 1-D robustness re-runs** (larger m;
  alternate reconstruction weight) at a viable point to rule out knife-edge.
- **T3 — emergent-anisotropy gating task (attention + LoRA, NO planted manifold, NO
  hand-set D):** the planted/hand-installed operator only shows *if* eigen/singular
  structure is placeable *then* stable — not that **attention+LoRA can actually place
  it**. T3 tests that. Include a **homogeneous-difficulty control** so iteration variance
  can't be an artifact of the task recipe (R7).
- **Gradients in the few-loop regime:** implicit-vs-unrolled agreement + grad-norm/NaN
  battery at **k=1,2,3**; phantom gradients kept only as a YELLOW-rescue at the single
  viable point.
- **Defer (off the go/no-go critical path):** the full 4-D sweep; the T2 learned-manifold
  algorithmic task (its only unique contribution, benefit variance, is exhibited
  synthetically here; it **cannot flip a T1 RED**); the full gradient triad
  (well-posedness is decided by `cond(I−J)` in Phase 0).

---

## 5. The CORRECT diagnostics (v1 measured the wrong objects)

- **Contraction certificate = `σ_max(J)`, NOT `|λ_max|`.** Power-iterate on **`JᵀJ`**
  (autograd jvp *and* vjp) for `σ_max(J)`. For non-normal J, an operator with
  `|λ_⊥|=0.9` but `σ_⊥=3` is transiently expansive yet would pass an eigenvalue test —
  the v1 false-GREEN. Report `ρ(J)=|λ_max|` as a secondary descriptor only.
- **Measure the FULL coupled J**, not block-restricted `σ(P∥ J P∥)` / `σ(P⊥ J P⊥)`. The
  off-diagonal block `P⊥ J P∥` is exactly what drives non-normal amplification; both
  diagonal blocks can be ≤1 while the joint operator blows up. Report blocks +
  `‖P⊥ J P∥‖` as diagnostics; the **binding gate is the full-J transient bound**.
- **`cond₂(I−J) = σ_max/σ_min(I−J)`** via inverse power iteration / LOBPCG on
  `(I−J)ᵀ(I−J)` — *not* `1/(1−ρ̂)` (keep that only as a labelled lower-bound tripwire).
- **Existence certificate after relaxation:** verify the monotonicity/contraction
  sufficient condition still holds (symmetric-part constraint, or certified
  `sup_z σ_max(J(z)) < 1` over a sampled neighborhood), not just empirical residual→0.
- **Multi-token JOINT Jacobian (R2/R3):** all diagnostics on the joint `N·d` Jacobian
  with real attention mixing; **sweep N ∈ {2, 8, 32}.** Per-token block radii <1 with
  joint radius >1 is **NOT GREEN.** Add a heterogeneous-halting sub-test (R3): if
  `z*`/loss depends on the halt schedule, GREEN is conditional on a synchronous solve.

---

## 6. Decision rule — separable verdicts + "what is NOT a real RED"

Split the single program-RED into verdicts with different remedies:

- **RED-STABILITY** (transient blow-up / no contraction across the range) → **stop.**
- **RED-GRADIENT** (`cond(I−J)` un-tameable, *and phantom gradients also fail*) → **stop.**
- **RED-BENEFIT** (no iteration variance / no per-iter work / drift-not-improvement) →
  pivot or stop.
- The roadmap abandons the differentiator only on **RED-STABILITY or RED-GRADIENT
  surviving the guards.**

**Guards — these are NOT real REDs:** a knife-edge cell declared empty before **grid
refinement**; a RED on only **one** D-parameterization arm (require **both** arms RED);
a trainability-RED where **phantom gradients would train**; a RED from an **uncalibrated
over-large γ**; a RED on the **eig≈1 negative-control** arm (it's a strawman — the
buildable §10.4 margin-route is the headline).

- **GREEN:** viable region ≥1 decade of ε; gradients valid at k=1,2,3; bounded κ at
  calibrated γ; benefit-capable thresholds met; T1 recovers `P∥`; **T3 shows attention+
  LoRA can place the structure.** → run the signal ladder; use this operating point in
  Stage 2.

---

## 7. Cost

- **Phase 0:** <1 day, **CPU** (small matrices, numpy).
- **Phase 1:** 1–2 days, 1 GPU.
- **T2 / full 4-D sweep:** +2–3 days, **off the critical path**, cannot flip a verdict.

*(v1's flat "days, 1 GPU" was wrong and would have invalidated the roadmap's "cheapest
first" ordering — the per-phase breakdown restores it.)*

---

## 8. Stand on existing theory

Off-manifold contraction / well-posedness: monotone operator equilibrium nets (Winston &
Kolter 2020); non-Euclidean & contraction-theory implicit nets (Jafarpour et al. 2021–22).
Non-normality / transient growth: Trefethen & Embree **pseudospectra**; Kreiss matrix
theorem. Implicit-diff conditioning & Jacobian-free/phantom gradients (Geng et al. 2021;
Bai et al. 2021). **The open question this spike uniquely answers:** does relaxing toward
along-manifold compliance (via the §10.4 skip-path route) preserve `σ_max<1`, bounded
transient growth, and well-conditioned gradients under *injected, attention-calibrated*
non-normality — which prior uniformly-contractive work never has to face.

---

## 9. What this spike does NOT test

The **signal** (scouts / disagreement / residual-as-surprise) — Stage 0–2. **Wall-clock**
— R1 / Round E. **R6 cold-start** — Round F. This spike is *only* about whether the
operator itself is viable. A GREEN is necessary, not sufficient; a **RED-STABILITY/
GRADIENT is sufficient to stop the whole program**, which is why it goes first.
