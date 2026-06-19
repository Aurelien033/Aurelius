# Anticipatory ("Scout") Transformer — Research Notes

**Status:** Brainstorming / pre-spec
**Date:** 2026-06-19
**Origin:** Reading *Attention Is All You Need*; exploring adaptive-compute transformers.

---

## 1. Core idea (one sentence)

> A transformer in which lightweight expert **"scouts"** run *ahead* of the main
> model, leave cheap annotations (a guess + a disagreement score), and that
> **disagreement = "surprise"** signal decides how much computation the main
> model spends on each token — looping more and (later) recruiting heavier
> experts where surprise is high, cruising where it is low.

The scouts' preview also **primes** the main model's representation, so even
routine tokens are handled better than if met cold.

---

## 2. The problem we're reacting against

Standard transformers spend the **same compute on easy and hard inputs**. There
is no "wait, this is surprising — think harder" reflex. We want adaptive
per-token computation, triggered by a *principled, interpretable* signal.

*(This was the chosen failure mode — option C in early brainstorming. Other
candidates: frozen-after-training, catastrophic forgetting, fixed capacity.
Those are explicitly out of scope for the first project.)*

---

## 3. How the original intuitions map to real research

| Original phrasing | Maps to |
|---|---|
| "surprise reaction vector" | Prediction error / **Bayesian surprise** (free-energy principle, Friston) |
| "reactionary / adapts to any situation" | **Test-time adaptation**, adaptive computation |
| "expanding brain / learns new skills" | Continual learning, **dynamic network growth**, MoE (deferred to later work) |
| "multi-positional attention" | Multi-scale / hierarchical attention (left fuzzy — not load-bearing) |
| "recursive/regressive attention + MoE + MoD" | The compute-spending mechanisms, all driven by one surprise signal |
| "experts as scouts / preview before full context" | **Speculative preview** + **query-by-committee** + reading **preview benefit** |

---

## 4. Literature anchors (stand on these, don't reinvent)

**Adaptive compute lineage**
- Adaptive Computation Time — Graves, 2016 ("ponder until done").
- Universal Transformers — 2018 (recurrent depth, per-token halt).
- PonderNet — 2021 (clean differentiable halting).
- Mixture-of-Depths (MoD) — DeepMind, 2024 (tokens skip layers).

**The novel seam — where the surprise signal comes from**
- Query-by-committee — Seung et al., 1992: a committee of cheap models;
  **disagreement** marks informative/hard inputs. Principled uncertainty signal.
- Speculative decoding — 2023: a cheap draft model previews, the big model only
  does heavy work to verify. Proof that cheap previews save big-model compute.
- Preview benefit / parafoveal preview — Rayner et al. (reading research):
  humans preview the next word before fixating, and process it faster/more
  accurately as a result. Empirical analogue of "scouts that get ahead handle
  the task better."

**Key differentiator:** prior adaptive-compute work uses a learned *black-box*
halting probability. We replace that with **scout disagreement**, which is more
principled and interpretable.

---

## 5. Unified mechanism (why this isn't just bolting papers together)

MoD, recursive depth, and MoE are the **same decision at different
granularities**, and one surprise signal drives all of them:

- **MoD** — *whether* a token enters the heavy path (admit vs. skip).
- **Recursion** — *how many times* a token loops through the block (depth).
- **MoE** — *which* specialized weights process it (width/capacity). *[later]*

The four "scout types" are not four features — they are **one behavior plus a
choice of how to economize**:

- *What scouts do:* run **ahead** of the main model (lookahead).
- *How a scout is made cheap (pick one):*
  - **A. Shallow-depth** — full network run only 1–2 layers/loops. ← chosen
  - **B. Specialist** — several different small experts vote.
  - **C. Compressed-context** — read a downsampled/summarized context.

---

## 6. Project shape — chosen path

| Approach | What it proves | Verdict |
|---|---|---|
| 1. Just the gate (surprise-gated recursion, no scouts) | plumbing only; ≈ PonderNet | **Skip** — doesn't test our idea |
| **2. Scouts gate recursion** | the actual novel claim | **Build first** ✅ |
| 3. Full anticipatory transformer (+ MoE) | the whole vision | **Paper 2 / extension** |

**Decision:** Build **Approach 2** first, with the architecture explicitly
designed so Approach 3's MoE can bolt on later without a rewrite.

### Approach 2 in detail
- *k* shallow scouts (each = the block run 1–2 loops) run ahead, emit guesses.
- **Disagreement among scouts = surprise signal.**
- High disagreement → main model loops more on that token AND consumes the
  scout preview as a prior. Low disagreement → cruise.
- Falsifiable cleanly: a bad result implicates the scout signal specifically,
  not five tangled mechanisms.

### Economic success criterion (must hold for the idea to be worth anything)
```
(scout cost) + (heavy cost on hard tokens only)  <  (heavy cost on everything)
```
Scouts must be much cheaper than full processing, and their preview informative
enough to route correctly.

---

## 6.5 Stabilizer — "4D elasticity" (confirmed)

The recursion adds a 4th axis to activations: **(batch, sequence, feature,
loop-step)**. Elasticity = a **restoring force along the loop axis** that pulls
each iteration toward a stable resting point.

- The loop map's **Jacobian = deformation gradient** (continuum-mechanics term):
  how the representation stretches/shears from loop *n* to *n+1*.
- **Stability (answer A)** = global behavior: spectral norm < 1 → contraction →
  converges to a **fixed point** (Deep Equilibrium Models, Bai et al. 2019).
- **Manifold geometry (answer C)** = local/directional behavior of the same
  Jacobian: how it curves/stretches representation space.

**Productive tension (the real contribution):**
- Pure contraction collapses everything → destroys information (stable but useless).
- Pure isometry preserves meaning but gives no convergence guarantee.
- **Resolution = anisotropic elasticity:** a stiffness *tensor* that is **stiff
  off-manifold** (contract hard → converge, kill jitter) and **compliant
  along-manifold** (near-isometric → preserve information).

Pays off three ways: (1) provable convergence, (2) damps router jitter,
(3) **strain → 0 gives a differentiable halt rule**.

**Staging (confirmed):**
- Experiment 1: **scalar / isotropic** elasticity (global spectral bound) — proves
  the recursion is stable. Reuses spectral-norm machinery.
- Centerpiece extension: **anisotropic stiffness tensor** (the headline novelty).

## 6.6 Subtractive (residual) attention — the loop's update rule

Not a separate mechanism — it **defines what the recursion block computes each
loop**: subtract the "explained-so-far," then **attend to the residual** (the
unexplained, overlooked context). Where the residual is large = missed context.

- The **residual *is* the surprise signal, made spatial** — tells you *where* in
  the sequence the surprise lives, not just how much.
- Lineage to cite (NB: "subtractive attention" is non-standard — search instead
  for **residual / error-driven / iterative-refinement attention**):
  - Predictive coding (Rao & Ballard, 1999) — iterate on residual error.
  - Slot Attention (Locatello, 2020) — iterative explain-away attention.

**Unifying picture — five names, one loop:**
> Each iteration: subtract explained part → attend to residual → update
> explanation. Residual shrinks each loop. **Elasticity** is the damping that
> makes it shrink; **residual → 0** is the halt. **Scouts** are a cheap
> 1–2-loop preview of this same loop running ahead. **Surprise** = residual
> magnitude / scout disagreement.

**Caveat:** subtractive attention can *over*correct (over-relaxation →
oscillation). It is stable **only paired with elasticity/damping.** Update rule +
damping are a matched pair.

**Resolved — subtraction acts on the values (option B):** attend everywhere, but
read back only the residual:
```
value_read = V_actual − α · V_explained
```
This is predictive coding in pure form (the residual = the "error unit",
Rao & Ballard). `α` = a **gated/damped step size** (leak rate). Too large → over-
shoot → residual flips sign → oscillation.

**Key result:** `α` *is* the elasticity coefficient from §6.5. The damping
(§6.5) and the update rule (§6.6) are the **same knob** — not two stabilizers.
Subtract on *values* (not keys/scores) so the system is one consistent
predictive-coding loop rather than a recursion with a heuristic mask.

## 6.7 Scouts as a subject-neutral dispatcher (resource pre-staging)

**Design choice:** scouts are **subject-neutral** — identical, general-purpose
recognizers, NOT pre-assigned specialists. Specialization lives in the expert
pool; scouts are the *dispatcher* that recognizes content and provisions for it.
They "adjust to whatever content the context is." (Specialist scouts — the old
option B — are REJECTED.)

**Preview benefit = resource pre-staging, NOT answer warm-starting.** (Warm-start
rejected.) Truer to parafoveal preview: the preview *pre-activates the resource*,
it does not precompute the result.

**One neutral committee → two readouts:**
1. **Disagreement** (epistemic surprise, §7-#1) → *how much compute* to allocate
   (loop / attention budget). [option C — works without MoE]
2. **Mean routing prediction** → *which experts* to speculatively prefetch /
   pre-activate ahead of the main model. [option A — needs an expert pool]

→ Novel framing: **speculative, ahead-of-use expert routing** (branch-prediction
/ prefetch analogy; speculative MoE expert prefetching). Reconnects to message-1
"expanding brain / new skills": scouts recognize task → ready the right skill.

**Source of scout diversity — RESOLVED as a two-level committee (A embedded with B):**
- **A: MC-dropout** (Gal & Ghahramani 2016) → **within-mode** diversity (wiggles
  around one solution). Cheap, near-zero extra params.
- **B: separate inits = deep ensemble** (Lakshminarayanan 2017) → **between-mode**
  diversity (different loss basins / genuinely different "opinions").
- Orthogonal axes → combine: **m separate scouts × k dropout samples** = two-level
  epistemic estimate (between-scout + within-scout spread). Also makes the
  expert-prefetch recognition more robust.

**Cost discipline (non-negotiable):** scouts must satisfy the economic inequality.
`m × k` multiplies scout cost → keep **small & shallow**: m = 2–3, k = 2–3, each
scout 1–2 loops. A full m×k committee would erase the efficiency gain.

**Experimental rule:** B is an **ablation axis, not a baked-in assumption.**
Experiment 1 = pure A (cheapest; proves disagreement tracks difficulty). Then
*add* B and **measure** whether between-mode diversity improves the
difficulty-correlation enough to pay for its compute. Let the experiment decide
how much of B earns its keep.

**STAGING IMPACT:** option A (expert prefetch) pulls the expert pool toward the
*core* — it was previously "Approach 3 only." Revised staging TBD next session:
how big is experiment 1, given experts now matter to the headline idea?

## 6.8 Expansiveness — elastic ("breathing") attention

**Companion to elasticity (§6.5), not a new mechanism.** Elasticity = the
CONTRACTION/restoring half; **expansiveness = the STRETCH half.** Together = a
true elastic system (stretch under load, return to rest). This is the message-1
"expanding brain" landing in the architecture.

Driven by the **same surprise signal** — it just controls one more knob:
attention reach + token participation per loop.
- High surprise → attention **expands** (longer reach / more heads / capacity).
- Low surprise / converged → attention **contracts**; token **EXITS** the active
  set (stops attending / being attended to).

**Why it's necessary, not optional:** naive recursive attention costs
~`k loops × O(n²)`. With breathing attention the **active set shrinks each loop**
→ cost ≈ `Σ_loops O(n_active²)`, `n_active` → the few hard tokens. This is what
makes recursive attention affordable.

**4D pays off literally:** elasticity acts along the **loop** axis (damping →
convergence); expansiveness acts along **feature** (capacity) and **sequence**
(attention reach). All four axes (batch, sequence, feature, loop) adaptive.

**Prior art to differentiate from:** Adaptive Attention Span (Sukhbaatar et al.
2019 — learned but STATIC span); Reformer / Routing Transformer / BigBird (STATIC
sparsity); MoD token-dropping. **Our differentiator:** span is dynamic,
surprise-driven, per-token, per-loop, tied to an epistemic-uncertainty signal
inside a recursive loop. *(Novelty audit to confirm.)*

**Two tensions (overlap the running hardening pass — systems & theory lenses):**
1. **Systems/wall-clock:** ragged per-token attention hurts GPU batching →
   FLOP savings may NOT become wall-clock savings without block-sparse impl or
   surprise-sorting tokens into contiguous blocks. **#1 thing to verify.**
2. **Convergence:** a per-loop-changing attention map breaks the fixed-map
   contraction argument (§6.5). Fix: make expansion **monotone** — reach may only
   CONTRACT as tokens converge (active set only shrinks) → convergence preserved;
   or re-derive the contraction bound for time-varying maps.

## 7. Open design questions (next session)

1. **Disagreement metric — RESOLVED.** Surprise = **epistemic** uncertainty only
   (not aleatoric — never spend compute on irreducible noise). Measured as a
   **committee disagreement** → this is exactly query-by-committee / **BALD**
   (Houlsby et al., 2011): epistemic = mutual info between prediction and which
   scout you ask.
   - *Stage 1 (build first):* **JSD across the k scout output distributions**
     (= entropy-of-mean − mean-of-entropy; the subtraction removes aleatoric).
     Needs a lightweight prediction head per scout.
   - *Stage 2 (efficiency):* **hybrid** — route on cheap representation-space
     spread every token, but **calibrate/supervise it against output-space JSD**
     during training. B's speed with A's meaning.
2. **How the preview primes the main model — RESOLVED (see §6.7).** Via
   **resource pre-staging**, not answer warm-starting: subject-neutral scouts
   speculatively prefetch experts (A) and pre-allocate compute budget (C).
3. **Halting rule** — fixed loop budget vs. stop-when-converged vs.
   stop-when-scouts-agree.
4. **Task choice** — needs per-token difficulty variation (e.g., algorithmic
   tasks, or language with rare/hard spans) to show the effect.
5. **Baseline** — a fixed-compute model of *matched parameter count*; metric =
   accuracy vs. **average FLOPs/token**.
6. **MoE bolt-on interface** — what hook does Approach 3 need reserved now?

---

## 9. Failure modes & de-risking (collapse map)

**Staging — A and B unified (supersedes §6 A-vs-B):** build the FULL vision (B)
but put every risky component behind an **independent ablation switch** (A's
discipline). One codebase, full system by default, each piece toggleable to
isolate its contribution. Two enablers make this safe:

- **Adapter-experts (new):** the skeleton expert pool = a few **LoRA adapters**
  over one shared backbone (cf. LoRA-MoE / AdaMix). Cheap (respects the scout
  economic inequality), collapse-resistant vs full-FFN MoE, and lets speculative
  prefetch be tested in experiment 1 instead of deferred.
- **Robustness principle (new): *speculate softly, verify cheaply, never commit
  irreversibly.*** Generalizes speculative decoding's safety to the whole system.
  Every adaptive decision is a soft, recoverable *hint*, never a hard commitment.
  Kills collapse modes #2/#4/#6/#7 at once.

**Collapse map:**

| # | Collapse mode | Why fatal | Mitigation |
|---|---|---|---|
| 1 | Surprise ≠ difficulty | routing premise void | **Gating hypothesis** — cheap correlation test BEFORE full build |
| 2 | Recursion diverges/oscillates | NaNs, no convergence | elasticity/spectral bound (§6.5) + soft probabilistic halt |
| 3 | Constant-compute collapse | no adaptivity emerges (always max/min) | **compute-budget regularizer** (ponder cost / Lagrangian on avg FLOPs) — most-overlooked must-have |
| 4 | Expert/router collapse | few experts hog, rest die | LoRA adapter-experts + load-balance loss + soft mixing |
| 5 | Scout mode collapse | committee always agrees → no signal | MC-dropout + committee diversity regularizer |
| 6 | Prefetch misprediction | wrong experts → overhead/quality loss | soft speculative prefetch (hint, main model verifies) |
| 7 | Discrete decisions non-diff | routers/halt won't train | soft relaxations (Gumbel-softmax, probabilistic halt) |
| 8 | Scout↔main co-adaptation | chicken-egg training instability | two-timescale training + stop-gradient targets |
| 9 | Token starvation | tokens get 0 compute, silent degrade | minimum-compute floor (every token ≥1 pass) |

**Row 3 is the silent killer:** without an explicit cost term fighting the
accuracy objective, the model never learns to be frugal and adaptivity never
appears. Must be in the design from line one. **Row 1 stays the gate** — validate
the core correlation before hardening anything.

## 10. Hardening review findings (workflow wc78zh4ac — 25 agents, web-verified, 2026-06-19)

### 10.1 Citation corrections (fix before any write-up)
- **Bayesian surprise / free-energy — WRONG as written.** "Bayesian surprise" =
  **Itti & Baldi 2005/2009** (= KL(posterior‖prior)), NOT Friston, NOT "prediction
  error." Free-energy principle = **Friston 2010** (surprise = −log p(sensory)).
  Cite separately; drop the "prediction error = surprise" gloss.
- **DEQ contraction/stability — MISATTRIBUTED, load-bearing.** Bai/Kolter/Koltun
  2019 = fixed-point framing only; gives **NO** contraction/Lipschitz/existence
  guarantee. Contraction stability = **Winston & Kolter 2020 (monotone DEQ)** +
  later contraction-theory. Our whole "elasticity → σ<1 → fixed point" must cite
  2020+, not DEQ 2019.
- **Deep-ensemble between-mode diversity — MISATTRIBUTED.** That's **Fort, Hu &
  Lakshminarayanan 2019** ("Loss Landscape Perspective"); Lakshminarayanan 2017 =
  calibration only. Scout-committee justification must cite Fort 2019.
- **Anachronisms (claims hold, wording loose):** epistemic/aleatoric split =
  **Kendall & Gal 2017 / Gal et al. 2017**, not BALD 2011 / MC-dropout 2016.
  **PonderNet halt is differentiable at TRAINING but SAMPLES (non-diff) at
  inference** — our "differentiable halt" claim needs this caveat.
- *Confirmed as cited:* Transformer (Vaswani 2017), ACT (Graves 2016), Universal
  Transformer (Dehghani 2018/19), MoD (Raposo 2024 — note: fixed capacity, static
  graph, not variable total FLOPs), QBC (Seung 1992; practical metrics = Freund
  1997), speculative decoding (Leviathan / Chen 2023 — distributional not
  bit-identical), preview benefit (Rayner — **SPEED not accuracy; semantic preview
  weak in English**), predictive coding (Rao & Ballard 1999), Slot Attention
  (Locatello 2020), BALD MI (Houlsby 2011), LoRA-MoE (LoRAMoE/Dou 2024; AdaMix
  AVERAGES not routes — don't cite as gated MoE).

### 10.2 Novelty verdict — NOT novel as parts; partially anticipated as a system
Every primitive is published; several pairs already combined. Closest hits, each
killing a claimed primitive:
- **"Agreeing to Stop" (Jang & Simeone 2024)** — committee disagreement *as the
  adaptive-compute halting signal.* Kills "disagreement gates compute." *(most
  dangerous hit.)*
- **Mixture-of-Recursions (Bae et al., NeurIPS 2025)** — unifies adaptive-depth +
  per-token routing + halting in one model. Kills "one mechanism unifies all three."
- **CALM (Schuster et al., NeurIPS 2022)** — per-token adaptive compute + the
  soft/speculative *safety guarantee* we called our "safety principle."
- **MoE-SpeQ / SP-MoE / pre-attention expert prediction (2024–26)** — cheap-preview
  speculative expert prefetch incl. "errors cost compute not correctness." Kills
  "anticipatory prefetch."

**Defensible novelty surface (narrow):** (a) ONE neutral committee → TWO coupled
readouts (disagreement→budget AND mean→prefetch) — "no precedent found," not
proven; (c) **ANISOTROPIC** Jacobian (stiff off-manifold / compliant along) on an
equilibrium loop — prior work is isotropic. Claim (a)+(c) only; expect a reviewer
to say "engineering composition" unless (c) is operationalized and shown to matter.

### 10.3 New risks (beyond the known 9), ranked
- **R1 [HIGH, breaks premise] Ragged per-token loops destroy the dense GEMM.**
  FLOP savings are a *counter artifact, not wall-clock.* Mask finished lanes → pay
  worst-case FLOPs; gather/compact → kills cuBLAS autotune + CUDA-graphs,
  bandwidth-bound. Few-hard-tokens regime = GPU <10% util. Scouts add a serial,
  low-arithmetic-intensity, sync-barriered prologue; scout-ahead serializes
  autoregressive decode; adaptive depth breaks static KV-cache layout. **Directly
  hits §6.8 breathing-attention.** *Fix:* optimize tokens/sec not FLOPs; 2–4
  discrete compute tiers + bucket tokens per tier (dense GEMM + pre-captured graph
  each); fuse committee into one batched GEMM + streaming (Welford) JS; require
  scout_time+sync ≪ saved_main_time on a roofline before believing anything.
- **R2 [HIGH] Per-token σ<1 ≠ joint convergence.** Attention couples tokens; the
  joint N·d operator can have spectral radius >1 while every diagonal block <1 — the
  certificate lies (worse for long context). *Fix:* block-Gershgorin / Lipschitz-
  attention; monitor the JOINT residual; halting = readout, not certificate.
- **R3 [HIGH] Heterogeneous halting → asynchronous (chaotic-relaxation) system.**
  Frozen halted tokens feed still-iterating ones → no single fixed-point map; needs
  weighted max-norm contraction; fixed point depends on halt order. *Fix:* keep the
  solve synchronous; apply halting only to readout/cost accounting.
- **R4 [HIGH] Anisotropic "eigenvalue≈1 along-manifold" = marginal stability +
  non-normal transient blowup; scalar α can't be anisotropic.** Eig 1 is exactly
  what Banach excludes (non-unique fixed point; residual never settles). Non-normal
  Jacobians: ρ<1 does NOT bound transient growth — a "contracting" token can blow up
  by κ in the first few loops (= our regime). *Fix:* control **singular values
  ≤1−ε** (not eigenvalues); recover info via skip path + reconstruction loss; α → SPD
  preconditioner; define anisotropy in a fixed symmetric metric.
- **R5 [HIGH] Gradient-through-fixed-point invalid exactly where we operate.**
  Implicit diff needs (I−J)⁻¹ (no eigenvalue at 1) but we *target* eig≈1 →
  conditioning →∞, gradients explode on "converged" tokens. *Fix:* hard ρ≤1−ε margin
  (incompatible with eig-1 anisotropy — pick ONE); phantom/Jacobian-free grads, same
  step count all tokens; track 1/(1−ρ̂) as divergence alarm.
- **R6 [HIGH] Cold-start collapse to always-halt.** At init: shared weights, m=2–3,
  MC-dropout diversity is *aleatoric* (which JS subtracts) → epistemic signal ≈0,
  while the FLOPs cost term has full gradient → races to no-compute corner. Worse:
  cost gradient into the committee makes the cheapest way to cut compute be
  *collapsing JSD itself* ("surprise becomes a learned constant"); halt-on-residual
  admits a no-op optimum (α→identity). *Fix:* anneal cost term from ~0; diversify
  init/seeds + larger m; **stop-gradient surprise on the compute-budget path**; gate
  halting on task-loss improvement vs a frozen reference, NOT raw residual.
- **R7 [MED] Scouts solve a DIFFERENT fixed-point than the converged main.**
  Truncated 1–2-loop iterates → JS = *transient* variance, not posterior/epistemic;
  BALD identity needs same hypothesis space. *Fix:* train scouts to predict the
  *converged* main output; validate disagreement-vs-actual-extra-compute on held-out.
- **R8 [HIGH] Eval confounds can manufacture a fake win.** The committee is itself an
  accuracy-boosting ensemble (params+FLOPs+ensembling confound); the gating-
  correlation test is **circular** (disagreement IS the allocation rule → no
  counterfactual for low-surprise tokens). See §10.5 controls.

### 10.4 Architecture changes (adopt)
- **Drop "α IS the elasticity coefficient"** — overloaded (step size vs contraction
  vs anisotropy); FLOPs reg drives α past the stability bound. Split: step-size
  preconditioner + *separately-certified* operator-norm constraint (spectral norm on
  LoRA+attention).
- **Drop eigenvalue-≈1 isometry** → strict singular-value margin ≤1−ε + skip path +
  reconstruction loss. (Fixes R3/R4/R5 + two-timescale non-stationarity at once.)
- **Bound the JOINT map** (Lipschitz-attention), never advertise per-token bounds.
- **Keep solver synchronous; halting = readout/cost only.**
- **Replace the 24-micro-pass committee** with one batched GEMM + streaming JS, or a
  learned uncertainty head / last-layer ensembling; hard wall-clock gate.
- **Quantize compute to 2–4 tiers + bucket tokens by tier**; confine iteration within
  a layer with fixed external KV footprint.
- **Stop-gradient surprise on compute path; halt on loss-improvement, not residual.**
- **Decouple prefetch sharpness from load-balancing**; train prefetch vs straight-
  through argmax; sparsify soft-mix to small top-k or prefetch is pure overhead.
- **If the LoRA pool fits in HBM, DROP prefetch** — it's selection not data movement;
  soft-mixing touches all adapters anyway. Count ALL speculative FLOPs in the meter.

### 10.5 Experiment 1 — run NOW on the existing pretrained model (supersedes the dry-run sketch)
**Hypothesis (load-bearing):** committee/MC-dropout disagreement on scout-depth
iterates predicts where extra compute *actually* reduces per-token loss —
**decorrelated from the allocation policy.** If this fails, the design is dead and
no engineering fix matters.

Offline, no training, no policy:
1. Frozen block. Per token, m=3 dropout passes run 1–2 loops (+2–3 seeds if avail).
   Record per-token **JS** (entropy-of-mean − mean-of-entropy) = surprise.
2. **Independently** run *every* token at *every* depth k=1..K (policy DISABLED);
   record loss(k). Benefit = loss(1) − loss(K) (or marginal loss(k)−loss(k+1)).
3. Difficulty label = held-out loss of a **separate frozen reference model** (never
   the surprise signal itself).

Measure: Spearman(surprise, *uncensored* benefit) **within FLOP-spent strata**;
calibration reliability diagram; confirm JS isolates epistemic (label-noise text →
low JS despite high loss); report estimator run-to-run std (m=3 may be noisier than
the effect).

**Green-light:** monotone significant correlation (Spearman ≳0.4) with uncensored
marginal benefit, AND correlation with independent difficulty survives within
compute strata, AND > 2× cross-seed std. **Bonus:** JS beats single-model
softmax-entropy (the CALM baseline) — if not, no reason to pay for a committee.

**Mandatory controls:** never correlate surprise vs live-policy compute (circular);
independent difficulty label; stratify (avoid Simpson reversal); fixed/reported
seeds; compare vs **iso-FLOP uniform-extra-compute** — if "compute everywhere" ties
"compute where surprise is high" at equal budget, adaptivity is falsified before any
build.

---

## 11. Validation ladder (spec review wf w4frxgyyu — "needs-rework", 11 critical fixes)

**Key finding — construct validity:** a single offline correlation test on a frozen
model does **NOT** de-risk the real architecture; it validates a weaker proxy claim.
Extra **layers/parameters** do different work than extra **iterations of one tied
operator**, and fixed-depth MC-dropout (within-mode) ≠ truncated-loop scout
disagreement (transient). **Variant B (cross-scale) is the most misleading, not the
truest** — it rewards added parameters/knowledge a parameter-free loop can't supply.
→ Validation is now a **3-stage ladder** (see `experiment-1-spec.md` v2):

- **Stage 0 — pilot** (~1/10 cost): 2–4 tiers, T≥16, ~25k hard-oversampled tokens,
  noise via bootstrap over the T passes (drops R=5), free CALM baseline. Gates Stage 1.
- **Stage 1 — proxy study** (Variant A early-exit primary; B confirmatory only):
  proves cheap disagreement tracks **depth** benefit. **Necessary, not sufficient.**
- **Stage 2 — tied-loop prototype** (mandatory before build): residual-surprise vs
  **loop**-benefit on a small weight-tied block. The construct-valid test.

**Critical methodology fixes baked into the spec:** document-level cluster bootstrap
(tokens are autocorrelated → token-N overstates power 1–3 orders); GREEN = CI lower
bound > threshold, not point estimate; tail/decision metric (NLL captured by top-x%-
by-S vs oracle ceiling), not global ρ; ρ(S,B) is the ONLY gate, ρ(S,D) diagnostic;
iso-FLOP as a true policy sim crediting realized (signed) B incl. regressions + oracle
headroom; instrument-validity gate (dropout p>0; S≈0 → Plan-B, NOT red); tuned lens on
a document-disjoint split fit to gold tokens; pinned marginal B; matched free CALM
baseline; honest cost (days, not "one run") + a minimum-quality bar below which a null
is inconclusive. Pseudocode bugs fixed (full-NLL via real logits, streaming JSD,
shared shifted nll helper, dropout enable/disable).

## 12. Stage-2 review (wf wyvommwcf — "needs-rework"); the program's crux

A naive Stage-2 does **not** justify the build. Four transfer conditions (spec v2 §0):
signal must survive on the **anisotropic** operator (not isotropic), beat trivial
baselines on the **natural-language** arm (not just synthetic), win on **measured
wall-clock with the scout's own cost charged** (R1 quantified, not hand-waved), and
**untrained truncated scouts must predict converged benefit** (R7).

**Contraction-vs-benefit tension — resolved:** isotropic σ≤1−ε and per-token
loop-benefit genuinely *fight* (residual decays (1−ε)^k → benefit flattens → false-RED;
or ε small → not the build's regime → GREEN won't transfer). Only the **anisotropic**
operator escapes (contract off-manifold, stay compliant along-manifold). → headline arm
must be anisotropic; **sweep ε, never toggle**; GREEN = ∃ε meeting both convergence and
benefit-variance.

**The crux this exposes for the whole program:** the design's *only* defensible novelty
(anisotropic stiffness, [§10.2]) is also where *all* the HIGH risk concentrates
(R4/R5: marginal stability, non-normal transient blowup, gradient-through-fixed-point
conditioning →∞). So the entire research bet reduces to one question: **can an
anisotropic equilibrium operator be made stable, trainable, AND benefit-rich at once?**
Everything else is downstream of that.

## 8. Next step

**Design + 3-stage validation plan are now review-hardened (two adversarial passes).**
Path: (1) **Stage 0 pilot** on the existing model — cheapest go/no-go; (2) **Stage 1**
proxy study (necessary, not sufficient); (3) **Stage 2** anisotropic tied-loop prototype
+ ε-sweep + NL arm + iso-wall-clock gate — the only thing that green-lights the build,
and only conditionally (R6 still unobserved until trained with the cost term).
Instrument/underpowered/over-damped REDs are NOT real REDs. **Highest-leverage open
question before any of this: settle whether the anisotropic operator is buildable at all
(§12 crux)** — a small math/stability spike could pre-empt the whole ladder.
