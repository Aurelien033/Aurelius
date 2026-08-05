# Analysis: Anticipatory ("Scout") Transformer packet (2026-06-19)
### what it is, whether it helps, and how it reconciles with our evidence

> Source: branch `research/anticipatory-scout-transformer` (13 files, ~2.3k ln: design notes + 6 validation
> scripts). I read the design and **ran the cheap kill-test smokes locally** (A0 numpy, E roofline, F guard).

---

## 1. What it is
**Scout-disagreement-gated adaptive recursion** (the AMPLIFY rung of the original SKIP/NORMAL/AMPLIFY thesis):
lightweight "scout" passes run *ahead* of the main model, leave a guess + a **committee-disagreement** score, and
that disagreement ("surprise") decides **how many times each token loops** through the main block — more loops
where hard, cruise where easy; the preview also primes the representation. Grounded in ACT / Universal
Transformers / PonderNet / MoD + **query-by-committee** (disagreement as uncertainty) + speculative decoding +
parafoveal-preview. Differentiator vs prior adaptive-compute: replace the black-box learned halting prob with an
**interpretable disagreement signal**. First build = "Approach 2: scouts gate recursion."

## 2. Quality — high, and the opposite of the 4DEE scope-trap
This is disciplined work: a staged **kill-test ladder** (A0 operator stability → A1 learned operator → Stage 0/1
signal study → Stage 2 build gate) with **GREEN/YELLOW/RED gates that can stop the program**, σ_max contraction
certificates on the **full coupled Jacobian** (correctly *not* eigenvalues — handles non-normal transient growth),
**document-level bootstrap CIs** (gates on the CI lower bound, not point estimates), iso-FLOP **true policy**
simulation, and instrument-validity guards. Adversarially code-reviewed. One focused idea, falsifiable — exactly
the discipline the program rewards (and the antithesis of the 15-mechanism 4DEE framework).

## 3. Smoke results (ran locally, CPU)
- **A0 (numpy operator kill-test): runs, sound.** Correctly implements the σ_max-not-eigenvalue check (its CASE 1 is literally the "eig<1 but non-normal false-GREEN" trap); operator STABLE across the contraction sweep. Instrumentation is real.
- **F coldstart guard: correct** — `--k_scout 5, K=6` raises a clean `AssertionError`, not an `IndexError`.
- **Round E roofline: an HONEST NEGATIVE already.** The FLOP-based speedup gate (≥1.15× *and* scout+bucket < dense) is **never satisfied**, and the script itself notes **token-level adaptivity is not cleanly benchmarkable because attention couples tokens** → you must bucket at the sequence/request level. (CPU caveat noted; needs GPU re-run.)

## 4. Reconciliation with our arc (the honest cautions)
1. **The Stage 0/1 crux is the same genus we've repeatedly found NULL.** "Does a *cheap signal* predict *where compute helps*?" is exactly what our SKIP work answered no for (entropy anti-informative; per-layer gain ≤ random; per-task selector null at k=2 and k=4). The signal here (committee **disagreement**) and the lever (**loop depth / AMPLIFY**) are genuinely *different* from ours, so it's **not pre-falsified** — but our three nulls weigh on the prior. **Guarded.**
2. **The efficiency premise faces our speedup wall — and Round E already shows it.** Even if the signal predicts benefit, FLOP savings ≠ wall-clock (our byte-model audit: skip gave ~no speedup), and attention coupling forces sequence/request-level bucketing (Round E's own finding). So a "signal works" GREEN still has to clear a hard *systems* gate.
3. **AMPLIFY is the less-explored rung.** We hammered SKIP; adding compute where hard, driven by disagreement, is genuinely untested by us. That's the upside.

## 5. Verdict & where it fits
**Keep it as a live, cheap-to-test research probe — not a priority over the release path.** It's high-quality and
falsifiable, but (a) it re-enters the cheap-signal-predicts-benefit regime that's been null for us, and (b) its own
roofline already flags the speedup/bucketing wall. The saving grace is that the **early gates are cheap and
decisive**: A0 is done (sound); **Stage 0/1** (does scout disagreement predict where compute helps, with the
bootstrap-CI lower-bound gate) is a few GPU-hours on a tiny pretrained LM and would settle the crux before any
build. Slot it as **E-8 (adaptive-compute / AMPLIFY probe)** in the menu, behind v2→RLVR, with the explicit
decision rule: *Stage 0/1 must clear its CI-lower-bound gate on real data, or it stops* — and even a pass must then
beat Round E's systems gate, not just the FLOP count.

**Net:** genuinely good work, correctly skeptical of itself; worth the cheap Stage-0/1 falsification when there's
slack — but our own evidence says enter it eyes-open about the signal-predicts-benefit prior and the speedup wall.
