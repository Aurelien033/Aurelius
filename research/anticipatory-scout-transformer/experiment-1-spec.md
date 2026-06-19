# Experiment 1 — Does cheap disagreement predict *where compute helps*?

**A staged validation ladder for the Anticipatory Scout Transformer.**
Offline, frozen models, **no base-model training.**
Companion to [research notes §10](anticipatory-transformer-research-notes.md).
*(v2 — rewritten after adversarial review `w4frxgyyu`; supersedes v1.)*

---

## 0. What this proves — and what it does NOT (read first)

The architecture spends compute by **looping one weight-tied block** (subtractive/
residual attention toward a fixed point) and routes on **truncated-loop scout
disagreement**. We cannot test that on a frozen standard transformer directly, so
this ladder tests it in **three escalating stages**, each gating the next:

| Stage | Cost | What it tests | If it passes, you've earned… |
|---|---|---|---|
| **0. Pilot** | ~1/10 of Stage 1 | is there *any* signal? | permission to run Stage 1 |
| **1. Proxy study** | days | does cheap disagreement predict **depth** benefit? | a **necessary** result — not sufficient |
| **2. Tied-loop prototype** | a small training run | does it predict **loop** benefit (the real axis)? | green-light to build |

> **Honest construct-validity statement (do not skip).** A GREEN at Stage 1
> proves only: *cheap fixed-depth disagreement tracks the benefit of more
> depth/parameters.* It does **not** prove the real claim, because (a) extra
> **layers/parameters** do qualitatively different work than extra **iterations of
> one tied operator** — a token helped by a late specialized layer may be exactly
> where extra loops do nothing; (b) fixed-depth MC-dropout wiggles *within* one
> converged solution, whereas truncated-loop scouts disagree because they sit at
> different points on a *non-converged* trajectory (risk R7 stays untested);
> (c) subtractive attention / residual-as-surprise is absent entirely. **Stage 2
> is mandatory before any build commitment.** Stage 1 is the cheap filter that
> decides whether Stage 2 is worth running.

This also means: a **RED at Stage 1 from the pure-MC-dropout committee or a badly
fit lens does NOT kill the idea** — it may be an instrument failure (see §3, §5).

---

## 1. What you need

- **Main model (frozen):** your pretrained LM with accessible per-layer hidden
  states. No fine-tuning.
- **Held-out corpus** in your domain, counted in **independent documents**, not
  tokens (see §6 — tokens are autocorrelated). **Oversample hard/technical spans**
  so the decisive cell isn't starved (§6, C11).
- **Separate reference model** (different family) for an independent difficulty
  label `D`. Byte/char-level NLL aggregated to your tokenizer's spans (§7).
- **Dropout modules with p>0** for MC-dropout — *verified*, not assumed (§3 gate).
  If absent → Plan-B committee (seeds/sizes), **not** a RED.

---

## 2. Compute axis — Variant A is primary

| | **Variant A — single-model early-exit (PRIMARY)** | **Variant B — cross-scale (AVOID for the headline)** |
|---|---|---|
| "More compute" | read out at layer ℓ vs full depth, via a **tuned lens** | small vs large model |
| Status | always available; closest cheap proxy for "more passes over the same weights" | **rewards extra parameters/knowledge a parameter-free loop cannot supply → spurious ρ.** Only as a confirmatory check, and only on a **byte-identical-tokenizer, same-training-run** ladder (or early-exit of the large model). |

Bound Variant A's grid to the **2–4 discrete tiers** the architecture will
actually use (not "every layer") — fewer lens fits, and it matches the real
policy's decision (§MAJOR).

---

## 3. Definitions (all indexed by the **predicted-token** position; one shared mask)

- **Surprise `S(t)` — MC-dropout JSD committee.** `T = 16` stochastic passes
  (dropout on); per token, next-token dists `p_1..p_T`.
  `S(t) = H(mean_i p_i) − mean_i H(p_i)` (entropy-of-mean − mean-of-entropy =
  epistemic part). **Streaming** accumulation (never stack — §C2). fp32,
  `log_softmax`, `clamp_min(0)`.
- **Benefit `B(t)` — PINNED to the marginal at the scout→next-tier increment:**
  `B(t) = NLL(tier_k) − NLL(tier_{k+1})` — this is what a live policy actually
  decides. (Total `NLL(low)−NLL(full)` is reported-only.) **`B` is signed** —
  deeper hurts some tokens; report negative-`B` prevalence and charge regressions
  in the iso-FLOP test (§MAJOR).
- **Difficulty `D(t)`** — reference-model NLL, byte-aggregated to main tokens.
  **Diagnostic only — never a gate** (§6).
- **Instrument-validity gate (run BEFORE scoring):** enumerate `nn.Dropout`; assert
  some `p>0`; confirm two MC passes differ on a probe batch above a stated floor
  (cross-token variance + test–retest reliability). **Fail → Plan-B committee,
  record as "instrument failed," NOT RED.** Pre-register ≥1 non-MC-dropout
  estimator (seed/size ensemble, cite Fort 2019) as a required comparison.

---

## 4. Procedure (deterministic passes in `eval()`; dropout only inside `surprise_jsd`)

1. Fix & record all seeds.
2. **Tuned lens (Variant A):** fit one affine map per tier on a split **disjoint at
   the document level** from all scored tokens; fit to predict the **gold next
   token** (cross-entropy, Belrose recipe) — **not** to regress onto final logits
   (that drags shallow readout toward the deep answer and shrinks `B` → false-red).
   Fit **two independent lenses**; require results stable across both. Report lens
   val-NLL and the final-tier lens-vs-true-NLL gap as QC. Variant A is
   **confirmatory**, since `B` is probe-dependent.
3. **Benefit sweep** (`eval()`, policy disabled): every tier → `NLL(tier)` per
   token. The CALM entropy baseline is computed **for free** from these logits.
4. **Surprise pass** (dropout on via `try/finally`): `T=16` → `S(t)`.
5. **Difficulty pass** (`eval()`): reference model → `D(t)`, byte-aggregated.
6. Apply the **shared boolean mask** (drop pad/BOS/no-next-token) identically to
   `S, B, D, C` and all baselines; report post-mask counts per stratum.

---

## 5. Stage 0 pilot (run this FIRST — ~1/10 cost, same yes/no)

- 2–4 tiers only; `T=16`; **~25k hard-oversampled tokens**.
- Estimate `S`'s noise by **bootstrapping over the T existing passes** — this
  **replaces the R=5 re-run entirely** (same noise estimate, zero extra forwards).
- CALM entropy baseline free from the sweep logits.
- **Gate:** only run the full Stage 1 if the pilot's ρ(S,B) **document-bootstrap CI
  lower bound** clears the threshold. A noisy null here = "underpowered,
  inconclusive," not RED.

---

## 6. Statistics (this is where v1 would have lied)

- **Resampling unit = the document/sequence** (cluster/block bootstrap), never the
  token — tokens are heavily autocorrelated, so token-N overstates power by 1–3
  orders of magnitude (the #1 false-green route in v1). Report **effective N /
  document count** per stratum.
- **GREEN requires the 95% CI *lower bound* > threshold**, not the point estimate.
  Set the corpus size from a **stated power (≥80%)** to detect the minimum ρ you'd
  build on.
- **Primary metric is tail/decision-relevant, not global ρ:** fraction of total
  recoverable NLL captured by allocating extra compute to the **top-x%-by-`S`**
  tokens, as a fraction of the **oracle top-x%-by-`B` ceiling** (precision/nDCG
  style). Global Spearman ρ(S,B) is a coarse monotonicity check only — the system
  acts on the tail, and global ρ is bulk-dominated.
- **Estimator-noise gate (replaces the incoherent "2× std" rule):** compute ρ on
  each bootstrap/replicate; require the ρ CI (combining T-pass noise **and**
  document bootstrap) above threshold. Report **disattenuation** (test–retest
  reliability of `S`) so a finite-`T` RED isn't a sampling artifact.
- **Multiplicity:** pre-register the exact strata boundaries and **one** aggregate
  rule (e.g. "CI lower bound > threshold in every stratum with ≥N_min documents +
  a pre-specified pooled estimate"); apply Holm/Bonferroni or a hierarchical model.
- **ρ(S,B) on uncensored benefit is the ONLY gate.** ρ(S,D) is a diagnostic that
  can never substitute for it. The real test inside the hard stratum: does `S`
  still rank tokens by `B` — i.e. separate **reducible-hard** from
  **irreducible-hard**? Report the fraction of high-`S` tokens that are
  irreducibly hard (high `D`, `B≈0`).

---

## 7. Baselines & controls (mandatory)

1. **Iso-FLOP, as a true policy simulation.** Discretize budget to realizable
   tiers. Allocate using **only `S`** (with its estimator noise), pick threshold
   `x` on a **disjoint split**, then credit **realized `B(t)` including
   regressions** on negative-`B` tokens. Uniform = same tier / round-robin at equal
   total FLOPs. Add an **oracle ceiling** (allocate by true `B`) and report
   S-allocation gain **as a fraction of oracle gain** (the headroom), with a
   cluster-bootstrap CI. **If S-allocation ties uniform → RED.**
2. **CALM single-model entropy** (free from §4.3 logits): dropout off, single pass
   at the **same tier** as `S`, same mask/shift, scored on the **identical `B`
   array**. **If entropy predicts `B` as well as JSD → the committee is
   unjustified** (YELLOW), regardless of how good the JSD number looks.
3. **Aleatoric criterion + positive control.** On difficulty- and frequency-matched
   **aleatoric** spans (random names/IDs), mean `S` must be statistically ≤ `S` on
   easy tokens and well below `S` on reducible-hard tokens (CIs). Pair with a
   **positive control**: rare-but-determinate continuations where `S` must be high
   — so a collapsed committee can't pass by emitting low `S` everywhere.
4. **Circularity & alignment guards.** Never put live-policy compute on any axis.
   `+1`-shift control must give ρ≈0 (catches off-by-one). Reference `D` and Variant
   B require **byte-identical** tokenization, unit-checked.

---

## 8. Pseudocode (corrected)

```python
def set_mc_dropout(model, on):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train(on)   # dropout only; norms stay eval

def nll_t(logits, input_ids, mask):                  # ONE shared, shifted helper
    lp = logits[:, :-1].log_softmax(-1).float()
    tgt = input_ids[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return nll, mask[:, 1:]                           # everything indexed at predicted token

@torch.no_grad()
def surprise_jsd(model, batch, T=16):                 # STREAMING — never stack [T,B,L,V]
    set_mc_dropout(model, True)
    try:
        mean_p, mean_H = 0., 0.
        for _ in range(T):
            lp = model(batch).logits[:, :-1].log_softmax(-1).float()
            p  = lp.exp()
            mean_p = mean_p + p / T
            mean_H = mean_H + (-(p * lp).sum(-1)) / T
        H_mean = -(mean_p * mean_p.clamp_min(1e-12).log()).sum(-1)
        return (H_mean - mean_H).clamp_min(0)         # JSD per predicted-token position
    finally:
        set_mc_dropout(model, False)                  # deterministic again

@torch.no_grad()
def benefit_depth(model, lenses, batch, ids, mask, k):    # Variant A, tuned lens
    hs = model(batch, output_hidden_states=True).hidden_states   # [0]=emb, [i]=block-i out
    nll_full,_ = nll_t(model(batch).logits, ids, mask)           # FULL = real logits (C1)
    nll_k,_    = nll_t(lenses[k](hs[tier_layer[k]]), ids, mask)
    nll_k1,_   = nll_t(lenses[k+1](hs[tier_layer[k+1]]), ids, mask)
    return (nll_k - nll_k1)                                       # marginal, signed

# sanity asserts before trusting anything:
assert torch.allclose(model.lm_head(model.final_norm(hs[-1])), model(batch).logits, atol=1e-3)
# +1-shift control on S vs B must give rho ~ 0
```

All correlations: **document-level cluster bootstrap → CI lower bound vs threshold.**

---

## 9. Honest cost & quality bar (v1 lied about "one eval run")

Real cost ≈ `T` surprise forwards + the full benefit grid + per-tier lens fits +
reference pass + alignment = **days, not one run** (Stage 0 first to de-risk that).
**Minimum-quality bar below which a null is INCONCLUSIVE, not RED:** `T≥16`;
tuned (not raw) lens with QC passing; ≥**2–3k independent tokens in the hard×high-
compute cell** specifically; instrument-validity gate passed. Under-resourcing
produces a noisy null — never score that RED.

---

## 10. Outcomes → next move (note: GREEN ≠ build)

- **Stage 1 GREEN** (ρ(S,B) CI clears bar within strata, beats iso-FLOP uniform):
  → proceed to **Stage 2 tied-loop prototype**, NOT to the full build. R1 wall-clock
  and R2–R6 remain unvalidated.
- **STRONG GREEN:** additionally JSD beats single-model entropy → the committee earns
  its cost.
- **YELLOW** (overall holds, dies within strata, or ties entropy): signal may be real
  but a *committee* is unjustified — reconsider the cheap-scout design.
- **RED via "uniform ties S"** → R1: too little compute-hungry-token mass.
- **RED via instrument** (S≈0, gate failed) → **not** a real RED; switch to Plan-B
  committee.
- **RED via "S tracks D but not B"** → S finds hard, not *reducible*, tokens; rethink
  the signal.

**Stage 2 (the construct-valid test, required before build):** a small weight-tied
loop prototype running the **residual-surprise** signal against **loop-benefit**
(`NLL(loop 1) − NLL(loop K)` on the same tied block), with scouts at truncated loop
count and a between-mode committee. Far cheaper than the architecture it gates.
