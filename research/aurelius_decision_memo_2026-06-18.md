# Aurelius — State & Decision Memo
### 2026-06-18 · a one-page scorecard + the three forward options, with costs and priors

> Purpose: lay out the strategic call cleanly so it's a *decision*, not a drift. Honest-first. Full evidence:
> `aurelius_selection_arc_writeup_2026-06-17.md`.

---

## 1. The scorecard (what is actually established)

Tested *hard* on **FROZEN-BASE-v1** (Qwen2.5-1.5B), a functionally-verified F2/F3 repair gym, with
independent re-scoring throughout.

**FALSIFIED — powered:**
- Cheap per-layer estimator (`g_entropy` anti-informative; `g_gain` ≤ random).
- Cheap **per-task selector** from prefill activations — **null at k=2 *and* k=4 (3 powered nulls)**.
- **Co-train** ("make all skips cheap") → uniform mediocrity.
- A single-pass layer-skip win **that scales** — prune is ≥ dense at k=2 (~2 layers, modest) but **−12pp at
  k=4**.

**SURVIVES — real, replicated:**
- *Existence*: a good skip set exists for almost every input (true k=2 ceiling 96.7% > dense).
- *Mechanism*: non-additivity — specific poison layers (11/14/15) + adjacency, **not** "early layers."
- *Global structured prune*: ≥ dense, modest, pool-dependent (~2 layers of slack).
- **LayerDelta substitution (preliminary positive — the arc's first):** a cheap rank-32 surrogate replaces
  the k=4 best set and **matches dense** (+18.5pp over identity-skip; ≈ dense), ~10% cheaper. Those layers'
  deltas are **low-rank-approximable**.

**IN FLIGHT:** LayerDelta **frontier** (how many layers can be substituted?) + **out-of-family**
generalization (running/queued).

**BUILT, LAUNCH-READY:** skip-native (the `skip_layers` hook + sandwich/distill pretrainer, smoke-validated)
— needs only a tokenized corpus + a GO.

**One-line state:** the *strong* thesis (cheap, dynamic, per-input routing) is **not supported on a frozen
base** (3 nulls); the *defensible* core is a **non-additive structured-substitution** result (LayerDelta ≈
dense at less compute) + a methodology. A real, modest, honest position.

---

## 2. The three forward options

### A. Develop LayerDelta (build on the working positive)
- **What:** finish the frontier + generalization (in flight); then bigger ranks, a real depth-substituted
  kernel (actual deployed speedup), and a paper: *non-additive low-rank layer substitution*.
- **Cost:** LOW — frozen base, existing harness, ~hours of GPU per experiment.
- **Prior:** MODERATE–HIGH — it already works at k=4; the open question is *how far it scales / generalizes*
  (the in-flight runs answer exactly this).
- **Success looks like:** a smaller/faster frozen-base model (N layers → cheap surrogates) that matches
  dense out-of-family → a deployable result + a clean paper.
- **Caveat:** it is *static global substitution* — it does **not** revive the dynamic per-input thesis.

### B. Bet on skip-native (the only thesis-revival path)
- **What:** train a from-scratch model *built to route* (the launch-ready pretrainer) + a dense-twin control;
  test whether routability can be **trained in** where a frozen base never had it.
- **Cost:** REAL — ~$40–100, ~days, tokenize a 2B open corpus, 2 runs. (Build is done.)
- **Prior:** LOW — 3 powered nulls weigh heavily; the *only* reason for hope is structural (frozen ≠ built-
  to-route). More likely to **confirm** the negative than overturn it.
- **Success looks like:** a model that organizes its own compute per input → revives CLM-001 (big). A null
  **decisively closes** the thesis (a stronger negative).
- **Caveat:** 92M scale; a positive needs replication at ~0.5–1B before any generality claim.

### C. Consolidate + publish (lock in what exists)
- **What:** finalize the paper (abstract + related work + results incl. LayerDelta + methodology are
  drafted); drop in the in-flight numbers; submit.
- **Cost:** LOW — CPU, days of writing, **no GPU**.
- **Prior:** N/A — the result *exists*; this is a sure thing.
- **Success looks like:** a published, rigorous **powered negative** + structured-substitution positive +
  a verification methodology — a genuine contribution *today*.

---

## 3. Recommendation

These are **not mutually exclusive** — the honest sequencing:

1. **Finish A's in-flight runs** (frontier + generalization) — cheap, imminent, and they decide whether
   LayerDelta is a *headline* (scales + generalizes out-of-family) or a *footnote* (a few-layer, gym-bound
   trick). Do this first; it's nearly free.
2. **Do C regardless.** The paper is ~80% done and the result is real *now*. Consolidating it is the
   **floor** — a guaranteed contribution that doesn't depend on any further experiment. No reason to gamble
   the sure thing.
3. **Treat B (skip-native) as a deliberate, separately-decided big bet.** Given the low prior, gate it on a
   clear question: *do you want the dynamic-routing thesis decisively answered* (worth the $ + days, for
   closure or a low-odds revival), *or is the structured-substitution result enough*? It's **launch-ready**,
   so the decision can wait until GPU quota resets and the LayerDelta results are in — they'll sharpen the
   call (if LayerDelta scales + generalizes, the static result may be enough and skip-native is optional; if
   LayerDelta is limited, skip-native becomes the more interesting swing).

**My honest lean:** **A + C now** (finish LayerDelta, consolidate the paper) → **B is worth doing once, for
closure, but is not urgent** and should be chosen with eyes open about the low prior. The program is in a
genuinely good place: a rigorous negative, a real (small) positive, a clean mechanism, and a methodology —
publishable today, with one cheap experiment (the frontier/generalization) potentially upgrading the
positive, and one deliberate big bet (skip-native) available if you want the thesis question closed for good.
