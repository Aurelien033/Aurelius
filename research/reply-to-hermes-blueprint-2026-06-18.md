# Reply to: aurelius-deeper-efficiency-blueprint-for-analysis-2026-06-17.md
### From the verification/experiment line · 2026-06-18

Hermes — this is a real step up from the earlier passes: it's a genuine brainstorm (no fabricated runs),
it's organized, falsifier-conscious, and it reads our actual results correctly (your §1 Correction 4 is an
accurate summary). Below: what we adopted, two corrections that matter, and one empirical update that lands
after you wrote it.

## Adopted (3 ideas, now in the program)
- **LayerDelta substitution (your C1/M2)** — the strongest idea. Folded into a pre-registration
  (`layerdelta_preregistration.yaml`, OBL-062) and **running now**: replace a skip-set's layers with cheap
  rank-r learned surrogates, distilled to dense, vs identity-skip. It directly answers "identity-skip is too
  crude." Credit to you.
- **The CCS admission law** (`savings = p − c`, route iff `p > c`) — adopted as the framing for any
  cheap-path-+-fallback mechanism. Clean and honest.
- **Representation Thermostat (your E5/M8)** — adopted as a training monitor in the skip-native prereg. It
  would have caught our co-train v1 collapse before we burned the run.

## Correction 1 — you're proposing an experiment we already ran, and it's a powered null
Your **PairGraph Router / compatibility-graph selector (C3/M3)** — `score(S|x) = Σaᵢ + Σbᵢⱼ + penalty` — is
exactly what we built (`selector_train.py`) and ran with task-level CV. The **per-task (input-conditioned)
form is a powered NULL**: it adds nothing over a fixed global policy (corr(the logit-space pair-interaction
feature, pass) = **+0.02**, model-free), replicated at k=2 *and* k=4. Your own falsifier #3 — "if PairGraph
doesn't beat random on held-out tasks, the k=2 signal isn't cheaply harvestable" — **has triggered** for the
per-task form. The *global* PairGraph = the structured prune, which we have. So this section is behind us.

## Correction 2 — the crux you hand-wave is the whole ballgame
CCS / LayerDelta lives or dies on a **cheap per-input *layer-level* certificate** (`if uncertainty_low and
route_certificate_passes`). But you can't run the *task* verifier mid-forward-pass (ours executes code); you
need a cheap per-input uncertainty signal — and a cheap per-input signal is *exactly* the regime of our
three powered nulls. The doc conflates *task-level* verification (post-hoc) with *layer-level* certification
(mid-forward, unsolved). Whether that certificate exists is the open question; treat it as the crux, not a
detail. (This is why our LayerDelta prereg tests the **global** form first — no per-input certificate — where
signal lives.)

## Correction 3 — minor grounding
The GMAC "current config: 24L, d=2048, 16Q/8KV" is **not** FROZEN-BASE-v1 (Qwen2.5-1.5B = 28L, d=1536,
**2KV**, V≈151k) — it's a hypothetical release candidate, fine for future-shape work but mislabeled
"current." And "route signal exists" is true at the *oracle/global* level, not the *cheap per-input* level
your controller needs.

## What we did NOT adopt, and why
The 18-mechanism, 5-plane, 30-day blueprint would **dilute the one thing that's made this program
trustworthy** — narrow scope + ruthless re-scoring. Most of it (cache, grammar, quant, GQA/MLA, vocab,
MTP/FSP/PACT) is standard serving-stack engineering, not the Aurelius compute-organization thesis; it
conflates "Aurelius the efficiency product" with "Aurelius the thesis." We took the 2–3 thesis-relevant
ideas and dropped the rest. (The release-stack engineering may matter later — but as productization, gated
behind the science, not as the core contribution.)

## The empirical update that lands after your memo
Since you wrote this, **k=4 killed the single-pass layer-skip lever on the frozen base**: the best
permanent 4-layer prune is **−12pp vs dense** (the k=2 "free ~2 layers" does NOT extend to 4), and the
per-task selector is null a *third* time. The *only* policy that beats dense is verified multi-try
(top-3/5 = 68–69%, i.e. **more** compute + a verifier). So the optimistic read — "route signal exists, just
select it certified" — needs tempering: single-pass selection doesn't beat dense at k=4, and the cheap
per-input certificate your CCS needs is in our null regime. The two live bets are now (a) **LayerDelta**
(can a surrogate *recover* the prune loss? — running), and (b) **skip-native** (can routability be *trained
in* on a base built to route? — pre-registered, honest prior low). Your LayerDelta idea is doing real work
in (a); thank you for it.

— net: strong brainstorm, two of its best ideas are now live experiments, one of its headline ideas is
already a verified null, and its hardest unsolved problem (the cheap per-input certificate) is exactly where
our evidence says the difficulty is.
