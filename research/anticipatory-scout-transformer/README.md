# Anticipatory Scout Transformer — validation scripts

Runnable experiments for the staged validation ladder. Specs:
[research notes](anticipatory-transformer-research-notes.md) ·
[Round A](round-A-anisotropic-operator-spike.md) ·
[Stage 0/1](experiment-1-spec.md) · [Stage 2](experiment-2-spec.md) ·
[roadmap](program-roadmap.md).

> **These scripts were authored on a machine without Python/torch and have NOT been
> executed.** First action on the target box: a smoke test (tiny dims, 1 step) per
> script. Treat any runtime error as expected debugging, not a design failure.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

## Run order (two independent bets — A and B/C can run in parallel)

| Script | Round | Needs | Cost | Decides |
|---|---|---|---|---|
| `round_A0_prespike.py` | A0 | numpy (CPU) | minutes | operator stable/well-conditioned? (kill-test) |
| `round_A1_learned_spike.py` | A1 | torch (1 GPU ok on CPU too) | ~1–2 h | learned anisotropic op stable+trainable+benefit-capable? |
| `stage01_signal_study.py` | B/C | torch+transformers + a pretrained LM | hours | does cheap disagreement predict where compute helps? |
| `stage2_tied_loop_prototype.py` | D | torch (1 GPU) | hours–day | does loop-disagreement predict loop-benefit? (build gate) |

**Gate logic:** a RED-STABILITY/GRADIENT from A0/A1 stops the program. A RED from
Stage 0/1 (real, not instrument/underpowered) stops the signal bet. Stage 2 is the
only thing that green-lights a build, and only conditionally (R6 unobserved until
trained with the cost term). See each spec's decision rule.

## What each script prints

- **A0 / A1:** σ_max(J) (the *singular-value* contraction certificate — NOT
  eigenvalues), transient amplification, cond(I−J), and benefit metrics (iteration-count
  CV, along-manifold work fraction). Decision: GREEN / YELLOW / RED-{STABILITY,GRADIENT,
  BENEFIT}.
- **Stage 0/1:** Spearman(surprise, benefit) with **document-level bootstrap CIs**, the
  **tail metric** (NLL captured by top-x%-by-surprise vs the oracle ceiling), the
  **iso-FLOP** policy comparison, and the **CALM single-pass-entropy** baseline.
- **Stage 2:** loop-benefit vs residual-surprise (S1) and committee disagreement (S2),
  with the **tautology controls** (partial-correlation on current-NLL and
  truncated-loop-NLL) and the same bootstrap/tail machinery.

## Key correctness notes baked in (from the adversarial reviews)

- Contraction certificate is **σ_max via power-iteration on JᵀJ / dense SVD**, on the
  **full coupled Jacobian** — eigenvalues do NOT bound transient growth for non-normal J.
- All correlations bootstrap over **documents/instances**, not tokens (autocorrelation),
  and gate on the **CI lower bound**, not the point estimate.
- Iso-FLOP is a **true policy simulation** charging the scout's own compute and crediting
  signed realized benefit (including regressions).
- Instrument-validity gate: if MC-dropout gives ~0 signal (dropout p=0), that's an
  **instrument failure → Plan-B**, NOT a RED.

## Code-review status

All six scripts were adversarially code-reviewed (none executed on the authoring box).
Fixes applied: Stage 2 bootstrap-control crash + signed-tail denominator + S2 relabel;
Round E single-tier guard; Round F `k_scout` guard; A0 wrong-subspace projector + path-based
along-fraction + γ-bisection assert; A1 param seeding + transient-norm floor; Stage 0/1
logit-lens QC gate + oracle headroom + verdict-label fix. Known residual limitations (noted
in-code, fix before trusting a verdict): Stage 0/1 uses a raw logit lens (use a **tuned**
lens); Stage 2's second nuisance control coincides with current-NLL (add an **independent
readout**) and S2 is **dropout-only m=1** (use ≥2 seeded nets for the real committee).

## Smoke-test order (fastest first — validates the crash guards, then logic)

```bash
python round_A0_prespike.py                         # numpy only; check along_frac is NOT ~1.0 everywhere
python round_E_roofline.py && python round_E_roofline.py --tiers 10   # single-tier edge case
python round_F_coldstart.py --K 6 && python round_F_coldstart.py --K 6 --k_scout 5   # must assert, not IndexError
python round_A1_learned_spike.py --seed 0           # run twice; outputs must MATCH (reproducibility)
python stage2_tied_loop_prototype.py                # needs torch; eval batch=1 default
python stage01_signal_study.py --model <tiny> --data <small.txt>   # heaviest; QC assert must pass
```
