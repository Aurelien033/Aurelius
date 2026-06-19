# Program Roadmap — Anticipatory Scout Transformer

Sequenced rounds with gates, dependencies, costs, and kill-criteria. Ordering
principle: **fail fast** — the cheapest tests most likely to kill the program run
first, so you never pay for a later round until the earlier one survives.

Artifacts: [research notes](anticipatory-transformer-research-notes.md) ·
[Round A spike](round-A-anisotropic-operator-spike.md) ·
[Stage 0/1 spec](experiment-1-spec.md) · [Stage 2 spec](experiment-2-spec.md).

---

## Two independent premises (this is why the order works)

The program rests on **two separable bets** that can be tested in parallel:

- **Operator bet:** an *anisotropic equilibrium loop* can be stable + trainable +
  benefit-rich. → **Round A.**
- **Signal bet:** a *cheap disagreement / residual signal* predicts where more compute
  reduces loss. → **Rounds B–C** (proxy) then **D** (real axis).

Either bet failing kills the program. Run A and B/C **concurrently** — fail fast on
whichever breaks first. **Round D is where the two bets converge** (the signal must
work *on the anisotropic operator*).

---

## The rounds

| # | Round | Question it answers | Method | Rough cost | KILL-criterion (RED) | Gates |
|---|---|---|---|---|---|---|
| **A0** | Anisotropic op — analytic pre-spike | Is the operator even stable/well-conditioned? | **CPU, no learning:** σ_max(full coupled J) via JᵀJ power-iteration, transient `sup_k‖J^k‖`, Kreiss/abscissa, `cond(I−J)=σ_max/σ_min`, on a 1-D ε × few-γ grid with injected attention-calibrated non-normality | **hours, CPU** | transient blow-up in k≤5 for all benefit-permitting ε, or cond(I−J) past fp32 floor → **RED-STABILITY/GRADIENT** | gates A1 |
| **A1** | Anisotropic op — learned confirm | Stable+trainable+benefit-capable together? | T1 planted + synthetic-benefit; phase-LINE over ε + 2 robustness checks; **T3 emergent anisotropy (attention+LoRA)**; gradients at k=1,2,3; joint N∈{2,8,32} Jacobian | 1–2 days, 1 GPU | no viable ε region after refinement (both D-arms); RED separated into STABILITY/GRADIENT/BENEFIT | gates D, G |
| **B** | Stage 0 pilot | Any signal at all (cheaply)? | Frozen model, 2–4 tiers, T≥16, ~25k hard-oversampled tokens, bootstrap over T | hours–1 day | pilot ρ(S,B) CI lower-bound below bar (and resourced) | gates C |
| **C** | Stage 1 proxy study | Does cheap disagreement track *depth* benefit? | Frozen model, tuned-lens early-exit; full stats battery + iso-FLOP + CALM | days | no monotone ρ; uniform ties S; S tracks D not B | gates D (signal side); **necessary not sufficient** |
| **D** | Stage 2 prototype | Does it track *loop* benefit on the **anisotropic** op, in NL, beyond trivial baselines, non-tautologically? | Train small anisotropic tied loop; ε-sweep; S1 primary; tautology gates; NL gating arm | weeks | RED across full ε range + anisotropic arm; or §4 tautology gates fail | gates F, G |
| **E** | R1 wall-clock roofline | Does tiered+scout beat dense on **measured tokens/sec**, scout cost charged? | 2–4 dense-GEMM tiers + CUDA graphs; fused committee; roofline benchmark | days | no ≥~1.15× tokens/sec at util floor; scout_time+sync ≥ saved_main_time | gates G (systems side); co-req with D §7a |
| **F** | R6 cold-start surrogate | Does the signal survive training *with* the compute-budget cost term? | Retrain a D-arm with annealed FLOPs/ponder cost + stop-grad on surprise path | days–week | surprise variance collapses to constant under cost pressure | gates G (training side) |
| **G** | Staged build | Does the full system work & pay off? | G1 tied-loop + surprise routing → G2 + LoRA expert prefetch → G3 + breathing attention | the project | each sub-stage has its own gate | terminal |

---

## Dependency DAG

```
        A ────────────┐
                       ├──► D ──► F ──┐
   B ──► C ────────────┘             ├──► G (G1 ► G2 ► G3)
                        E ───────────┘
```
- **A** and **B→C** run in parallel (operator bet vs signal bet).
- **D** requires **A** GREEN (operator viable) **and** **C** GREEN (signal premise).
- **E** can prototype alongside **D** (systems track); it is the §7a co-requisite.
- **F** requires a trained **D** arm to add the cost term.
- **G** requires **D + E + F** all GREEN, and proceeds in its own gated sub-stages.

---

## Decision tree / what each RED means

- **A RED** → novel core unbuildable → **stop**, or pivot to isotropic adaptive compute
  (≈ already published per [§10.2] → likely abandon the differentiator).
- **C RED** (real, not instrument/underpowered) → disagreement doesn't track benefit →
  **stop** or redesign the signal.
- **D RED** → loop premise or transfer fails → **stop**. (Instrument / over-damped /
  underpowered REDs are NOT real REDs — see Stage 2 §7.)
- **E RED** → no wall-clock win → the *efficiency* premise (R1) fails → **stop**, or
  rescope to FLOP-constrained-only deployments (a much narrower value proposition).
- **F RED** → cold-start collapse → training-curriculum research needed before any build.
- **G** → the build, itself staged with gates; experts (G2) and breathing attention (G3)
  are deferred precisely because they add the most-finicky failure modes.

---

## Spec status

- **Round A** — spec'd ([round-A-anisotropic-operator-spike.md](round-A-anisotropic-operator-spike.md)). *New.*
- **Rounds B, C** — spec'd ([experiment-1-spec.md](experiment-1-spec.md) v2).
- **Round D** — spec'd ([experiment-2-spec.md](experiment-2-spec.md) v2).
- **Round E** — outline only (Stage 2 §7a); needs a standalone kernels/benchmark spec.
- **Round F** — outline only (Stage 2 §7 conditional cap); needs a standalone training spec.
- **Round G** — not spec'd (gated; design in notes §6, §10.4). Spec only after D/E/F GREEN.

---

## The single most important thing on this page

Three reviews converged on it: **the program's fate is decided in Round A — and its
Phase A0 is an afternoon of CPU linear algebra.** The signal experiments (B–C) are cheap
and worth running in parallel, but if the anisotropic operator can't be stable +
trainable + benefit-capable at once, nothing downstream matters. **Run A0 first: it is
the single cheapest way to learn whether the whole idea is alive — hours, no GPU, no
training. A RED-STABILITY/GRADIENT there ends the program before you spend a dollar on
the ladder.**
