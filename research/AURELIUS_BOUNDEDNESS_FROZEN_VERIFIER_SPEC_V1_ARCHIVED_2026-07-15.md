# AURELIUS — BOUNDEDNESS & FROZEN-VERIFIER SELF-IMPROVEMENT SPEC
## Evidence-gated design constraint for the "self-improving" program claim
## Generated 2026-07-15 · Author: Hermes Agent
## Grounded in: RSI_MASTER_REPORT.md + analyses_rsi/batch_A/{2505.22954,2510.10232,2603.06333}
##                    + 2607.07663 (verification hierarchy) + 2607.11022 (verifier false-positives)
##                    + 2607.04277 (quasi-introspection floor) + 2607.12166 (SAE causal audit)

================================================================
0. SCOPE & PURPOSE
================================================================

This spec converts the RSI research dossier into an ENFORCEABLE design constraint for
Aurelius. It defines what "self-improving" may and may not mean, the frozen boundaries,
the acceptance gate, the drift watchdog, and the verifier-false-positive probe. Every
number below is taken from the deep-read papers on disk (not invented).

It exists because the literature establishes a hard empirical floor:
  - 2607.04277: current LLMs have only QUASI-introspection (no fixed-point self-
    modification, feedforward, no complete self-access). True autonomous RSI is NOT
    achievable in current LLMs. 2607.12254 agrees: "not evidence RSI has been achieved."
  - No paper in the dossier demonstrates WEIGHT-LEVEL RSI. None proves alignment — only
    measured-improvement risk (batch_A summary).

Therefore Aurelius's self-improvement is scoped as: BOUNDED, VERIFIER-GATED,
INFERENCE-TIME / SCAFFOLDING-LEVEL self-refinement. NOT weight-level recursion.

================================================================
1. THE FROZEN BOUNDARY (non-negotiable)
================================================================

PRINCIPLE (from Darwin Gödel Machine 2505.22954):
  Keep the intelligence source FROZEN. Only evolve the surrounding code/tools/prompts/
  verifier scaffolding. DGM demonstrated 20.0% → 50.0% SWE-bench / 14.2% → 30.7% Polyglot
  by keeping the foundation model frozen and evolving only the agent's own repo, gated by
  an external evaluator.

AURELIUS ENACTMENT:
  1.1 BASE WEIGHTS are immutable for the life of a version. The clean-teacher firewall
      means: public releases use weights that were NEVER modified by a self-improvement
      loop. Any self-modification operates on TOOLING, not weights.
  1.2 The GenRM → PRM → Lane-A/B/C GRADER is a separate, version-pinned artifact. It is
      EXTERNAL to the model under improvement and FROZEN against self-rewrite (per Red
      Queen Gödel Machine 2606.26294 lesson: if the evaluator co-evolves with the agent,
      you need a meta-stable loop — we avoid that by pinning the grader).
  1.3 A self-modification cycle may propose changes to: prompts, tool definitions,
      verifier scaffolding, routing policy, repair templates, best-of-N sampling config.
      It may NOT propose base-weight deltas. If a proposal touches weights, it is REJECTED
      at intake (hard rule, no gate needed).

================================================================
2. ACCEPTANCE GATE — SGM STATISTICAL RISK BUDGET (2510.10232)
================================================================

PRINCIPLE: Replace "provably beneficial" (impossible for LLMs) with "statistically
certified at confidence 1−δ." Admit a modification only when a paired evaluation
certifies improvement under a fixed global error budget δ.

CONCRETE FORM (from batch_A/2510.10232.md):
  2.1 Each proposal θ'_t is paired-evaluated against incumbent θ_t on a held-out eval
      set; bounded normalized differences Δ_i ∈ [−1,1].
  2.2 Per-edit safety: Pr(harmful accept at t) ≤ δ_t.
      Cumulative safety: Pr(∃t≤T : harmful accept) ≤ δ.
  2.3 Accept iff Hoeffding one-sided LCB μ̂ − (b−a)√(½n · ln(1/δ_t)) > 0,
      OR e-value wealth W_t ≥ 1/δ (anytime variant; supermartingale, Ville's inequality).
  2.4 CTHS budget spend: δ_t = δ / (t · H_B), H_B=Σ_{i=1}^B 1/i — concentrate error budget
      on promising edits (compute-saving on 6GB).
  2.5 SGM correctness check: bounds require bounded differences + truthful paired
      harness. If the harness is misspecified, bounds are invalid → the harness (PRM)
      must itself pass the 2607.11022 false-positive probe (see §4).

AURELIUS ENACTMENT:
  - Set global δ (recommend δ = 0.05 for public releases; δ = 0.1 acceptable for internal
    experimentation per SGM's own experiment value).
  - Lane-A/B/C candidate = proposal θ'_t. GenRM/PRM score delta = paired Δ_i. The SGM
    Gate = acceptance controller with the hard δ budget. No candidate reaches a released
    version without SGM acceptance.

================================================================
3. DRIFT WATCHDOG — SAHOO GOAL DRIFT INDEX (2603.06333)
================================================================

PRINCIPLE: SGM answers "is this edit better (statistically)?" SAHOO answers "is this edit
drifting from goals?" Stack them. SGM gates acceptance; SAHOO watches drift across cycles
and HARD-STOPS.

CONCRETE FORM (from batch_A/2603.06333.md):
  3.1 Goal Drift Index GDI: learned multi-signal detector. Component weights (from paper):
      semantic 0.38, distributional 0.29, structural 0.21, lexical 0.12.
  3.2 Constraint Preservation Score CPS: syntactic correctness + non-hallucination;
      CPS = 1.00 = perfect.
  3.3 HARD STOP conditions (verbatim from paper):
        (a) GDI < 0.01 for 3 consecutive iterations → converge/stop.
        (b) regression risk > calibrated threshold → stop.
        (c) any cycle yields CPS = 0 → abort.
        (d) max cycle count reached → stop.
        (e) GDI > 0.44 (calibrated threshold) → HARD STOP.
  3.4 Empirically: +18.3% code quality at GDI=0.320 (below 0.44); GDI stabilizes by
      cycles 3–5; mean convergence at cycle 8.8.

AURELIUS ENACTMENT:
  - Feed GenRM/PRM semantic + structural signals into a GDI-like index each cycle.
  - HARD STOP at GDI > 0.44 or CPS = 0. These are non-negotiable halt conditions.
  - SAHOO safety is calibration-dependent (18-task calibration in the paper) — pair with
    SGM's statistical guarantee + clean-teacher firewall. SAHOO monitors; it does not
    prevent a misaligned base (the firewall does).

================================================================
4. VERIFIER-FALSE-POSITIVE PROBE (2607.11022)
================================================================

PRINCIPLE: Natural-language verifiers produce FALSE POSITIVES during RL (the paper:
"When the Reward Suite Is Leaky" — preregistered causal contrast of verifier false
positives). If the PRM rewards the wrong thing, best-of-N optimizes the wrong thing.

AURELIUS ENACTMENT:
  4.1 Build a held-out probe suite of inputs where the superficially-correct-looking
      output is actually wrong (verifier-trap cases).
  4.2 Before any Lane-B/C promotion, run the PRM on the probe suite. If false-positive
      rate exceeds a threshold (recommend < 2% on the trap set), the Lane change is
      BLOCKED pending PRM repair.
  4.3 The probe suite must be versioned alongside the grader (§1.2) so regressions are
      detectable.

================================================================
5. HUMAN-IN-THE-LOOP (top of the verification hierarchy)
================================================================

From 2607.07663 (survey): verification signals rank Formal verifier (strongest) → PRM
(mid) → intrinsic self-assessment (weakest, collapse-prone). The human sets research
DIRECTION at the top of the hierarchy.

AURELIUS ENACTMENT:
  5.1 Any self-modification that changes the GRADER, the δ budget, or the GDI threshold
      requires explicit human approval (these are meta-parameters, not cycle candidates).
  5.2 Routine cycle candidates (prompts/tools/scaffolding) may be auto-gated by SGM+SAHOO,
      but their accumulated diff is logged and human-reviewable (DGM traceability lesson).
  5.3 The "self-improving" claim is reviewed against this spec before any public statement.

================================================================
6. BOUNDEDNESS STATEMENT (for the program doc)
================================================================

Per 1312.6764 (Bounded RSI) / 1805.06610 (RSI efficiency): state the bound explicitly.

  "Aurelius self-improvement is BOUNDED: (i) base weights are frozen and clean (no
   weight-level recursion); (ii) every proposed change is verifier-gated by a statistical
   risk budget (SGM, Pr(harmful accept) ≤ δ); (iii) a drift watchdog hard-stops on
   GDI > 0.44 or CPS = 0 (SAHOO); (iv) the grader is external and version-pinned; (v) the
   verifier is probed for false positives (2607.11022); (vi) meta-parameters require human
   approval. This is inference-time / scaffolding-level self-refinement, not open-ended
   recursive self-modification. It is reproducible and frontier-credible; it does not
   claim autonomous RSI, which current LLMs cannot perform (2607.04277)."

================================================================
7. WHAT THIS SPEC MAKES DEFENSIBLE
================================================================

- The "self-improving" claim is now evidence-gated and honest: bounded verifier-gated
  inference-time refinement, validated by 2602.03094 (maj@N/repair) and SCOPE (frozen
  self-judge).
- The clean-teacher firewall has a concrete mechanism: frozen weights + SGM gate + SAHOO
  watchdog + external pinned grader.
- Reward-hacking risk is actively probed (§4), not assumed away.
- On 6GB hardware: SGM paired-eval and SAHOO re-scoring are cheap (small eval sets,
  PRM as scorer) — the spec is sized for the constraint, not just for frontier labs.

================================================================
8. OPEN ITEMS / FUTURE WORK
================================================================

- Calibrate GDI on Aurelius's own domains (the paper used 18 tasks; replicate with your
  eval set).
- Implement the SGM Gate as a standalone module in the verifier path.
- Decide δ per release tier (public δ=0.05, internal δ=0.1).
- Periodically re-run the 2607.11022 probe as the PRM evolves.

================================================================
END OF SPEC
================================================================
