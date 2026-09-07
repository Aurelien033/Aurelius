# CascadeBank MVP Traceability

> Companion traceability for the plan at `../plans/2026-05-27-cascadebank-implementation.md`.
> Depends on DreamBank MVP traceability at `DREAMBANK_MVP_TRACEABILITY.md`.

## The Claim

> A routing policy built on top of an already-existing alignment gate
> (DreamBank `alpha_t`) can produce measurable quality/compute tradeoffs
> without adding any trainable parameters or any new forward-pass cost.

## What We Are Not Claiming

- A mixture-of-depths router (those train new per-token routers with their own
  parameters). Mixture-of-Depths (Raposo et al. 2024) trains one; we don't.
- A trained router of any kind.
- A full inference harness. We prove the policy decision exists and is
  measurable; the real serving-layer consumer is CB-06 (future tranche).
- Quality improvement. This is compute-budget routing; quality is measured
  against baseline by the next tranche.

## Prior Art Boundary

| System | Year | What they do | How we differ |
|---|---|---|---|
| Mixture-of-Depths (Raposo et al.) | 2024 | Trains per-token routers at every layer | We reuse an existing gate signal; zero new params |
| LLMCascade (Anagnostidis et al.) | 2024 | Trains layer-level exit classifiers | We don't train; we route on a downstream signal |
| AdaLLM (Bhat et al.) | 2024 | Learns a meta-controller for sample-adaptive compute | Our router is pure function of (alpha, confidence) thresholds |
| Mamba-of-Experts, Switch Transformers | various | MoE routing with learned top-k | DreamBank reads a single bank, not a MoE; we reuse its read signal |

CascadeBank's differentiation is that the routing signal already exists as a
side-effect of the DreamBank adapter forward pass. The marginal cost of making
a routing decision on that signal is ~0.

## Architecture Contract

```text
AMCTransformer.forward(ids, preference_bank)
    ├─ [layers + DreamBank adapter]          <- existing DreamBank
    ├─ AMCModelOutput:
    │    bank_alpha:       Tensor (B, T, 1)  <- gate activation
    │    bank_confidence:  Tensor (B, T, 1)  <- read confidence
    └─ logits

CascadeRouter.decision(bank_alpha, bank_confidence)
    ├─ if either is None            -> fallback (BALANCED)
    ├─ if mean(confidence) < 0.40   -> FAST
    ├─ if max(alpha) >= 0.70 and
    │   mean(confidence) >= 0.30    -> THOROUGH
    └─ else                         -> BALANCED
```

Thresholds are configurable via the frozen `CascadeRouterConfig` dataclass.

## Evidence Produced by MVP

| Tran | Evidence | Location |
|---|---|---|
| CB-01 | Router is parameter-free, gradient-free, fail-safe to BALANCED | `tests/inference/test_cascade_routing.py` (18 tests) |
| CB-01 | Threshold table matches plan's routing logic exactly | same file |
| CB-01 | Telemetry reports class distribution | same file |
| CB-02 | Consumes real `AMCModelOutput` shape `(B, T, 1)` | `tests/inference/test_cascade_routing_integration.py` |
| CB-02 | Empty-bank path never produces THOROUGH | same file |
| CB-02 | Disabled-DreamBank path returns FALLBACK | same file |
| CB-03 | Empty-bank decisions collapse to FAST (low-confidence) | `tests/eval/test_cascadebank_ablation.py` |
| CB-03 | Filled-bank with extreme router config produces ≥2 classes | same file |
| CB-03 | `CascadeAblationResult.to_dict()` is JSON-serializable | same file |

## Evidence NOT Produced by MVP (next tranche)

- Real serving-layer latency / FLOPs savings (requires GPU serving).
- Win-rate quality measurement: MT-Bench / AlpacaEval across
  FAST/BALANCED/THOROUGH routing policies.
- Paper-grade threshold: THOROUGH must outperform BALANCED by ≥+3pp while
  FAST degrades by ≤1pp and saves ≥10% p95 latency.

If those thresholds are not met, the failure is typically the gate signal
distribution — which means the next tranche becomes "train the bank to
produce router-friendly signal distributions" rather than "wire more
serving code."

## Invariants Baked Into the Router

```text
1. Zero new trainable parameters.
2. Missing DreamBank signals ALWAYS produce the configured fallback.
3. No modifications to src/memory/hlm_bank.py, src/model/hlm_bank_adapter.py,
   or src/model/amc_transformer.py — CascadeBank is purely additive.
4. Stage only CascadeBank files; no DreamBank or pre-existing drift bundled.
5. DreamBank-focused tests unchanged (64 tests, all still green).
```

## Files Owned by This MVP

| File | Lines | Role |
|---|---|---|
| `src/inference/cascade_routing.py` | 101 | `ComputePolicy` enum + `CascadeRouterConfig` + `CascadeRouter` |
| `src/eval/cascadebank_ablation.py` | ~80 | `CascadeAblationResult` + `run_cascade_ablation()` |
| `tests/inference/test_cascade_routing.py` | 160 | 18 unit tests |
| `tests/inference/test_cascade_routing_integration.py` | ~105 | Live `AMCTransformer` × router integration |
| `tests/eval/test_cascadebank_ablation.py` | ~95 | Ablation harness tests |
| `docs/reports/CASCADEBANK_MVP_TRACEABILITY.md` | this file |
| `docs/research-brief.md` | +25 lines | Section 6 appended |

## Recommended Next Step

CB-06: a real serving-layer harness that measures:

1. Greedy decode (no DreamBank, no routing) — baseline.
2. DreamBank-enabled decode with `BALANCED` routing — quality reference.
3. DreamBank-enabled decode with `CASCADE_FAST` routing — skip expensive
   post-processing when possible.
4. DreamBank-enabled decode with `CASCADE_THOROUGH` routing — escalate to
   chain-of-thought or self-consistency.

Metrics: preference win rate (MT-Bench, AlpacaEval), p50/p95 latency
per-decision-class, estimated FLOPs savings at observed decision distribution.
