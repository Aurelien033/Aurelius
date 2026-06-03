# Tranche 5 Completion Report

**Tranche:** 5  
**Completed On:** 2026-05-31  
**Lead:** Cursor Agent (Ring 1 Tranche 5)  
**Branch:** `feature/ring1-tranche5-20260531`

## Artifacts Produced

### New files
- `src/eval/ring1_model_loader.py` — resolve legacy step dirs, infer AMC config, load weights, SHA256
- `src/eval/ring1_measured_lift.py` — bank rehydration + logit-delta measured lift
- `scripts/ring1_tranche5_workflow.py` — checkpoint-attached DreamBank + measured lift gates
- `configs/ring1_tranche5.yaml` — checkpoint step, measured lift scale, trace reuse
- `verification/verify_ring1_tranche5.sh` — pytest + Tranche 5 workflow
- `tests/test_ring1_model_loader.py` — 4 tests
- `tests/test_ring1_measured_lift.py` — 3 tests
- `docs/tranche5/README.md` — usage instructions

### Extended files
- `src/eval/ring1_trace_logger.py` — `compute_checkpoint_sha256()` hashes `model.safetensors` in dirs
- `src/eval/ring1_eval_harness.py` — `run_eval_harness_dual(..., measured_lift=...)`

### Generated artifacts (local)
- `docs/reproducibility/ring1_tranche5/gate_verification_report.json`
- `docs/reproducibility/ring1_tranche5/workflow_summary.json`
- `docs/reproducibility/ring1_tranche5/dreambank_run/` — real param hash proof
- `docs/reproducibility/ring1_tranche5/ring1-repro-*.tar.gz`

## Preliminary Gate Results

| Gate | Status | Key evidence |
|------|--------|--------------|
| **R1-GA** | **PASS** | ≥200 traces/condition; memory audit; holdout lift ≥5pp |
| **R1-GB** | **PASS** | Real param hash unchanged; **measured lift = 3.12pp** (threshold 1.0pp) |
| **R1-GC** | **PASS** | Repro pack validates with real checkpoint SHA |

Workflow exit code **0** — all gates preliminary-pass.

## What Actually Works Now

- **Checkpoint loading** from `checkpoints/aurelius-1.3b/step-0000002/model.safetensors` with config inference
- **Zero-grad proof** with real param hash (not `no-model-attached`)
- **Measured DreamBank lift** via `run_ablation` bank-on vs bank-off on held-out trace prompts
- **33/33 Ring 1 tests** green across Tranches 1–5

## Open Items

1. **Full 1B checkpoint** — current weights are a tiny 2-layer smoke checkpoint (~462KB); production gate needs forge-scale weights
2. **End-to-end task re-eval** — lift is logit-delta-derived, not full agent re-run on held-out seeds
3. **Maintainer sign-off** — preliminary automation complete; official pass still requires human review

## Anti-Scope Evidence

- No modifications to `src/alignment/dreambank.py`, `src/model/*`, training, serving, Tier-3, LPD, PMA
- Checkpoint is read-only via eval-layer loader

## Sign-off

- Maintainer: _pending review_
- Date: _pending_
