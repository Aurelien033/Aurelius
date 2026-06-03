# Tranche 3 Completion Report

**Tranche:** 3  
**Completed On:** 2026-05-31  
**Lead:** Cursor Agent (Ring 1 Tranche 3)

## Artifacts Produced

### New files
- `src/eval/ring1_dreambank_runner.py` — trace → DreamBank sleep + zero-grad param hash proof
- `src/eval/ring1_eval_harness.py` — metrics, ablation table, bootstrap CI, lift estimate
- `src/eval/ring1_repro_pack.py` — reproducibility pack builder + validator
- `scripts/ring1_tranche3_workflow.py` — end-to-end workflow CLI
- `configs/ring1_tranche3.yaml` — Tranche 3 configuration
- `verification/verify_ring1_tranche3.sh` — pytest + workflow + pack validation
- `tests/test_ring1_dreambank_runner.py` — 4 tests
- `tests/test_ring1_eval_harness.py` — 4 tests
- `tests/test_ring1_repro_pack.py` — 2 tests
- `docs/tranche3/README.md` — usage instructions

### Generated artifacts (local)
- `docs/reproducibility/ring1_tranche3/workflow_summary.json`
- `docs/reproducibility/ring1_tranche3/dreambank_run/` — sleep logs + hash proof
- `docs/reproducibility/ring1_tranche3/metrics/` — ablation + CI JSON
- `docs/reproducibility/ring1_tranche3/ring1-repro-*.tar.gz` — full repro pack

## What Actually Works Now

- **DreamBank sleep** on Tranche 2 traces (16 writes, bank_fill=14 in smoke run)
- **Zero-grad proof** via `pre_cycle_model_hash.txt` / `post_cycle_model_hash.txt` (unchanged when model attached)
- **Eval harness** — AMC vs no-memory ablation table on 100 traces with bootstrap CI
- **Repro pack** — spec-aligned layout with `verification/verify.sh`
- **22/22 Ring 1 tests** green across Tranches 1–3

## Open Items

- DreamBank lift uses smoke proxy estimator; Tranche 4 needs real held-out re-evaluation with model + bank
- No real checkpoint attached (`param_hash_unchanged=true` with `no-model-attached`)
- Smoke scale (100 traces); gate runs require 200+ per condition

## Anti-Scope Evidence

- No modifications to `src/alignment/dreambank.py`, `src/model/*`, `src/memory/hlm_bank.py`, serving, Tier-3, LPD, PMA
- Branch: `feature/ring1-tranche3-20260531`

## Tranche 4 Handoff

Scale to 200+ traces per condition, wire real `AMCTransformer` + `run_ablation` logit delta, run gate verification (R1-GA/GB/GC), replace lift proxy with measured post-sleep held-out eval.

## Sign-off

- Maintainer: _pending review_
- Date: _pending_
