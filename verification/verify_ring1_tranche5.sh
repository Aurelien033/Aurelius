#!/usr/bin/env bash
# Ring 1 Tranche 5 — checkpoint-attached DreamBank + measured lift
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Ring 1 tests (Tranches 1-5)"
.venv/bin/python -m pytest \
  tests/test_ring1_trace_format.py \
  tests/test_ring1_amc_integration.py \
  tests/test_ring1_dreambank_runner.py \
  tests/test_ring1_eval_harness.py \
  tests/test_ring1_repro_pack.py \
  tests/test_ring1_gate_verifier.py \
  tests/test_ring1_model_loader.py \
  tests/test_ring1_measured_lift.py \
  -q

echo "==> Tranche 5 measured-lift workflow"
set +e
.venv/bin/python scripts/ring1_tranche5_workflow.py --config configs/ring1_tranche5.yaml
workflow_exit=$?
set -e
if [ "$workflow_exit" -ne 0 ] && [ "$workflow_exit" -ne 2 ]; then
  echo "Tranche 5 workflow failed unexpectedly (exit $workflow_exit)" >&2
  exit "$workflow_exit"
fi

echo "Tranche 5 verification complete. Review docs/reproducibility/ring1_tranche5/gate_verification_report.json"
if [ "$workflow_exit" -eq 2 ]; then
  echo "Note: exit code 2 means one or more gates did not preliminary-pass."
fi
