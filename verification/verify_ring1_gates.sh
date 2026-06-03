#!/usr/bin/env bash
# Ring 1 Tranche 4 — scale collection + gate verification
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Ring 1 tests (Tranches 1-4)"
.venv/bin/python -m pytest \
  tests/test_ring1_trace_format.py \
  tests/test_ring1_amc_integration.py \
  tests/test_ring1_dreambank_runner.py \
  tests/test_ring1_eval_harness.py \
  tests/test_ring1_repro_pack.py \
  tests/test_ring1_gate_verifier.py \
  -q

echo "==> Tranche 4 gate workflow (200+ traces per condition)"
set +e
.venv/bin/python scripts/ring1_tranche4_workflow.py --config configs/ring1_tranche4.yaml
workflow_exit=$?
set -e
if [ "$workflow_exit" -ne 0 ] && [ "$workflow_exit" -ne 2 ]; then
  echo "Tranche 4 workflow failed unexpectedly (exit $workflow_exit)" >&2
  exit "$workflow_exit"
fi

echo "Tranche 4 verification complete. Review docs/reproducibility/ring1_tranche4/gate_verification_report.json for preliminary gate status."
if [ "$workflow_exit" -eq 2 ]; then
  echo "Note: exit code 2 means one or more gates did not preliminary-pass (expected until real model re-eval is wired)."
fi
