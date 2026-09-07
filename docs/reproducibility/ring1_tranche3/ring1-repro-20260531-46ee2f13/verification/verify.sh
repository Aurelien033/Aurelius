#!/usr/bin/env bash
# Ring 1 Tranche 3 verification — pytest + workflow smoke + pack validation
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Ring 1 trace + integration tests"
.venv/bin/python -m pytest \
  tests/test_ring1_trace_format.py \
  tests/test_ring1_amc_integration.py \
  tests/test_ring1_dreambank_runner.py \
  tests/test_ring1_eval_harness.py \
  tests/test_ring1_repro_pack.py \
  -q

echo "==> Tranche 3 workflow (requires Tranche 2 traces)"
if [[ ! -f data/ring1_traces/tranche2/traces.jsonl ]]; then
  echo "Generating Tranche 2 traces..."
  .venv/bin/python scripts/ring1_trace_collector.py \
    --config configs/ring1_tranche2.yaml \
    --num_traces 100 \
    --output_dir data/ring1_traces/tranche2
fi

.venv/bin/python scripts/ring1_tranche3_workflow.py --config configs/ring1_tranche3.yaml

echo "==> Validate repro pack"
.venv/bin/python scripts/ring1_tranche3_workflow.py \
  --config configs/ring1_tranche3.yaml \
  --validate-only

echo "Ring 1 Tranche 3 verification passed."
