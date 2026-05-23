#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${1:-logs/latest/checkpoint-final.pt}"
OUT_DIR="${2:-docs/reproducibility/results/eval_$(date +%Y%m%d_%H%M%S)}"
PROFILE="${3:-ci}"
MODE="${EVAL_MODE:-oracle}"
BACKEND="${EVAL_BACKEND:-mock}"

if [[ "${1:-}" == "--mode" ]]; then
  MODE="${2:-oracle}"
  CHECKPOINT="${3:-}"
  OUT_DIR="${4:-docs/reproducibility/results/eval_smoke}"
  PROFILE="${5:-smoke}"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"

echo "Running evaluation (profile=$PROFILE mode=$MODE)..."
"$PYTHON" scripts/evaluate_forge.py \
  --checkpoint "$CHECKPOINT" \
  --config configs/amc_forge_1b.yaml \
  --output-dir "$OUT_DIR" \
  --profile "$PROFILE" \
  --mode "$MODE" \
  --backend "$BACKEND"

"$PYTHON" tests/security/run_security_audit.py \
  --checkpoint "$CHECKPOINT" \
  --output docs/reproducibility/results/security_audit.json

echo "✅ Evaluation artifacts in $OUT_DIR and docs/reproducibility/results/"
