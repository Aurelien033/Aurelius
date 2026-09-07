#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${1:-}"
OUT="${REPO_ROOT}/docs/reproducibility/results/ablation_scores.jsonl"
MODE="${ABLATION_MODE:-oracle}"
BACKEND="${ABLATION_BACKEND:-mock}"
PYTHON="${REPO_ROOT}/.venv/bin/python"

ARGS=(--mode "$MODE" --output "$OUT")
if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ "$MODE" == "engine" ]]; then
  ARGS+=(--backend "$BACKEND")
fi

echo "Running ablation study → $OUT (mode=$MODE)"
"$PYTHON" scripts/run_ablation.py "${ARGS[@]}"
echo "✅ Ablation complete."
