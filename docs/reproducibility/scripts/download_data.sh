#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-data/tokenized/amc_forge_1b}"
echo "Preparing tokenized training data → ${OUT_DIR}"

if [[ -f scripts/prepare_data.sh ]]; then
  bash scripts/prepare_data.sh
else
  "${REPO_ROOT}/.venv/bin/python" scripts/prepare_training_data.py \
    --output-dir "$OUT_DIR" \
    --vocab-size 128000 \
    --max-seq-len 2048
fi

echo "✅ Data preparation finished (or resumed)."
