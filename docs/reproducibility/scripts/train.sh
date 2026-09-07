#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-configs/amc_forge_1b.yaml}"
RUN="${2:-forge_1b_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/${RUN}"
GPUS="${3:-4}"
PYTHON="${REPO_ROOT}/.venv/bin/python"

mkdir -p "$LOG_DIR"

echo "╔════════════════════════════════════════════╗"
echo "║  Aurelius AMC Training (repro bundle)     ║"
echo "╚════════════════════════════════════════════╝"
echo " Config : $CONFIG"
echo " Run    : $RUN"
echo " Logs   : $LOG_DIR"
echo " GPUs   : $GPUS"
echo ""

"$PYTHON" scripts/count_params.py --config "$CONFIG" | tee "$LOG_DIR/param_count.log"

if [[ "$GPUS" -gt 1 ]] && command -v deepspeed >/dev/null 2>&1; then
  deepspeed --num_gpus "$GPUS" \
    src/training/launch_amc_training.py \
    --config "$CONFIG" \
    --data "${REPO_ROOT}/data/tokenized/amc_forge_1b" \
    --log_dir "$LOG_DIR" \
    --deepspeed configs/deepspeed_zero2.json \
    2>&1 | tee "$LOG_DIR/training.log"
else
  "$PYTHON" src/training/launch_amc_training.py \
    --config "$CONFIG" \
    --data "${REPO_ROOT}/data/tokenized/amc_forge_1b" \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/training.log"
fi

echo ""
echo "✅ Training log: $LOG_DIR/training.log"
echo "   Next: bash docs/reproducibility/scripts/evaluate.sh $LOG_DIR/checkpoint-final.pt"
