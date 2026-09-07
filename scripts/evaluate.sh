#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-logs/latest/checkpoints/checkpoint-final.pt}"
OUT_DIR="${2:-logs/eval_$(date +%Y%m%d_%H%M%S)}"
PROFILE="${3:-ci}"

echo "╔════════════════════════════════════════════╗"
echo "║    Aurelius Forge Evaluation (T22)        ║"
echo "╚════════════════════════════════════════════╝"
echo " Checkpoint: $CHECKPOINT"
echo " Output    : $OUT_DIR"
echo " Profile   : $PROFILE"
echo ""

python "${REPO_ROOT}/scripts/evaluate_forge.py" \
  --checkpoint "$CHECKPOINT" \
  --config "${REPO_ROOT}/configs/amc_forge_1b.yaml" \
  --output-dir "$OUT_DIR" \
  --profile "$PROFILE" \
  --mode "${EVAL_MODE:-engine}" \
  --backend "${EVAL_BACKEND:-mock}"

echo ""
echo "═══════════════════════════════════════════"
echo "Summary"
echo "═══════════════════════════════════════════"
python - <<PY
import json
from pathlib import Path

out = Path("${OUT_DIR}")
summary_path = out / "summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for name, metrics in summary.get("results", {}).items():
        score = metrics.get("overall_score", metrics.get("accuracy", metrics.get("score", "N/A")))
        print(f"  {name:25s} : {score}")
else:
    for path in sorted(out.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        score = data.get("overall_score", data.get("accuracy", data.get("score", "N/A")))
        print(f"  {path.stem:25s} : {score}")
PY

echo ""
echo "All evaluations written to ${OUT_DIR}"
