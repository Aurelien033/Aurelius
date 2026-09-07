#!/usr/bin/env bash
# T31: structural cross-validation on a fresh workdir (local dry-run).
set -euo pipefail

AMC_REPO_URL="${AMC_REPO_URL:-}"
AMC_REF="${AMC_REF:-HEAD}"
AMC_WORKDIR="${AMC_WORKDIR:-$(mktemp -d -t amc-xval-XXXXXX)}"
AMC_SKIP_TRAINING="${AMC_SKIP_TRAINING:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULT_JSON="${REPO_ROOT}/docs/reproducibility/results/cross_validation_local.json"

mkdir -p "$(dirname "${RESULT_JSON}")"

if [[ -n "${AMC_REPO_URL}" ]]; then
  git clone --depth 1 --branch "${AMC_REF}" "${AMC_REPO_URL}" "${AMC_WORKDIR}/repo"
  WORK="${AMC_WORKDIR}/repo"
else
  rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '.git/objects' \
    "${REPO_ROOT}/" "${AMC_WORKDIR}/checkout/"
  WORK="${AMC_WORKDIR}/checkout"
fi

cd "${WORK}"

if command -v uv >/dev/null 2>&1; then
  uv sync
  PY="${WORK}/.venv/bin/python"
else
  python3 -m venv .venv
  PY="${WORK}/.venv/bin/python"
  "${PY}" -m pip install -e . -q
fi

"${PY}" -m pytest tests/reproducibility/test_reproducibility_bundle.py -q --tb=short

if [[ "${AMC_SKIP_TRAINING}" == "1" ]]; then
  ABLATION_MODE=oracle "${PY}" scripts/run_cross_validation.py --profile smoke
else
  bash docs/reproducibility/scripts/install.sh
  bash docs/reproducibility/scripts/ablation.sh
  "${PY}" scripts/run_cross_validation.py --profile full
fi

"${PY}" - <<'PY'
import json
from pathlib import Path

out = Path("docs/reproducibility/results/cross_validation_local.json")
payload = {
    "status": "EXECUTED_LOCAL_STRUCTURAL",
    "profile": "smoke" if __import__("os").environ.get("AMC_SKIP_TRAINING", "1") == "1" else "full",
    "external_clean_machine": False,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

cp -f "${WORK}/docs/reproducibility/results/cross_validation_local.json" "${RESULT_JSON}" 2>/dev/null || true
echo "Wrote ${RESULT_JSON}"
