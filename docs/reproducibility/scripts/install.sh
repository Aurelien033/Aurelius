#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "╔════════════════════════════════════════════╗"
echo "║  Aurelius AMC — Reproducibility Install   ║"
echo "╚════════════════════════════════════════════╝"
echo " Repository: $REPO_ROOT"
echo ""

if command -v uv >/dev/null 2>&1; then
  echo "Using uv to sync environment..."
  uv sync --extra train --extra dev
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  echo "uv not found; using python -m venv + pip"
  if [[ ! -d .venv ]]; then
    python3.12 -m venv .venv || python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install -e ".[train,dev]"
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

echo ""
echo "Validating Forge config..."
"$PYTHON" scripts/validate_amc_forge_config.py

echo ""
echo "✅ Install complete. Activate with: source .venv/bin/activate"
