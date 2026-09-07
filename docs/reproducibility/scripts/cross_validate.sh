#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-smoke}"
echo "Running cross-validation (profile=$PROFILE)..."
"${REPO_ROOT}/.venv/bin/python" scripts/run_cross_validation.py --profile "$PROFILE"
echo "✅ Report: docs/reproducibility/CROSS_VALIDATION_REPORT.md"
