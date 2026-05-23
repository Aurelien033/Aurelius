#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="${ROOT}/release/dist"
ZIP="${DIST}/amc_arxiv_source.zip"
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
fi

mkdir -p "${DIST}"
STAGE="${DIST}/amc_arxiv_stage"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/paper"
cp -R "${ROOT}/paper/main.tex" "${ROOT}/paper/sections" "${ROOT}/paper/tables" \
  "${ROOT}/paper/figures" "${ROOT}/paper/references.bib" "${ROOT}/paper/CLAIMS_LEDGER.md" \
  "${STAGE}/paper/" 2>/dev/null || true

if [[ "${DRY}" == "1" ]]; then
  echo "Dry-run: would zip ${STAGE}/paper -> ${ZIP}"
  find "${STAGE}" -type f
  exit 0
fi

(cd "${STAGE}" && zip -r "${ZIP}" paper)
echo "Created ${ZIP}"
