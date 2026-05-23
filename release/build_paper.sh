#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/paper"
if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not installed; skip PDF build" >&2
  exit 1
fi
latexmk -pdf -interaction=nonstopmode main.tex
echo "PDF: ${ROOT}/paper/main.pdf"
