#!/usr/bin/env bash
# reconcile_security_repo.sh
#
# Squash-merges the companion aurelius-security-remediation repo into the
# current branch as a single commit, then archives the companion directory.
#
# DO NOT RUN without explicit user approval — this rewrites branch history.
#
# Usage:
#   bash scripts/reconcile_security_repo.sh
#
# Prerequisites:
#   - ../aurelius-security-remediation/ exists and has a clean working tree
#   - Current branch is NOT main/master (the script refuses if it is)
#   - git is available

set -euo pipefail

COMPANION_DIR="../aurelius-security-remediation"
ARCHIVE_DIR="archive/aurelius-security-remediation"
SQUASH_MSG="chore(security): merge ring1 security evidence (May 2026)"
DIVERGENCE_SHA="46ee2f13"

# ── Safety checks ─────────────────────────────────────────────────────────────

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" == "main" || "$CURRENT_BRANCH" == "master" ]]; then
  echo "[error] Refusing to run on protected branch '$CURRENT_BRANCH'."
  echo "        Check out a feature branch first."
  exit 1
fi

if [[ ! -d "$COMPANION_DIR" ]]; then
  echo "[error] Companion repo not found at $COMPANION_DIR"
  echo "        Clone it alongside this repo first."
  exit 1
fi

# Check companion tree is clean
if ! git -C "$COMPANION_DIR" diff --quiet || ! git -C "$COMPANION_DIR" diff --cached --quiet; then
  echo "[error] Companion repo has uncommitted changes. Commit or stash them first."
  exit 1
fi

# ── List divergent commits ─────────────────────────────────────────────────────

echo "=== Commits in companion repo since divergence at $DIVERGENCE_SHA ==="
git -C "$COMPANION_DIR" log --oneline "$DIVERGENCE_SHA"..HEAD || {
  echo "[warn] Could not list divergent commits (SHA may not exist in companion repo)."
}

# ── Squash merge ───────────────────────────────────────────────────────────────

echo ""
echo "=== Adding companion as a temporary remote ==="
git remote add _security_companion "$(realpath "$COMPANION_DIR")" 2>/dev/null || \
  git remote set-url _security_companion "$(realpath "$COMPANION_DIR")"

git fetch _security_companion

COMPANION_HEAD=$(git -C "$COMPANION_DIR" rev-parse HEAD)
echo "=== Squash-merging $COMPANION_HEAD ==="
git merge --squash --allow-unrelated-histories "$COMPANION_HEAD"
git commit -m "$SQUASH_MSG"

git remote remove _security_companion

# ── Archive companion directory ────────────────────────────────────────────────

echo ""
echo "=== Archiving companion to $ARCHIVE_DIR ==="
mkdir -p "$(dirname "$ARCHIVE_DIR")"
cp -r "$COMPANION_DIR" "$ARCHIVE_DIR"

cat > "$ARCHIVE_DIR/README-redirect.md" <<'EOF'
# Archived: aurelius-security-remediation

This directory is an archive of the companion security-remediation repo that was
squash-merged into the main Aurelius repo on the feature/ring1-tranche5 branch.

The full history is available at the original companion repo location.
The squash commit message: "chore(security): merge ring1 security evidence (May 2026)"
EOF

echo ""
echo "=== Done. Review the squash commit with: git show HEAD ==="
echo "=== Archived companion at: $ARCHIVE_DIR ==="
