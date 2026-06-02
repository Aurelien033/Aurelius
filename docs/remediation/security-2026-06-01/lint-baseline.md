# Lint baseline (2026-06-01)

This document captures the pre-remediation lint baseline.
The CI security workflow (`security.yml`) enforces
zero-tolerance on **new** lint findings, but pre-existing
warnings (added before commit `46ee2f13`) are tracked
here so that the gate does not mask the fix.

## Tooling versions

- ruff: 0.6.x (Python)
- bandit: 1.7.x (Python)
- eslint: 8.x (TypeScript)
- cargo clippy: stable (Rust)

## Baseline (snapshot)

The numbers below are the count of pre-existing warnings
that the CI workflow tolerates via `--exit-zero` on the
specific files listed. New warnings on the same files
(after 2026-06-01) are treated as failures.

| File / glob | ruff | bandit | eslint | clippy | Owner |
|------------|------|--------|--------|--------|-------|
| agent/legacy_* | 12 | 0 | 0 | 0 | @aurelius/platform |
| gateway/legacy_* | 5 | 0 | 0 | 0 | @aurelius/platform |
| middle/src/store/* | 0 | 0 | 8 | 0 | @aurelius/bff |
| server_legacy/ | 0 | 0 | 0 | 0 | (archived) |

## Reset policy

The baseline is reset (counts go to zero) when:
- The file is removed from the repo.
- The owner signs off on a `git log` PR that resolves the
  warnings without functional regression.

A new "Pre-existing warnings" section is appended to this
file at each reset.
