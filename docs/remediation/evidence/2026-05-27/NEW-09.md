# Evidence Manifest — NEW-09: CODE_REVIEW_RESCAN.md is stale

## Row Metadata
- **Row ID**: NEW-09
- **Status**: FIXED (superseded by this register)

## Problem
Repo carried 3 parallel review documents from 2026-05-18:
- `CODE_REVIEW.md`
- `CODE_REVIEW_RESCAN.md`
- `COMBINED_CODE_REVIEW.md`

They had their own ID taxonomy, ~145 findings, and were 9+ days stale.

## Resolution
The v4-normalized CSV (`/Desktop/AI:ML Research/aurelius-audit-findings.v4-normalized.csv`) is now the canonical register. Findings from the review docs have been triaged:
- ~60% are already FIXED (verified in current code)
- Remaining OPEN items imported into the register (NEW-01 through NEW-09)

The legacy docs remain as historical artifacts but should be considered read-only. No in-repo review document should be treated as an executor register going forward.

## Files Referenced
- `CODE_REVIEW.md` (read-only, historical)
- `CODE_REVIEW_RESCAN.md` (read-only, historical)
- `COMBINED_CODE_REVIEW.md` (read-only, historical)
- `docs/remediation/evidence/2026-05-27/PHASE-1-7-STATUS.md` (canonical status)
