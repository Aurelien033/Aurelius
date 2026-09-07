# AMC Branch Curation Readiness — 2026-05-22

## Branch

- **Branch:** `clean/amc-curation-20260521-101220`
- **HEAD:** `6665d5d5` (orchestrator commits `841182b7`, `6dd23fee`, `6665d5d5` ahead of `9ab1eb30`)
- **Push:** not performed (operator approval required)

## Divergence vs origin

```bash
git log --oneline origin/clean/amc-curation-20260521-101220..HEAD
```

Shows T00–T32 implementation commits plus orchestrator fixes (promotion ST, SDB test, ruff).

## Validation status

See `docs/reports/AMC_VALIDATION_GREEN_GATE_20260522.md` — **PASS WITH SKIPS**.

## Dirty / untracked (grouped)

**Intended AMC (stage for next commits):**

- `docs/prompts/TRANCHE_STATUS.md`
- `paper/` (sections, tables, CLAIMS_LEDGER, main.tex)
- `docs/reproducibility/scripts/cross_validate_clean_machine.sh`
- `docs/reproducibility/CROSS_VALIDATION_REPORT.md`
- `docs/reports/AMC_*_20260522.md`
- `release/`
- `scripts/validate_paper.py`, `tests/paper/`, `tests/reproducibility/test_cross_validation_artifacts.py`

**Do not bundle into AMC tranche commits:**

- `uv.lock` (modified)
- `docs/prompts/AMC_FULL_BUILDOUT_PROMPTS_PART*.md`, `docs/prompts/README.md`
- `docs/AMC_COMPLETE_BUILDOUT.md`, `docs/IMPROVEMENT_PLAN.md`, `docs/architecture/`
- `llm-skills/`, `projects/`, `skills/mlops/`, `skills/skill-generative-agents/`
- Unrelated reports (`AURELIUS_*_20260522.md`)

**Note:** commit `841182b7` accidentally added `docs/prompts/AMC_FULL_BUILDOUT_PROMPTS.md` (large); consider reverting that file from history on curate if prompts should stay untracked.

## Commits since T00 (summary)

| Tranche | Commit | Purpose |
|---------|--------|---------|
| T00 | `07dd177f` | Mamba-2 block |
| T01–T03 | `e20cf004`–`6061d688` | Tier-1, promotion (pre-fix), SDB log |
| T04–T10 | `c58f358c`–`571c0399` | Transformer stack |
| T11–T14 | `8db339be`–`e5879f07` | Runtime + WAL |
| T15–T22 | `40addc62`–`49ffbbc6` | Training + eval |
| T23–T27 | `0df15457`–`4046daa9` | Agent memory |
| T28–T30 | `a97609f8`–`01bb24b0` | Ablation, security, repro bundle |
| T31 | `e2427515` | Cross-validation runner |
| T32 | `9ab1eb30` | Paper skeleton (monolithic) |
| Orchestrator | `841182b7` | **ST fix** + doc formula |
| Orchestrator | `6dd23fee` | SDB mismatch test |
| Orchestrator | `6665d5d5` | Ruff hygiene |

## Recommended strategy (do not execute)

**Strategy B:** Finish orchestrator commits on this branch (paper split, release/, reports, tracker), then open PR. Archive or drop unrelated untracked trees in a separate branch (Strategy C) if prompts/skills must not ship with AMC.

### Commands — DO NOT RUN UNTIL APPROVED

```bash
# git push -u origin clean/amc-curation-20260521-101220
# git merge clean/amc-curation-20260521-101220  # on target branch
```
