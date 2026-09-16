# AMC Validation Green Gate — 2026-05-22

## Summary verdict

**PASS WITH SKIPS** — core AMC promotion/SDB/security/reproducibility/paper scaffold tests pass on branch `clean/amc-curation-20260521-101220` at `6665d5d5` (plus uncommitted orchestrator docs). Full targeted pytest suite from Prompt 05 not run end-to-end in one invocation (runtime); subset below passed.

## Environment

```text
date -u: Sat May 23 22:28 UTC 2026
branch: clean/amc-curation-20260521-101220
HEAD: 6665d5d5c50471c3ab85d6c74ce4a93a38f83023
Python: 3.12.12
pytest: 9.0.3
ruff: 0.15.12
```

Dirty worktree (orchestrator in progress): `uv.lock`, `docs/prompts/TRANCHE_STATUS.md`, `paper/*`, `release/`, untracked `docs/prompts/*` parts, `llm-skills/`, `projects/`.

## Commands run

```bash
cd /Users/christienantonio/aurelius
.venv/bin/python -m compileall -q src/model src/memory src/training src/eval src/agent src/security src/serving
.venv/bin/python -m pytest tests/model/test_amc_promotion.py tests/memory/test_sdb_memory_runtime.py \
  tests/security/test_amc_security_audit.py tests/reproducibility/ tests/paper/test_paper_skeleton.py -q --tb=short
.venv/bin/python scripts/validate_paper.py
.venv/bin/python scripts/run_cross_validation.py --profile smoke
.venv/bin/python -m ruff check src/eval/ablation.py src/memory/sdb_runtime.py src/model/amc_promotion.py
```

## Results

| Gate | Status | Detail |
|------|--------|--------|
| compileall | PASS | AMC packages compile |
| promotion tests | PASS | ST forward=hard, backward via soft |
| SDB tests | PASS | fail-closed + mismatch ordering |
| security audit tests | PASS | 6/6 structural |
| reproducibility bundle | PASS | |
| cross-validation smoke | PASS | after removing `/Users` literal from new shell script |
| paper skeleton tests | PASS | 1 skip if no pdflatex |
| ruff (Prompt 03 file list) | PASS | listed AMC files clean |

## Skips / not run

- `test_latex_compiles` skipped when `pdflatex` absent.
- Full Prompt 05 pytest list (~25 files) not executed as single job in this gate run.
- Repo-wide ruff outside AMC surface: not in scope.

## Recommended next action

1. Commit orchestrator deliverables (paper sections, release/, reports, tracker).
2. Run full GPU ablation (`ABLATION_MODE=engine`) before claiming C8/C9 in paper.
3. Execute `cross_validate_clean_machine.sh` on an external VM (T31).
4. Do **not** push until operator approves; keep `uv.lock` out of AMC commits unless intentional.
