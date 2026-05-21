# Aurelius P0 Reclassification and Remediation Tracker

Source report: `/Users/christienantonio/Desktop/Aurelius/COMBINED_CODE_REVIEW.md`
Validation date: 2026-05-18

## Reclassified findings

| ID | New classification | Rationale | Remediation status |
|---|---|---|---|
| C-04 | FALSE_POSITIVE | The current PPO code forms negative losses (`-ratio * advantages`) and minimizes `torch.max(loss_unclipped, loss_clipped)`, which is equivalent to minimizing `-min(ratio*A, clipped_ratio*A)`. | No functional fix required; optional readability rewrite only. |
| C-09 | RECLASSIFY | `search_code()` builds a command string, but execution currently goes through `shlex.split()` with `shell=False`, so this is not classic shell injection. It remains command/path-safety debt because argv is not structured and paths are not confined. | Fix as command argv/path-containment hardening. |
| C-13 | ALREADY_FIXED | Current `aurelius_cli/main.py` routes explicit `aurelius chat --model-path ...` to `_run_chat()`. The old inverted routing report no longer matches current code. | No P0 fix required. |
| C-15 | RECLASSIFY | Scheduler self-calls `/api/command`, but the current command endpoint sends text to the provider router rather than executing an OS shell command. It remains unsafe task-dispatch design: arbitrary task text, self-HTTP dispatch, and missing auth propagation. | Fix as scheduler command-type allowlist and direct authenticated dispatch hardening. |
| C-22 | RECLASSIFY | Tests normalize command-string scheduling but do not prove OS shell execution. They still lack negative tests for rejected scheduler payloads. | Add rejection tests after scheduler validation is implemented. |

## Confirmed P0/P1 blockers being fixed

- C-01: MoE residual branch overwrites residual stream.
- C-02: Top-p sampling is inverted in `generate()` and `generate_stream()`.
- C-03: `TrainConfig.model_vocab_size` default does not match `AureliusConfig().vocab_size`.
- C-05/C-06/C-07/C-08/C-23: gateway auth, workspace, license, and timing hardening.
- C-12: registry fallback misses category and serialization helpers.
- C-14: Alembic downgrade ordering/index/data-loss risk.
- C-16/C-17/C-18/C-19/C-20/C-21/C-24: deployment hardening.
- C-10/C-11 plus reclassified C-09/C-15/C-22: CLI and scheduler safety hardening.

## Validation expectations

Each remediation tranche should add regression/static tests first where feasible, then run targeted tests plus:

- `ruff check`
- `ruff format --check`
- `python3 -m compileall`
- relevant `pytest` and Node/Vitest commands for touched surfaces
