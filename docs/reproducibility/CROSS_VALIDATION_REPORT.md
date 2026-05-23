# AMC Reproducibility Cross-Validation Report (T31)

## External clean-machine status

**NOT EXECUTED ON EXTERNAL MACHINE** — only local structural smoke on this host.
Use `docs/reproducibility/scripts/cross_validate_clean_machine.sh` on a fresh VM
to record external evidence.

---

- **Date (UTC):** 2026-05-23T22:17:38.154030+00:00
- **Host:** Sapphire.local
- **Python:** 3.12.12
- **Profile:** `smoke`
- **Overall:** PASS

## Summary

| Check | Status | Detail |
|-------|--------|--------|
| forge_config | PASS | validate_amc_forge_config.py ok |
| bundle_structure | PASS | reproducibility bundle tests passed |
| hardcoded_paths | PASS | no /Users or /home literals in bundle scripts |
| ablation_row_count | PASS | 20 config×benchmark rows |
| ablation_monotonicity | PASS | baseline=0.583 → tier1_only=0.723 → tier12=0.863 → full_amc=0.953 |
| ablation_score_drift | PASS | within ±0.08 of reference means |
| security_audit | PASS | 6/6 probes passed |

## Procedure

1. Clone repository on a clean machine (no cached venv/data).
2. `bash docs/reproducibility/scripts/install.sh`
3. `bash docs/reproducibility/scripts/ablation.sh` (oracle smoke) or full GPU path.
4. `python scripts/run_cross_validation.py --profile smoke`
5. Compare `results/ablation_scores.jsonl` to reference ordering:
   `baseline < tier1_only < tier12 < full_amc` on `amc_memory`.

## Deviations

None for smoke profile on this host.

## Full GPU cross-validation (manual)

Repeat on a fresh 4×A100 instance with:

```bash
bash docs/reproducibility/scripts/install.sh
bash docs/reproducibility/scripts/download_data.sh
bash docs/reproducibility/scripts/train.sh
bash docs/reproducibility/scripts/evaluate.sh logs/<run>/checkpoint-final.pt
bash docs/reproducibility/scripts/ablation.sh logs/<run>/checkpoint-final.pt
ABLATION_MODE=engine python scripts/run_cross_validation.py --profile full
```
