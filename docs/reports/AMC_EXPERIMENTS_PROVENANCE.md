# AMC Experiments Provenance

## Artifact paths

| Table / claim | Artifact | Parser |
|---------------|----------|--------|
| Ablation AMC-Memory | `docs/reproducibility/results/ablation_scores.jsonl` | `src/eval/ablation.py` `summarize_results` |
| Safety audit | `docs/reproducibility/results/security_audit.json` | `src/security/amc_security_audit.py` |

## Rows extracted (AMC-Memory, oracle smoke)

| config | score | stderr | n_samples |
|--------|-------|--------|-----------|
| baseline | 0.583 | 0.007 | 5 |
| tier1_only | 0.723 | 0.007 | 5 |
| tier12 | 0.863 | (see jsonl) | 5 |
| full_amc | 0.953 | (see jsonl) | 5 |

## Missing fields

- RULER, LongBench, GSM8K, MMLU engine scores on trained checkpoints: **TODO**
- Significance stars in paper: bootstrap p-values in jsonl (mostly $> 0.05$ for oracle noise)

## Manual transformations

- Paper table rounds AMC-Memory to three decimals from jsonl means.
- Non--AMC-Memory columns marked `TODO` in LaTeX (not `X.XX`).

## Commands to regenerate

```bash
cd /Users/christienantonio/aurelius
ABLATION_MODE=oracle .venv/bin/python scripts/run_ablation.py
.venv/bin/python scripts/run_security_audit.py
```
