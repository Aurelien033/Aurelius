# Reproducing the AMC Paper

This bundle contains configs, scripts, seeds, and result artifacts needed to
reproduce the Aurelius Memory Core (AMC) ablation study, security audit, and
evaluation harness.

## Prerequisites

- Python 3.12+
- 4× NVIDIA A100 40GB (full training) or CPU for smoke/oracle validation
- ~200 GB disk (full data + checkpoints)
- ~6–10 days wall time for full Forge-1B training (4×A100)

## Quick start (smoke / CI, no GPU)

From the repository root:

```bash
bash docs/reproducibility/scripts/install.sh
bash docs/reproducibility/scripts/evaluate.sh --mode oracle --profile smoke
bash docs/reproducibility/scripts/ablation.sh --mode oracle
python docs/reproducibility/scripts/plot_results.py
python tests/security/run_security_audit.py
```

Expected wall time: **~5 minutes** on a laptop (oracle/mock backends).

## Full reproduction (trained checkpoint)

```bash
bash docs/reproducibility/scripts/install.sh
bash docs/reproducibility/scripts/download_data.sh
bash docs/reproducibility/scripts/train.sh
bash docs/reproducibility/scripts/evaluate.sh logs/<run>/checkpoint-final.pt
bash docs/reproducibility/scripts/ablation.sh logs/<run>/checkpoint-final.pt
python docs/reproducibility/scripts/plot_results.py
```

| Step | Command | Expected wall time |
|------|---------|-------------------|
| Install | `install.sh` | 5–15 min |
| Data | `download_data.sh` | 30–120 min |
| Train | `train.sh` | 6–10 days (4×A100) |
| Evaluate | `evaluate.sh` | ~2 h |
| Ablation | `ablation.sh` | ~4 h |
| Plot | `plot_results.py` | ~1 min |

## Expected outputs

| Artifact | Path |
|----------|------|
| Ablation scores | `docs/reproducibility/results/ablation_scores.jsonl` |
| Ablation figure | `docs/reproducibility/results/ablation_figure.pdf` |
| Security audit | `docs/reproducibility/results/security_audit.json` |
| Eval summary | `logs/eval_*/summary.json` |
| Final checkpoint | `docs/reproducibility/checkpoint/amc_forge_1b_final.pt` (not shipped in git) |

## Configs (four ablation variants)

| File | Description |
|------|-------------|
| `configs/baseline.yaml` | All-attention control (no SSM layers) |
| `configs/tier1_only.yaml` | SSM working memory only |
| `configs/tier12.yaml` | SSM + episodic, no Tier-3 |
| `configs/full_amc.yaml` | Full 3-tier AMC (Forge-1B) |

Configs symlink to `configs/` at the repository root.

## Seeds

All documented seeds live in `seed.txt`. Training and evaluation entrypoints read
`42` by default unless overridden via environment variables.

## Environment lockfile

`environment.yml` is a `pip freeze` snapshot from the reference venv used during
T18–T29 validation. Recreate with:

```bash
bash docs/reproducibility/scripts/install.sh
.venv/bin/pip freeze > docs/reproducibility/environment.yml
```

## Validation (developer)

```bash
pytest tests/reproducibility/test_reproducibility_bundle.py -q
```

## Cross-validation (T31)

Smoke cross-validation (local CI, no GPU):

```bash
bash docs/reproducibility/scripts/cross_validate.sh smoke
# or: python scripts/run_cross_validation.py --profile smoke
```

Writes `CROSS_VALIDATION_REPORT.md` and `results/cross_validation.json`.

For full GPU validation, repeat on a fresh cloud instance per the report’s manual section.
