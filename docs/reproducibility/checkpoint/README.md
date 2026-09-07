# Checkpoint slot

Final trained weights are **not** committed to git (multi-GB artifact).

After training, copy or symlink your checkpoint here:

```bash
cp logs/<run>/checkpoint-final.pt docs/reproducibility/checkpoint/amc_forge_1b_final.pt
```

Evaluation and ablation scripts accept any checkpoint path; this location is the
conventional path referenced by the paper reproducibility bundle.
