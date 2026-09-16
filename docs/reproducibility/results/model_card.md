# Aurelius-Forge 1B — Model Card (placeholder)

> Populated after T32 paper tranche. Checkpoint weights are not stored in git.

| Field | Value |
|-------|-------|
| Model | `aurelius-forge-1b-amc` |
| Parameters | ~1.05B (hybrid MLA + AMC-SSM) |
| Config | `configs/amc_forge_1b.yaml` |
| Training data | RedPajama-1T sample (~10M tokens smoke path) |
| License | See repository `LICENSE` |

## Intended use

Research reproduction of the Aurelian Memory Core (AMC) three-tier memory hierarchy.

## Limitations

- Smoke/oracle evaluation paths do not require GPU weights.
- Full engine evaluation requires `docs/reproducibility/checkpoint/amc_forge_1b_final.pt`.
