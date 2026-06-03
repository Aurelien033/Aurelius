# Ring 1 Tranche 5 — Real Checkpoint + Measured DreamBank Lift

Tranche 5 attaches a real `AMCTransformer` checkpoint to DreamBank sleep for a genuine zero-grad param-hash proof, and replaces the proxy lift estimator with **measured bank-on vs bank-off logit ablation** on held-out traces.

## One-command workflow

```bash
python scripts/ring1_tranche5_workflow.py --config configs/ring1_tranche5.yaml
```

Full verification:

```bash
bash verification/verify_ring1_tranche5.sh
```

## What it does

1. Reuse or collect **200 AMC + 200 baseline traces** (defaults to Tranche 4 corpus)
2. **Load checkpoint** from `checkpoints/aurelius-1.3b/step-0000002/model.safetensors`
3. Run **DreamBank sleep with model attached** (param hash proof)
4. **Measure R1-GB lift** via `run_ablation` logit delta on held-out prompts
5. Run **gate verification** + **repro pack**

## Key files

| File | Role |
|------|------|
| `src/eval/ring1_model_loader.py` | Resolve legacy step dirs, infer config, load weights |
| `src/eval/ring1_measured_lift.py` | Bank rehydration + measured lift |
| `scripts/ring1_tranche5_workflow.py` | End-to-end Tranche 5 pipeline |
| `configs/ring1_tranche5.yaml` | Checkpoint path, lift scale, reuse traces |

## Output

```
docs/reproducibility/ring1_tranche5/
  gate_verification_report.json
  workflow_summary.json
  dreambank_run/
  metrics/
  ring1-repro-*.tar.gz
```

## Scope

- Does **not** modify `src/alignment/dreambank.py`, `src/model/*`, or training code
- Reads existing checkpoint weights only (inference + ablation)
