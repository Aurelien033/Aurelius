# Ring 1 Tranche 3 — Evaluation + DreamBank + Reproducibility

Tranche 3 adds the measurement layer on top of Tranche 2 traces:

- **Evaluation harness** — task success rate, bootstrap CI, ablation table
- **DreamBank runner** — trace → sleep cycles with zero-grad param hash proof
- **Reproducibility pack** — manifest, metrics, DreamBank artifacts, `verify.sh`

## One-command workflow

```bash
# Requires Tranche 2 traces at data/ring1_traces/tranche2/traces.jsonl
python scripts/ring1_tranche3_workflow.py --config configs/ring1_tranche3.yaml
```

Full verification:

```bash
bash verification/verify_ring1_tranche3.sh
```

## Tests

```bash
.venv/bin/python -m pytest \
  tests/test_ring1_trace_format.py \
  tests/test_ring1_amc_integration.py \
  tests/test_ring1_dreambank_runner.py \
  tests/test_ring1_eval_harness.py \
  tests/test_ring1_repro_pack.py -v
```

## Output layout

```
docs/reproducibility/ring1_tranche3/
  workflow_summary.json
  dreambank_run/
    sleep_cycle_log.jsonl
    pre_cycle_model_hash.txt
    post_cycle_model_hash.txt
  metrics/
    raw_results.json
    aggregated_metrics.json
    bootstrap_ci.json
  ring1-repro-<date>-<sha>/
    README.md
    manifest.json
    config/
    traces/test_split/
    dreambank_run/
    metrics/
    verification/verify.sh
  ring1-repro-<date>-<sha>.tar.gz
```

## Scope

- Does **not** modify `src/alignment/dreambank.py`, `src/model/*`, or serving code
- Smoke-scale (50–100 traces); full 200+ trace gate runs are Tranche 4
