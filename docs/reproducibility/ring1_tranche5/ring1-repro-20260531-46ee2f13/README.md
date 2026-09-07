# Ring 1 Reproducibility Pack — ring1-repro-20260531-46ee2f13

## One-command replay

```bash
./verification/verify.sh
```

## Contents

- `config/` — experiment, inference, and DreamBank configs
- `traces/test_split/` — held-out trace JSONL + memory sidecars
- `dreambank_run/` — sleep cycle logs + pre/post param hashes
- `metrics/` — aggregated metrics and bootstrap CI
- `manifest.json` — config hash, git SHA, artifact paths

Regenerate traces if missing:

```bash
python scripts/ring1_trace_collector.py --config configs/ring1_tranche2.yaml --num_traces 100
python scripts/ring1_tranche3_workflow.py --config configs/ring1_tranche3.yaml
```
