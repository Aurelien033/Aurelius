# Ring 1 Tranche 1 — Trace Collection

Tranche 1 provides infrastructure to generate valid 4–12 step Ring 1 traces with full memory event logging. No real MCTS, tool calling, or AMC execution yet.

## Quick Start

From the Aurelius repo root:

```bash
python scripts/ring1_trace_collector.py \
  --config configs/ring1_tranche1.yaml \
  --num_traces 50 \
  --max_steps 12 \
  --output_dir data/ring1_traces/tranche1
```

Validate an existing collection:

```bash
python scripts/ring1_trace_collector.py \
  --config configs/ring1_tranche1.yaml \
  --output_dir data/ring1_traces/tranche1 \
  --validate-only
```

Run tests:

```bash
.venv/bin/python -m pytest tests/test_ring1_trace_format.py -v
```

## Output Layout

```
data/ring1_traces/tranche1/
  traces.jsonl                          # one trace object per line
  sidecars/
    <trace_id>_memory_events.ndjson     # parallel memory event log
```

## Tranche 1 Scope

- Creates: `src/eval/ring1_trace_logger.py`, `src/eval/ring1_dummy_agent.py`, `scripts/ring1_trace_collector.py`, `configs/ring1_tranche1.yaml`, `tests/test_ring1_trace_format.py`
- Does **not** modify AMC model code, DreamBank training, serving, Tier-3, LPD, or PMA

## Tranche 2 Handoff

Tranche 2 should replace `ring1_dummy_agent.py` with real MCTS + AMC glue calls while keeping `ring1_trace_logger.py` format-stable.
