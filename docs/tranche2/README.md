# Ring 1 Tranche 2 — AMC-Integrated Trace Collection

Tranche 2 wires the Observe → Think(MCTS) → Act → Reflect loop to live **AMCTier2Hook** and **SDBMemoryRuntime** surfaces. Memory reads influence MCTS action selection; writes go through surprise gating + SDB propose/verify/commit audit.

## Quick Start

```bash
python scripts/ring1_trace_collector.py \
  --config configs/ring1_tranche2.yaml \
  --num_traces 100 \
  --max_steps 12 \
  --output_dir data/ring1_traces/tranche2
```

Run Tranche 1 dummy mode (backward compatible):

```bash
python scripts/ring1_trace_collector.py \
  --config configs/ring1_tranche1.yaml \
  --num_traces 50 \
  --output_dir data/ring1_traces/tranche1
```

## Tests

```bash
.venv/bin/python -m pytest tests/test_ring1_trace_format.py tests/test_ring1_amc_integration.py -v
```

## New Files (Tranche 2)

| File | Role |
|------|------|
| `src/eval/ring1_amc_glue.py` | Tier-2/SDB/MCTS adapters → trace events |
| `src/eval/ring1_agent.py` | Integrated agent (replaces dummy for Tranche 2) |
| `configs/ring1_tranche2.yaml` | Tranche 2 config (`agent.type: integrated`) |
| `tests/test_ring1_amc_integration.py` | Cross-step memory + SDB audit tests |

## Integration Contract Compliance

- **Observe**: no memory writes
- **Think**: episodic + working + constitutional reads; MCTS consults memory-augmented candidates
- **Act**: action parameters include `memory_informed` flag
- **Reflect**: surprise-gated Tier-2 writes with full SDB audit trail

## Tranche 3 Handoff

Keep `ring1_trace_logger.py` stable. Tranche 3 adds evaluation harness, DreamBank runner, and reproducibility packs on top of Tranche 2 traces.
