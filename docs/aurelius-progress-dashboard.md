# Aurelius Progress Dashboard

A local, dependency-light dashboard for actively watching Aurelius research progress.

## What it shows

The dashboard is a command center for:

- Overall roadmap progress across all Aurelius workstreams.
- Phase-by-phase roadmap with evidence, blockers, owners, and next actions.
- Track progress for data, training, reasoning/interpretability, safety, agentic deployment, and publication.
- Training/eval metric series from JSONL logs.
- Risk register: representation collapse, contamination, scope creep, and future risks.
- Event feed for run starts, checkpoints, eval gates, safety regressions, and publication milestones.
- API endpoints for appending events, metrics, and milestone updates.

## Files

- `src/monitoring/progress_store.py`
  - Persistent JSON/JSONL state store.
  - Reads/writes roadmap, events, and metrics.
  - Builds the dashboard snapshot.

- `gateway/aurelius_progress_dashboard.py`
  - Local web dashboard server.
  - No external JS/CSS dependencies.
  - Polls `/api/snapshot` every 5 seconds.

- `scripts/aurelius_progress.py`
  - CLI helpers for seeding state and appending events/metrics/milestone updates.

- `tests/monitoring/test_progress_store.py`
  - Unit tests for state, metrics, imports, and snapshot generation.

- `data/aurelius_progress/`
  - Runtime data directory.
  - Contains `state.json`, `events.jsonl`, and `metrics.jsonl`.
  - This directory is ignored by `.gitignore`; keep sensitive experiment details local unless you explicitly choose to version them.

## Start the dashboard

From the Aurelius repo root:

```bash
python -m gateway.aurelius_progress_dashboard --port 7871
```

Open:

```text
http://127.0.0.1:7871
```

If you do not want it to open a browser automatically:

```bash
python -m gateway.aurelius_progress_dashboard --port 7871 --no-browser
```

## Seed or reset roadmap

```bash
python scripts/aurelius_progress.py seed --overwrite
```

This writes the initial roadmap into:

```text
data/aurelius_progress/state.json
```

## CLI examples

Append an event:

```bash
python scripts/aurelius_progress.py event "Started 1.4B smoke training" \
  --kind training \
  --severity info
```

Append a metric:

```bash
python scripts/aurelius_progress.py metric train_loss 2.31 \
  --step 100 \
  --run-id smoke-1
```

Update a milestone:

```bash
python scripts/aurelius_progress.py milestone phase1-baseline-1-4b \
  --status in_progress \
  --evidence "smoke training script starts and logs loss" \
  --owner Aurelius
```

List milestones:

```bash
python scripts/aurelius_progress.py list-milestones
```

Print the full snapshot:

```bash
python scripts/aurelius_progress.py snapshot
```

Import existing metric JSONL:

```bash
python scripts/aurelius_progress.py import-metrics path/to/metrics.jsonl --run-id smoke-1
```

Accepted imported row shapes:

```json
{"name": "train_loss", "value": 2.31, "step": 100, "run_id": "smoke-1"}
```

or:

```json
{"step": 100, "train_loss": 2.31, "val_loss": 2.08}
```

## HTTP API

Append event:

```bash
curl -X POST http://127.0.0.1:7871/api/event \
  -H 'Content-Type: application/json' \
  -d '{"message":"Checkpoint saved","kind":"training","severity":"info","metadata":{"step":1000}}'
```

Append metric:

```bash
curl -X POST http://127.0.0.1:7871/api/metric \
  -H 'Content-Type: application/json' \
  -d '{"name":"train_loss","value":2.31,"step":100,"run_id":"smoke-1"}'
```

Update milestone:

```bash
curl -X POST http://127.0.0.1:7871/api/milestone \
  -H 'Content-Type: application/json' \
  -d '{"milestone_id":"phase1-baseline-1-4b","status":"in_progress","evidence":"smoke run launched"}'
```

Fetch snapshot:

```bash
curl http://127.0.0.1:7871/api/snapshot
```

## Training loop integration pattern

Inside a training loop, log at the same cadence as your existing logger:

```python
from src.monitoring.progress_store import log_training_metric

log_training_metric(
    run_id="aurelius-1.4b-smoke",
    step=global_step,
    metrics={
        "train_loss": float(loss.item()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "tokens_per_sec": tokens_per_sec,
    },
)
```

For eval runs:

```python
from src.monitoring.progress_store import append_metric

append_metric("val_loss", val_loss, step=global_step, run_id="aurelius-1.4b-smoke")
append_metric("amc_score", score, step=global_step, run_id="aurelius-1.4b-smoke")
```

## Roadmap included in the seed state

The initial roadmap covers:

1. Phase 0 — Progress telemetry foundation.
2. Phase 1 — AMC-first baseline and reproducibility.
3. Phase 2 — Reasoning circuits and mechanistic interpretability.
4. Phase 3 — Scale path to 10B under budget.
5. Phase 4 — Alignment, safety, and reward modeling.
6. Phase 5 — Agentic deployment and human feedback loop.
7. Phase 6 — Papers, reproducibility, and public artifact release.

Each phase contains milestones with:

- status
- weight
- owner
- evidence
- notes
- next action
- blockers
- updated timestamp

## Design principles

- Evidence over vibes: every milestone has an evidence field.
- Local-first: no external dashboard service required.
- Append-only metrics/events: easy to import from training logs later.
- Roadmap is editable: update state manually or through the API.
- No benchmark theater: the dashboard separates proven measurements from future goals.
