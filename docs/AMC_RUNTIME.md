# AMC SDB Memory Runtime — Crash Recovery

The SDB memory runtime persists an append-only event log (SQLite WAL). Tier-2/Tier-3
state can be rebuilt by replaying **committed** events via `StateReconstructor`, or
recovered faster using a checkpoint plus WAL tail replay.

## Recovery procedure

1. **Open the WAL** — `SDBPersistentLog(db_path)` after process restart.
2. **Verify integrity** — `log.verify_chain_integrity()` must return `True`.
3. **Load checkpoint (optional)** — `load_amc_checkpoint(path)` or pass
   `checkpoint_path` to `recover_from_crash`.
4. **Replay tail** — `SDBMemoryRuntime.recover_from_crash()` loads the checkpoint
   (if provided) and applies committed events with `seq > checkpoint_seq` from
   `amc_checkpoints`.
5. **Validate** — Compare against a full `StateReconstructor(log).reconstruct_at()`
   or `verify_against_live()` on a known-good reference.

## Saving a checkpoint during normal operation

```python
from src.memory.amc_checkpoint import save_amc_checkpoint
from src.memory.state_reconstruction import StateReconstructor

state = StateReconstructor(log).reconstruct_at()
save_amc_checkpoint(state.tier2, state.tier3, log, path="tier.ckpt")
log.save_checkpoint({"amc_checkpoint_path": "tier.ckpt"})
```

## API

- `SDBMemoryRuntime.recover_from_crash(checkpoint_path=None)` → `ReconstructedState`
- `StateReconstructor.reconstruct_with_base(tier2, tier3, from_seq=...)` — incremental replay

## Invariants

- Only `committed` events mutate Tier-2/Tier-3 during replay.
- `rejected` / `proposed` / `verified` events are audit-only.
- Commit records include sanitized `payload` for deterministic reconstruction.
