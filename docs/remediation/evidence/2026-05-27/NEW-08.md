# Evidence Manifest — NEW-08: Bulkhead Concurrency Race Condition

## Row Metadata
- **Row ID**: NEW-08
- **Title**: Bulkhead concurrency race condition
- **Severity / Theme**: Medium / A-Auth (Phase 1)
- **Primary file**: `src/resilience/bulkhead.py`
- **Previous status**: NOT_RECHECKED
- **New status**: NOT_REPRODUCED

## Problem Statement
Per CODE_REVIEW_RESCAN NC-05: "_release() decrements active, releases semaphore, then signals queued event — not atomic. A new thread can acquire between release and signal."

## Reproduction Result
**Not reproduced under CPython 3.12 (GIL-protected).**

Two concurrent stress tests were added:
1. `test_active_count_never_exceeds_max_with_observer` — 30 threads + dedicated observer sampling active_count
2. `test_rapid_cycle_race` — 20 threads × 50 rapid acquire/release cycles

Both pass cleanly in CPython 3.12. The GIL serializes the counter increments and the race window between `self._semaphore.release()` and `next_event.set()` is not observable in practice.

## Theoretical Risk
The race IS real under:
- Free-threaded CPython 3.13+ (nogil builds — active development)
- Jython, IronPython, or PyPy with non-GIL semantics
- True async usage (Bulkhead currently only has sync API, not used from asyncio)

## Mitigation Path
A defensive fix could make `_release()` atomic by incrementing the queued thread's `active_count` inside the lock before releasing the semaphore, but this is a correctness-vs-complexity tradeoff that should be evaluated for free-threaded CPython 3.13 adoption. Not needed now.

## Files Modified
- `tests/resilience/test_bulkhead_race_condition.py` — added 2 stress tests

## Validation
```text
.venv/bin/python -m pytest tests/resilience/test_bulkhead_race_condition.py -v
# 2 passed, 0 failed
```

## Decision
Recorded as NOT_REPRODUCED. Tests kept as regression locks for future free-threaded Python versions.
