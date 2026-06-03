"""Tests for NEW-08: Bulkhead concurrent acquisition race condition.

CSV row: NEW-08 — "max_concurrent counter race per RESCAN. Concurrent
bulkhead acquisitions can exceed limit by up to N".

The race: in _release(), decrementing active, releasing the semaphore,
and signaling a queued event are not atomic. A new thread can acquire
the semaphore between release and signal, while the queued thread is
also waking — leading to active_count transiently exceeding max_concurrent.
"""

from __future__ import annotations

import threading
import time

from src.resilience.bulkhead import Bulkhead, BulkheadFullError


class TestBulkheadConcurrencyRace:
    """Stress test: active_count must never exceed max_concurrent.

    Uses a dedicated observer thread sampling active_count in a tight loop
    to catch transient over-counts that the GIL might otherwise hide.
    """

    def test_active_count_never_exceeds_max_with_observer(self) -> None:
        max_n = 3
        bh = Bulkhead(max_concurrent=max_n, max_queue=200, queue_timeout=30.0)
        observed_max = 0
        lock = threading.Lock()
        stop = threading.Event()

        def observer() -> None:
            nonlocal observed_max
            while not stop.is_set():
                with lock:
                    observed_max = max(observed_max, bh.active_count)
                time.sleep(0.0001)

        obs = threading.Thread(target=observer, daemon=True)
        obs.start()

        def work(duration: float) -> None:
            try:
                bh.execute(lambda: time.sleep(duration))
            except BulkheadFullError:
                pass

        threads = [
            threading.Thread(target=work, args=(0.002,))
            for _ in range(30)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30.0)

        stop.set()
        obs.join(timeout=5.0)

        assert observed_max <= max_n, (
            f"Observer saw active_count={observed_max}, exceeding max_concurrent={max_n}"
        )

    def test_rapid_cycle_race(self) -> None:
        """Rapid acquire/release cycles from many threads; check invariant after each."""
        max_n = 2
        bh = Bulkhead(max_concurrent=max_n, max_queue=100, queue_timeout=10.0)
        violations = []
        lock = threading.Lock()

        def cycle() -> None:
            for _ in range(50):
                bh._acquire()
                count = bh.active_count
                if count > max_n:
                    with lock:
                        violations.append(count)
                bh._release()

        threads = [threading.Thread(target=cycle) for _ in range(20)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30.0)

        assert not violations, (
            f"Saw {len(violations)} active_count violations: "
            f"max={max(violations)} (limit={max_n})"
        )
