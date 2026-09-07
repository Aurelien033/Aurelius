"""In-memory pub/sub event bus for agent events with typed Event dataclass."""

from __future__ import annotations

import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """Immutable agent event."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    trace_id: str | None = None


class EventBus:
    """Thread-safe in-memory pub/sub event bus.

    Supports synchronous and callback-style subscribers.
    Callbacks are stored via weak references when possible to avoid leaks.
    """

    def __init__(self, max_subscribers_per_event: int = 1_000) -> None:
        self._lock = threading.Lock()
        # event_type -> list of (callback_ref, strong_flag)
        self._subs: dict[str, list[tuple[Any, bool]]] = {}
        self._history_limit: int = 10_000
        self._history: deque[Event] = deque(maxlen=self._history_limit)
        self._max_subscribers: int = max_subscribers_per_event

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        *,
        weak: bool = True,
    ) -> Callable[[Event], None]:
        """Register a callback for an event type.

        Returns the callback for use with unsubscribe.
        """
        with self._lock:
            self._subs.setdefault(event_type, [])
            if len(self._subs[event_type]) >= self._max_subscribers:
                raise RuntimeError(
                    f"Subscriber limit ({self._max_subscribers}) reached for event {event_type!r}"
                )
            if weak:
                ref = weakref.ref(callback)
                self._subs[event_type].append((ref, False))
            else:
                self._subs[event_type].append((callback, True))
        return callback

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """Remove a callback. Returns True if removed."""
        with self._lock:
            subs = self._subs.get(event_type, [])
            for idx, (ref_or_cb, strong) in enumerate(subs):
                cb = ref_or_cb if strong else ref_or_cb()
                if cb is callback:
                    subs.pop(idx)
                    return True
            return False

    def publish(self, event: Event) -> int:
        """Publish an event to all subscribers. Returns number of deliveries."""
        with self._lock:
            self._history.append(event)
            subs = list(self._subs.get(event.event_type, []))

        delivered = 0
        alive: list[tuple[Any, bool]] = []
        for ref_or_cb, strong in subs:
            cb = ref_or_cb if strong else ref_or_cb()
            if cb is None:
                continue
            alive.append((ref_or_cb, strong))
            try:
                cb(event)
                delivered += 1
            except Exception:
                pass  # subscriber errors must not break bus
        # prune dead weak refs, preserving subscribers added during dispatch
        with self._lock:
            current = self._subs.get(event.event_type, [])
            subs_ids = {id(s) for s in subs}
            new_during_publish = [s for s in current if id(s) not in subs_ids]
            pruned = alive + new_during_publish
            if len(current) != len(pruned):
                self._subs[event.event_type] = pruned
        return delivered

    def publish_typed(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
    ) -> Event:
        """Convenience: create and publish an Event."""
        event = Event(
            event_type=event_type,
            payload=dict(payload) if payload else {},
            trace_id=trace_id,
        )
        self.publish(event)
        return event

    def get_history(
        self,
        *,
        event_type: str | None = None,
        trace_id: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Return matching events newest-first."""
        with self._lock:
            events = list(self._history)
        results: list[Event] = []
        for ev in reversed(events):
            if event_type is not None and ev.event_type != event_type:
                continue
            if trace_id is not None and ev.trace_id != trace_id:
                continue
            results.append(ev)
            if limit is not None and len(results) >= limit:
                break
        return results

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def subscriber_count(self, event_type: str) -> int:
        with self._lock:
            return len(self._subs.get(event_type, []))
