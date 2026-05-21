"""Episodic memory: event-stamped entries with recency scoring.

Now includes optional session-scoped deduplication — store() rejects duplicate
``(session_id, step, role, content)`` keys within a configurable window,
preventing the same observation from being recorded twice in one step.

Zero third-party imports — pure stdlib.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MemoryEntry:
    """A single episodic memory event."""

    role: str
    content: str
    importance: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str | None = None
    step: int | None = None


class EpisodicMemory:
    """Episodic memory store with capacity-bounded, recency/importance retrieval."""

    # ---------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------

    def __init__(
        self,
        max_entries: int = 1000,
        dedup_window: int = 100,
    ) -> None:
        """
        Args:
            max_entries: Hard cap on stored MemoryEntry objects.
            dedup_window: When ``session_id`` + ``step`` are provided to
                :meth:`store`, skip entries whose ``(session_id, step, role,
                content)`` tuple already appears inside the most-recent
                ``dedup_window`` entries.  Use 0 to disable dedup.
        """
        self._max_entries = max_entries
        self._dedup_window = dedup_window
        self._entries: list[MemoryEntry] = []

    # ---------------------------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------------------------

    def store(
        self,
        role: str,
        content: str,
        importance: float = 1.0,
        session_id: str | None = None,
        step: int | None = None,
    ) -> MemoryEntry:
        """Create and store a new MemoryEntry, evicting oldest if over capacity.

        Deduplication (optional):
            When *both* ``session_id`` and ``step`` are provided, the method
            checks whether a record already exists with the same
            ``(session_id, step, role, content)`` key inside the most-recent
            ``self._dedup_window`` entries.  If so the duplicate is silently
            dropped and the *existing* entry is returned instead of creating
            a new one.
        """
        # ── Optional dedup ────────────────────────────────────────────────────
        if session_id is not None and step is not None and self._dedup_window > 0:
            dedup_key = (session_id, step, role, content)
            search_start = max(0, len(self._entries) - self._dedup_window)
            for candidate in self._entries[search_start:]:
                if (
                    candidate.session_id == session_id
                    and candidate.step == step
                    and candidate.role == role
                    and candidate.content == content
                ):
                    return candidate  # idempotent return — no new entry
        # ── Normal store ───────────────────────────────────────────────────────
        entry = MemoryEntry(
            role=role,
            content=content,
            importance=importance,
            session_id=session_id,
            step=step,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            # Evict oldest (lowest index / earliest timestamp)
            self._entries = self._entries[-self._max_entries :]
        return entry

    def forget(self, entry_id: str) -> bool:
        """Remove entry by id. Returns True if found and removed."""
        for i, e in enumerate(self._entries):
            if e.id == entry_id:
                self._entries.pop(i)
                return True
        return False

    def clear(self) -> int:
        """Remove all entries. Returns count removed."""
        count = len(self._entries)
        self._entries.clear()
        return count

    # ---------------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------------

    def retrieve_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the last *n* entries in insertion order."""
        return list(self._entries[-n:]) if self._entries else []

    def retrieve_by_importance(self, threshold: float = 0.5) -> list[MemoryEntry]:
        """Return entries with importance >= threshold, sorted descending."""
        filtered = [e for e in self._entries if e.importance >= threshold]
        return sorted(filtered, key=lambda e: e.importance, reverse=True)

    def search(self, query: str) -> list[MemoryEntry]:
        """Case-insensitive substring search over entry content."""
        lower_q = query.lower()
        return [e for e in self._entries if lower_q in e.content.lower()]

    # ---------------------------------------------------------------------------
    # Dunder
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)
