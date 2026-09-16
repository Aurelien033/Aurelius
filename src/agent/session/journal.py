"""Session journal read/write surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from src.agent.session.util import _json_safe
from src.agent.session_journal import SessionJournal
from src.agent.session_journal import SessionJournalBranch
from src.agent.session_journal import SessionJournalCompaction
from src.agent.session_journal import SessionJournalEntry
from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "SessionJournalMixin",
]


class SessionJournalMixin:
    """Journal entries, branches, compactions and summaries."""

    # ------------------------------------------------------------------
    # journal
    # ------------------------------------------------------------------
    def get_journal(self, session_id: str, *, create: bool = True) -> SessionJournal | None:
        session = self._require_session(session_id)
        journal = self._journals.get(session_id)
        if journal is not None:
            return journal
        path = self._journal_path(session_id)
        if path.exists():
            journal = self._load_journal(path)
            self._journals[session_id] = journal
            return journal
        if not create:
            return None
        journal = SessionJournal.create(
            session_id,
            created_at=session.created_at,
            metadata={"workspace": session.workspace},
        )
        self._journals[session_id] = journal
        self._persist_journal(journal)
        return journal

    def append_journal_entry(
        self,
        session_id: str,
        *,
        kind: str,
        summary: str,
        branch_id: str = "main",
        thread_id: str | None = None,
        workstream_id: str | None = None,
        parent_entry_id: str | None = None,
        severity: str = "info",
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: tuple[str, ...] | list[str] | tuple[Any, ...] | list[Any] = (),
    ) -> SessionJournalEntry:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        entry = journal.append(
            kind=kind,
            summary=summary,
            branch_id=branch_id,
            thread_id=thread_id,
            workstream_id=workstream_id,
            parent_entry_id=parent_entry_id,
            severity=severity,
            payload=payload,
            metadata=metadata,
            tags=tags,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return entry

    def branch_journal(
        self,
        session_id: str,
        name: str,
        *,
        from_entry_id: str | None = None,
        source_branch_id: str = "main",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        branch = journal.branch(
            name,
            from_entry_id=from_entry_id,
            source_branch_id=source_branch_id,
            metadata=metadata,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return {
            "branch": _json_safe(asdict(branch)),
            "anchor_entry": (
                _json_safe(asdict(journal.get_entry(branch.head_entry_id)))
                if branch.head_entry_id is not None
                else None
            ),
        }

    def compact_journal(
        self,
        session_id: str,
        *,
        branch_id: str = "main",
        keep_last_n: int = 4,
        policy: str = "oldest_first",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        compaction = journal.compact(
            branch_id=branch_id,
            keep_last_n=keep_last_n,
            policy=policy,
            metadata=metadata,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return {
            "compaction": _json_safe(asdict(compaction)),
            "entry": (
                _json_safe(asdict(journal.get_entry(compaction.summary_entry_id)))
                if compaction.summary_entry_id is not None
                else None
            ),
        }

    def list_journal_entries(
        self,
        session_id: str,
        *,
        branch_id: str | None = None,
    ) -> list[SessionJournalEntry]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        return list(journal.entries_for_branch(branch_id))

    def list_journal_branches(self, session_id: str) -> list[SessionJournalBranch]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        return list(journal.list_branches())

    def get_journal_entry(self, session_id: str, entry_id: str) -> SessionJournalEntry | None:
        journal = self.get_journal(session_id, create=False)
        if journal is None:
            return None
        return journal.get_entry(entry_id)

    def list_journal_compactions(
        self,
        session_id: str,
        *,
        branch_id: str | None = None,
    ) -> list[SessionJournalCompaction]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        compactions = list(journal.compactions.values())
        if branch_id is not None:
            compactions = [item for item in compactions if item.branch_id == branch_id]
        return sorted(compactions, key=lambda item: (item.created_at, item.compaction_id))

    def journal_summary(self, session_id: str) -> dict[str, Any]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        branches = self.list_journal_branches(session_id)
        compactions = self.list_journal_compactions(session_id)
        latest_compaction = compactions[-1] if compactions else None
        return {
            **journal.describe(),
            "branches": [
                {
                    "branch_id": branch.branch_id,
                    "name": branch.name,
                    "head_entry_id": branch.head_entry_id,
                    "base_entry_id": branch.base_entry_id,
                    "entry_count": len(branch.entry_ids),
                    "metadata": _json_safe(branch.metadata),
                }
                for branch in branches
            ],
            "compactions": [
                {
                    "compaction_id": compaction.compaction_id,
                    "branch_id": compaction.branch_id,
                    "policy": compaction.policy,
                    "keep_last_n": compaction.keep_last_n,
                    "dropped_count": len(compaction.dropped_entry_ids),
                    "retained_count": len(compaction.retained_entry_ids),
                    "summary_entry_id": compaction.summary_entry_id,
                    "created_at": compaction.created_at,
                }
                for compaction in compactions
            ],
            "latest_compaction": _json_safe(asdict(latest_compaction))
            if latest_compaction is not None
            else None,
        }
