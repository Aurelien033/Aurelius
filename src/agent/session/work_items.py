"""Queued work items inside a session workstream."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
from typing import Any
import uuid

from src.agent.session.models import WorkItem
from src.agent.session.models import _WORK_ITEM_STATUSES
from src.agent.session.util import _json_safe
from src.agent.session.util import _require_non_empty
from src.agent.session.util import _utc_now
from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "SessionWorkItemMixin",
]


class SessionWorkItemMixin:
    """Queued work item surface."""

    # ------------------------------------------------------------------
    # work items / background jobs
    # ------------------------------------------------------------------
    def queue_work_item(
        self,
        session_id: str,
        workstream_name_or_id: str,
        *,
        kind: str,
        title: str | None = None,
        payload: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkItem:
        session = self._require_session(session_id)
        workstream = self._resolve_workstream(session, workstream_name_or_id)
        if workstream is None:
            workstream = self.create_workstream(
                session_id,
                workstream_name_or_id,
                workspace=session.workspace,
            )
            session = self._require_session(session_id)
        item = WorkItem(
            item_id=f"item-{uuid.uuid4()}",
            session_id=session.session_id,
            workstream_id=workstream.workstream_id,
            kind=_require_non_empty(kind, "kind"),
            title=_require_non_empty(title or kind, "title"),
            payload=dict(payload or {}),
            status="queued",
            created_at=_utc_now(),
            updated_at=_utc_now(),
            thread_id=thread_id,
            metadata=dict(metadata or {}),
        )
        session.queue.append(item)
        session.workstreams[workstream.workstream_id] = replace(
            workstream,
            queued_items=workstream.queued_items + (dict(asdict(item)),),
            updated_at=_utc_now(),
        )
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="work_item.queued",
            summary=f"Queued work item {item.title}",
            workstream_id=workstream.workstream_id,
            thread_id=thread_id,
            payload={"work_item": _json_safe(asdict(item))},
        )
        return item

    def update_work_item(
        self,
        session_id: str,
        item_id: str,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> WorkItem:
        session = self._require_session(session_id)
        if status not in _WORK_ITEM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_WORK_ITEM_STATUSES)}, got {status!r}"
            )
        updated_item: WorkItem | None = None
        new_queue: list[WorkItem] = []
        for item in session.queue:
            if item.item_id != item_id:
                new_queue.append(item)
                continue
            updated_item = replace(
                item,
                status=_require_non_empty(status, "status"),
                updated_at=_utc_now(),
                result=dict(result or {}) if result is not None else item.result,
                error=error,
            )
            new_queue.append(updated_item)
        if updated_item is None:
            raise InterfaceFrameworkError(f"unknown work item: {item_id!r}")
        session.queue = new_queue
        workstream = session.workstreams.get(updated_item.workstream_id)
        if workstream is not None:
            updated_items = tuple(
                dict(asdict(item))
                for item in session.queue
                if item.workstream_id == workstream.workstream_id
            )
            session.workstreams[workstream.workstream_id] = replace(
                workstream,
                queued_items=updated_items,
                updated_at=_utc_now(),
            )
        self._persist(session)
        self.append_journal_entry(
            session_id,
            kind="work_item.updated",
            summary=f"Updated work item {updated_item.title}",
            workstream_id=updated_item.workstream_id,
            thread_id=updated_item.thread_id,
            payload={"work_item": _json_safe(asdict(updated_item))},
        )
        return updated_item

    def cancel_work_item(self, session_id: str, item_id: str) -> WorkItem:
        return self.update_work_item(session_id, item_id, status="canceled")
