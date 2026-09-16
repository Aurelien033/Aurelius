"""On-disk persistence and lookup helpers for session state."""

from __future__ import annotations

from pathlib import Path
import json

from src.agent.session.models import SessionRecord
from src.agent.session.util import _resolve_safe_path
from src.agent.session.util import _safe_filename
from src.agent.session.util import _utc_now
from src.agent.session_journal import SessionJournal
from src.model.interface_framework import InterfaceFrameworkError
from src.model.interface_framework import Workstream

# File locking for multi-process safety
try:
    from filelock import FileLock

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_FILELOCK = False

__all__ = [
    "SessionPersistenceMixin",
    "_HAS_FILELOCK",
]


class SessionPersistenceMixin:
    """Disk paths, JSON persistence and session lookup helpers."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _journal_path(self, session_id: str) -> Path:
        safe_id = _safe_filename(session_id, "session_id")
        return _resolve_safe_path(self._journal_dir, self._journal_dir / f"{safe_id}.json")

    def _persist_journal(self, journal: SessionJournal) -> None:
        if not self.persist:
            return
        path = self._journal_path(journal.session_id)
        try:
            if _HAS_FILELOCK:
                lock_path = path.with_suffix(".json.lock")
                with FileLock(str(lock_path), timeout=5):
                    path.write_text(
                        json.dumps(journal.snapshot(), indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
            else:
                path.write_text(
                    json.dumps(journal.snapshot(), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise InterfaceFrameworkError(f"cannot persist journal: {path}") from exc

    def _load_journal(self, path: Path) -> SessionJournal:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InterfaceFrameworkError(f"journal payload is not an object: {path}")
        return SessionJournal.from_dict(payload)

    def _touch_session(self, session_id: str) -> None:
        session = self._require_session(session_id)
        session.updated_at = _utc_now()
        self._persist(session)

    def _session_path(self, session_id: str) -> Path:
        safe_id = _safe_filename(session_id, "session_id")
        return _resolve_safe_path(self.state_dir, self.state_dir / f"{safe_id}.json")

    def _persist(self, session: SessionRecord) -> None:
        if not self.persist:
            return
        path = self._session_path(session.session_id)
        payload = self.snapshot(session)
        try:
            if _HAS_FILELOCK:
                lock_path = path.with_suffix(".json.lock")
                with FileLock(str(lock_path), timeout=5):
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            else:
                path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise InterfaceFrameworkError(f"cannot persist session: {path}") from exc

    def _load_session(self, path: Path) -> SessionRecord:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InterfaceFrameworkError(f"session payload is not an object: {path}")
        return SessionRecord.from_dict(payload)

    def _require_session(self, session_id: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        return session

    def _resolve_workstream(
        self,
        session: SessionRecord,
        workstream_id_or_name: str | None,
    ) -> Workstream | None:
        if workstream_id_or_name is None:
            if session.active_workstream_id is None:
                return None
            return session.workstreams.get(session.active_workstream_id)
        if workstream_id_or_name in session.workstreams:
            return session.workstreams[workstream_id_or_name]
        for workstream in session.workstreams.values():
            if workstream.name == workstream_id_or_name:
                return workstream
        return None
