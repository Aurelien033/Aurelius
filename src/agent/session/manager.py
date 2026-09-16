"""The composed local-first session manager."""

from __future__ import annotations

from pathlib import Path

from src.agent.session.journal import SessionJournalMixin
from src.agent.session.models import SessionRecord
from src.agent.session.persistence import SessionPersistenceMixin
from src.agent.session.records import SessionRecordsMixin
from src.agent.session.sessions import SessionLifecycleMixin
from src.agent.session.work_items import SessionWorkItemMixin
from src.agent.session.workstreams import SessionWorkstreamMixin
from src.agent.session_journal import SessionJournal

__all__ = [
    "SessionManager",
]


class SessionManager(
    SessionPersistenceMixin,
    SessionLifecycleMixin,
    SessionJournalMixin,
    SessionWorkstreamMixin,
    SessionRecordsMixin,
    SessionWorkItemMixin,
):
    """Manage local-first persistent Aurelius sessions and workstreams."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        root_dir: str | Path | None = None,
        persist: bool = True,
    ) -> None:
        self.persist = bool(persist)
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir is not None else None
        if state_dir is None:
            if self.root_dir is not None:
                resolved_state = self.root_dir / ".aurelius" / "sessions"
            else:
                resolved_state = Path.home() / ".aurelius" / "sessions"
        else:
            resolved_state = Path(state_dir).expanduser().resolve()
        self.state_dir = resolved_state
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._journal_dir = self.state_dir / "journals"
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionRecord] = {}
        self._journals: dict[str, SessionJournal] = {}
