# ruff: noqa: F401, F403, F405
"""Local-first persistent sessions and named workstreams for Aurelius.

The session manager is deliberately small: it persists JSON-safe session
snapshots to disk, keeps an in-memory index for the current process, and
provides explicit helpers for workstreams, queued work items, threads,
approvals, checkpoints, background jobs, messages, and tool-call audit
records. It does not launch network services or background daemons.

This module is a thin re-export facade: the implementation lives in the
cohesive :mod:`src.agent.session` sub-modules (models, util, persistence,
sessions, journal, workstreams, records, work_items) and every historical
import path under ``agent.session_manager`` / ``src.agent.session_manager``
keeps working unchanged.
"""

from __future__ import annotations

from src.model.interface_framework import (
    ApprovalRequest,
    BackgroundJob,
    Checkpoint,
    InterfaceFrameworkError,
    MessageEnvelope,
    SkillBundle,
    TaskThread,
    Workstream,
)

from .session_journal import (
    SessionJournal,
    SessionJournalBranch,
    SessionJournalCompaction,
    SessionJournalEntry,
)

from src.agent.session.journal import *
from src.agent.session.manager import *
from src.agent.session.models import *
from src.agent.session.persistence import *
from src.agent.session.records import *
from src.agent.session.sessions import *
from src.agent.session.util import *
from src.agent.session.work_items import *
from src.agent.session.workstreams import *

__all__ = [
    "WorkItem",
    "SessionRecord",
    "SessionManager",
]
