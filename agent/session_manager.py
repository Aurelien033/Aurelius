# DEPRECATED: Use src.agent.session_manager instead.
# Shim module — re-exports all symbols from the canonical implementation
# to avoid duplicate-class identity issues (H-03 agent consolidation).
from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'agent.session_manager' is deprecated. "
    "Use 'src.agent.session_manager' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.agent.session_manager import *  # noqa: E402, F403
from src.agent.session_manager import (  # noqa: E402, F401
    ApprovalRequest,
    BackgroundJob,
    Checkpoint,
    MessageEnvelope,
    SessionJournal,
    SessionJournalBranch,
    SessionJournalCompaction,
    SessionJournalEntry,
    SessionManager,
    SessionRecord,
    SkillBundle,
    TaskThread,
    WorkItem,
    Workstream,
)
