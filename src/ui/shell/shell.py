"""The composed Aurelius terminal shell."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from src.model.interface_framework import ApprovalRequest
from src.model.interface_framework import AureliusInterfaceFramework
from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import Checkpoint
from src.model.interface_framework import TaskThread
from src.ui.shell.capabilities import ShellCapabilitiesMixin
from src.ui.shell.commands import ShellCommandsMixin
from src.ui.shell.interactions import ShellInteractionsMixin
from src.ui.shell.models import MessageEnvelope
from src.ui.shell.models import WorkflowRun
from src.ui.shell.models import Workstream
from src.ui.shell.render import ShellRenderMixin
from src.ui.shell.skills import ShellSkillsMixin
from src.ui.shell.snapshot import ShellSnapshotMixin
from src.ui.shell.state import ShellStateMixin
from src.ui.shell.workflow import ShellWorkflowMixin

__all__ = [
    "AureliusShell",
]


class AureliusShell(
    ShellStateMixin,
    ShellInteractionsMixin,
    ShellSkillsMixin,
    ShellWorkflowMixin,
    ShellSnapshotMixin,
    ShellCommandsMixin,
    ShellRenderMixin,
    ShellCapabilitiesMixin,
):
    """Terminal-first Aurelius shell built on the interface framework."""

    def __init__(
        self,
        framework: AureliusInterfaceFramework | None = None,
        *,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
        session_id: str | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        self.framework = framework or AureliusInterfaceFramework.from_repo_root(
            root_dir=root_dir,
            variant_id=variant_id,
        )
        self.session_id = session_id or f"session-{uuid.uuid4()}"
        self.workspace = str(workspace or self.framework.paths.repo_root)
        self.current_mode = self._default_mode_name()
        self.active_thread_id: str | None = None
        self.active_workstream_id: str | None = None
        self._threads: dict[str, TaskThread] = {}
        self._workstreams: dict[str, Workstream] = {}
        self._jobs: dict[str, BackgroundJob] = {}
        self._approvals: dict[str, ApprovalRequest] = {}
        self._checkpoints: dict[str, Checkpoint] = {}
        self._messages: list[MessageEnvelope] = []
        self._workflow_runs: dict[str, WorkflowRun] = {}
        self._tool_calls: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def from_repo_root(
        cls,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
        *,
        session_id: str | None = None,
        workspace: str | Path | None = None,
    ) -> AureliusShell:
        return cls(
            root_dir=root_dir,
            variant_id=variant_id,
            session_id=session_id,
            workspace=workspace,
        )
