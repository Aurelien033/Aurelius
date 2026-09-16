"""Thread, workstream and session state for the Aurelius shell."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any
import uuid

from src.model.interface_framework import ApprovalRequest
from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import Checkpoint
from src.model.interface_framework import ModePolicy
from src.model.interface_framework import SkillBundle
from src.model.interface_framework import TaskThread
from src.model.interface_framework import TaskThreadSpec
from src.ui.shell.models import AureliusShellError
from src.ui.shell.models import MessageEnvelope
from src.ui.shell.models import SkillRecord
from src.ui.shell.models import WorkflowRun
from src.ui.shell.models import Workstream
from src.ui.shell.util import _dedupe_strings
from src.ui.shell.util import _utc_now

__all__ = [
    "ShellStateMixin",
]


class ShellStateMixin:
    """Read/write surface over the shell's thread and workstream state."""

    @property
    def threads(self) -> tuple[TaskThread, ...]:
        return tuple(self._threads.values())

    @property
    def workstreams(self) -> tuple[Workstream, ...]:
        return tuple(self._workstreams.values())

    @property
    def jobs(self) -> tuple[BackgroundJob, ...]:
        return tuple(self._jobs.values())

    @property
    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(self._approvals.values())

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(self._checkpoints.values())

    @property
    def messages(self) -> tuple[MessageEnvelope, ...]:
        return tuple(self._messages)

    @property
    def workflow_runs(self) -> tuple[WorkflowRun, ...]:
        return tuple(self._workflow_runs.values())

    def _adopt_thread(self, thread: TaskThread) -> TaskThread:
        """Record ``thread`` in the session and make it the active thread."""
        self._threads[thread.thread_id] = thread
        self.active_thread_id = thread.thread_id
        return thread

    def _session_header(self) -> dict[str, Any]:
        """Session identity keys shared by ``describe`` and ``snapshot``."""
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "current_mode": self.current_mode,
            "active_thread_id": self.active_thread_id,
            "active_workstream_id": self.active_workstream_id,
        }

    def list_messages(
        self,
        *,
        channel: str | None = None,
        thread_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """List routed shell envelopes with optional channel/thread filters."""
        filtered: list[dict[str, Any]] = []
        for message in self._messages:
            if channel is not None and message.channel != channel:
                continue
            if thread_id is not None and message.thread_id != thread_id:
                continue
            filtered.append(asdict(message))
        return tuple(filtered)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the shell session."""
        framework_summary = self.framework.describe()
        return {
            **self._session_header(),
            "counts": {
                "threads": len(self._threads),
                "workstreams": len(self._workstreams),
                "jobs": len(self._jobs),
                "approvals": len(self._approvals),
                "checkpoints": len(self._checkpoints),
                "messages": len(self._messages),
                "workflow_runs": len(self._workflow_runs),
                "tool_calls": sum(len(entries) for entries in self._tool_calls.values()),
            },
            "framework": framework_summary,
            "backend_surface": self._describe_backends(),
            "surface_catalog": self.surface_catalog(),
            "shell_capabilities": {
                "thread_status": True,
                "workstream_status": True,
                "job_status": True,
                "workflow_execution": True,
                "skill_discovery": True,
            },
        }

    def list_modes(self) -> tuple[str, ...]:
        return tuple(self.framework.mode_catalog.keys())

    def set_mode(self, mode_name: str) -> ModePolicy:
        policy = self.framework.select_mode(mode_name)
        self.current_mode = policy.name
        return policy

    def create_workstream(
        self,
        name: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        if not isinstance(name, str) or not name.strip():
            raise AureliusShellError("name must be a non-empty string")
        workstream_id = self._build_workstream_id(name)
        now = _utc_now()
        workstream = Workstream(
            workstream_id=workstream_id,
            name=name,
            workspace=str(workspace) if workspace is not None else self.workspace,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._workstreams[workstream_id] = workstream
        self.active_workstream_id = workstream_id
        return workstream

    def create_thread(
        self,
        *,
        title: str,
        task_prompt: str,
        mode: str | None = None,
        host: str = "cli",
        workspace: str | Path | None = None,
        workstream_id: str | None = None,
        workstream_name: str | None = None,
        parent_checkpoint_id: str | None = None,
        channel: str | None = None,
        skills: Sequence[str | SkillBundle | SkillRecord] | None = None,
        repo_instructions: str | None = None,
        workspace_instructions: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> TaskThread:
        normalized_mode = mode or self.current_mode
        if not isinstance(normalized_mode, str) or not normalized_mode.strip():
            raise AureliusShellError("mode must be a non-empty string")
        skill_bundles = self._normalize_skill_inputs(skills or ())
        workspace_value = str(workspace) if workspace is not None else self.workspace
        workspace_roots = (workspace_value,) if workspace_value else ()
        resolved_workstream_name = workstream_name
        if workstream_id is None and resolved_workstream_name is not None:
            created_workstream = self.create_workstream(
                resolved_workstream_name,
                workspace=workspace_value,
            )
            workstream_id = created_workstream.workstream_id
            resolved_workstream_name = created_workstream.name
        if resolved_workstream_name is None and workstream_id is not None:
            existing_workstream = self._workstreams.get(workstream_id)
            if existing_workstream is not None:
                resolved_workstream_name = existing_workstream.name
        spec = TaskThreadSpec(
            title=title,
            mode=normalized_mode,
            task_prompt=task_prompt,
            host=host,
            session_id=self.session_id,
            workstream_id=workstream_id,
            workstream_name=resolved_workstream_name,
            workspace=workspace_value,
            workspace_roots=workspace_roots,
            channel=channel,
            attached_skills=tuple(skill.skill_id for skill in skill_bundles),
            repo_instructions=repo_instructions,
            workspace_instructions=workspace_instructions,
            metadata=dict(metadata or {}),
            thread_id=thread_id,
            parent_thread_id=parent_thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        thread = self.framework.create_thread(spec)
        if skill_bundles:
            thread = self.framework.attach_skills(thread, skill_bundles)
        if workstream_id is None and self.active_workstream_id is not None:
            workstream_id = self.active_workstream_id
        if workstream_id is not None:
            resolved_workstream = self._coerce_workstream(workstream_id)
            thread = replace(
                thread,
                workstream_id=resolved_workstream.workstream_id,
                workstream_name=resolved_workstream.name,
                updated_at=_utc_now(),
            )
        self._adopt_thread(thread)
        if workstream_id is None and self.active_workstream_id is not None:
            workstream_id = self.active_workstream_id
        if workstream_id is not None:
            self._attach_thread_to_workstream(workstream_id, thread.thread_id)
        return thread

    def attach_skills(
        self,
        thread: str | TaskThread,
        skill_inputs: Sequence[str | SkillBundle | SkillRecord],
    ) -> TaskThread:
        target = self._coerce_thread(thread)
        bundles = self._normalize_skill_inputs(skill_inputs)
        updated = self.framework.attach_skills(target, bundles)
        self._threads[updated.thread_id] = updated
        self.active_thread_id = updated.thread_id
        return updated

    def _build_workstream_id(self, name: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip())
        slug = "-".join(part for part in slug.split("-") if part)
        slug = slug[:32] or "workstream"
        return f"workstream-{slug}-{uuid.uuid4().hex[:8]}"

    def _default_mode_name(self) -> str:
        if "chat" in self.framework.mode_catalog:
            return "chat"
        return next(iter(self.framework.mode_catalog))

    def _build_runtime(self):
        from agent.interface_runtime import AureliusInterfaceRuntime

        return AureliusInterfaceRuntime.from_repo_root(
            root_dir=self.framework.paths.repo_root,
            variant_id=self.framework.model_context.get("variant_id"),
        )

    def _coerce_thread(self, thread: str | TaskThread) -> TaskThread:
        if isinstance(thread, TaskThread):
            return thread
        if isinstance(thread, str) and thread:
            try:
                return self._threads[thread]
            except KeyError as exc:
                raise AureliusShellError(f"unknown thread: {thread!r}") from exc
        raise AureliusShellError("thread must be a TaskThread or non-empty thread id")

    def _coerce_workstream(self, workstream: str | Workstream) -> Workstream:
        if isinstance(workstream, Workstream):
            return workstream
        if isinstance(workstream, str) and workstream:
            try:
                return self._workstreams[workstream]
            except KeyError as exc:
                raise AureliusShellError(f"unknown workstream: {workstream!r}") from exc
        raise AureliusShellError("workstream must be a Workstream or non-empty workstream id")

    def _coerce_job(self, job: str | BackgroundJob) -> BackgroundJob:
        if isinstance(job, BackgroundJob):
            job_id = job.job_id
        elif isinstance(job, str) and job:
            job_id = job
        else:
            raise AureliusShellError("job must be a BackgroundJob or non-empty job id")
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise AureliusShellError(f"unknown job: {job_id!r}") from exc

    def _coerce_checkpoint(self, checkpoint: str | Checkpoint) -> Checkpoint:
        if isinstance(checkpoint, Checkpoint):
            checkpoint_id = checkpoint.checkpoint_id
        elif isinstance(checkpoint, str) and checkpoint:
            checkpoint_id = checkpoint
        else:
            raise AureliusShellError("checkpoint must be a Checkpoint or non-empty checkpoint id")
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise AureliusShellError(f"unknown checkpoint: {checkpoint_id!r}") from exc

    def _attach_thread_to_workstream(self, workstream_id: str, thread_id: str) -> None:
        workstream = self._coerce_workstream(workstream_id)
        if thread_id not in self._threads:
            raise AureliusShellError(f"cannot attach unknown thread {thread_id!r}")
        thread_ids = _dedupe_strings(workstream.thread_ids + (thread_id,))
        updated = replace(
            workstream,
            thread_ids=thread_ids,
            updated_at=_utc_now(),
        )
        self._workstreams[updated.workstream_id] = updated
        self.active_workstream_id = updated.workstream_id

    def _inherit_workstream(self, parent_thread_id: str, child_thread_id: str) -> None:
        for workstream in self._workstreams.values():
            if parent_thread_id in workstream.thread_ids:
                self._attach_thread_to_workstream(workstream.workstream_id, child_thread_id)
                return
