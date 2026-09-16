"""Snapshot/restore persistence for the shell session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING
import copy

from src.model.interface_framework import ApprovalRequest
from src.model.interface_framework import AureliusInterfaceFramework
from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import Checkpoint
from src.model.interface_framework import MessageEnvelope as FrameworkMessageEnvelope
from src.model.interface_framework import SkillBundle
from src.model.interface_framework import TaskThread
from src.ui.shell.models import AureliusShellError
from src.ui.shell.models import MessageEnvelope
from src.ui.shell.models import WorkflowRun
from src.ui.shell.models import Workstream

if TYPE_CHECKING:
    from src.ui.shell.shell import AureliusShell

__all__ = [
    "ShellSnapshotMixin",
    "_thread_from_dict",
]


class ShellSnapshotMixin:
    """JSON snapshot and restore of the shell session state."""

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe shell state snapshot."""
        return {
            **self._session_header(),
            "framework_signature": {
                "title": self.framework.contract["metadata"]["title"],
                "schema_version": self.framework.contract["metadata"]["schema_version"],
                "variant_id": self.framework.model_context.get("variant_id"),
            },
            "threads": {thread_id: asdict(thread) for thread_id, thread in self._threads.items()},
            "workstreams": {
                workstream_id: asdict(workstream)
                for workstream_id, workstream in self._workstreams.items()
            },
            "jobs": {job_id: asdict(job) for job_id, job in self._jobs.items()},
            "approvals": {
                approval_id: asdict(approval) for approval_id, approval in self._approvals.items()
            },
            "checkpoints": {
                checkpoint_id: asdict(checkpoint)
                for checkpoint_id, checkpoint in self._checkpoints.items()
            },
            "messages": [asdict(message) for message in self._messages],
            "workflow_runs": {run_id: asdict(run) for run_id, run in self._workflow_runs.items()},
            "tool_calls": copy.deepcopy(self._tool_calls),
        }

    @classmethod
    def restore_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
        framework: AureliusInterfaceFramework | None = None,
    ) -> AureliusShell:
        if not isinstance(snapshot, Mapping):
            raise AureliusShellError("snapshot must be a mapping")
        shell = cls(
            framework=framework,
            root_dir=root_dir,
            variant_id=variant_id,
            session_id=snapshot.get("session_id"),
            workspace=snapshot.get("workspace"),
        )
        snapshot_mode = snapshot.get("current_mode", shell.current_mode)
        if isinstance(snapshot_mode, str):
            shell.set_mode(snapshot_mode)
        else:
            shell.current_mode = shell._default_mode_name()
        shell.active_thread_id = snapshot.get("active_thread_id")
        shell.active_workstream_id = snapshot.get("active_workstream_id")

        signature = snapshot.get("framework_signature", {})
        expected_signature = {
            "title": shell.framework.contract["metadata"]["title"],
            "schema_version": shell.framework.contract["metadata"]["schema_version"],
            "variant_id": shell.framework.model_context.get("variant_id"),
        }
        if dict(signature) and dict(signature) != expected_signature:
            raise AureliusShellError(
                "snapshot framework signature does not match the requested framework"
            )

        for thread_id, payload in dict(snapshot.get("threads", {})).items():
            thread = _thread_from_dict(payload)
            shell._threads[thread_id] = thread
        for workstream_id, payload in dict(snapshot.get("workstreams", {})).items():
            shell._workstreams[workstream_id] = Workstream.from_dict(payload)
        for job_id, payload in dict(snapshot.get("jobs", {})).items():
            shell._jobs[job_id] = BackgroundJob(**payload)
        for approval_id, payload in dict(snapshot.get("approvals", {})).items():
            shell._approvals[approval_id] = ApprovalRequest(**payload)
        for checkpoint_id, payload in dict(snapshot.get("checkpoints", {})).items():
            shell._checkpoints[checkpoint_id] = Checkpoint(
                checkpoint_id=payload["checkpoint_id"],
                thread_id=payload["thread_id"],
                created_at=payload["created_at"],
                lineage=tuple(payload.get("lineage", ())),
                thread_snapshot=dict(payload["thread_snapshot"]),
                contract_metadata=dict(payload["contract_metadata"]),
                model_context=dict(payload["model_context"]),
                memory_summary=payload["memory_summary"],
                last_model_response=payload.get("last_model_response"),
                last_tool_result=payload.get("last_tool_result"),
                session_id=payload.get("session_id"),
                workstream_id=payload.get("workstream_id"),
                workstream_name=payload.get("workstream_name"),
                pending_approval_ids=tuple(payload.get("pending_approval_ids", ())),
                active_job_ids=tuple(payload.get("active_job_ids", ())),
                tool_observations=tuple(
                    dict(item) for item in payload.get("tool_observations", ())
                ),
            )
        shell._messages = [
            MessageEnvelope.from_dict(payload) for payload in snapshot.get("messages", [])
        ]
        shell._workflow_runs = {
            run_id: WorkflowRun.from_dict(payload)
            for run_id, payload in dict(snapshot.get("workflow_runs", {})).items()
        }
        shell._tool_calls = {
            thread_id: list(entries)
            for thread_id, entries in dict(snapshot.get("tool_calls", {})).items()
        }
        return shell


def _thread_from_dict(payload: Mapping[str, Any]) -> TaskThread:
    try:
        skills = tuple(
            SkillBundle(
                skill_id=item["skill_id"],
                name=item["name"],
                description=item.get("description", ""),
                scope=item.get("scope", "thread"),
                instructions=item.get("instructions", ""),
                scripts=tuple(item.get("scripts", ())),
                resources=tuple(item.get("resources", ())),
                entrypoints=tuple(item.get("entrypoints", ())),
                version=item.get("version"),
                provenance=item.get("provenance"),
                source_path=item.get("source_path"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("skills", ())
        )
        message_history = tuple(
            FrameworkMessageEnvelope(
                envelope_id=item["envelope_id"],
                channel_id=item["channel_id"],
                thread_id=item.get("thread_id"),
                sender=item["sender"],
                recipient=item.get("recipient"),
                kind=item["kind"],
                payload=dict(item.get("payload", {})),
                created_at=item["created_at"],
                session_id=item.get("session_id"),
                workstream_id=item.get("workstream_id"),
                workspace=item.get("workspace"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("message_history", ())
        )
        return TaskThread(
            thread_id=payload["thread_id"],
            title=payload["title"],
            mode=payload["mode"],
            status=payload["status"],
            host=payload["host"],
            session_id=payload.get("session_id"),
            workstream_id=payload.get("workstream_id"),
            workstream_name=payload.get("workstream_name"),
            workspace=payload.get("workspace"),
            workspace_roots=tuple(payload.get("workspace_roots", ())),
            channel=payload.get("channel"),
            repo_instructions=payload.get("repo_instructions"),
            workspace_instructions=payload.get("workspace_instructions"),
            instruction_stack=tuple(payload.get("instruction_stack", ())),
            skills=skills,
            approvals=tuple(payload.get("approvals", ())),
            checkpoints=tuple(payload.get("checkpoints", ())),
            steps=tuple(dict(item) for item in payload.get("steps", ())),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            parent_thread_id=payload.get("parent_thread_id"),
            parent_checkpoint_id=payload.get("parent_checkpoint_id"),
            lineage=tuple(payload.get("lineage", ())),
            task_prompt=payload.get("task_prompt", ""),
            memory_summary=payload.get("memory_summary", ""),
            last_model_response=payload.get("last_model_response"),
            last_tool_result=payload.get("last_tool_result"),
            active_job_ids=tuple(payload.get("active_job_ids", ())),
            message_history=message_history,
            metadata=dict(payload.get("metadata", {})),
        )
    except KeyError as exc:
        raise AureliusShellError(f"thread snapshot missing required field: {exc.args[0]}") from exc
