"""Plain-text rendering of shell status surfaces."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any
import json

from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import TaskThread
from src.ui.shell.models import Workstream

__all__ = [
    "ShellRenderMixin",
    "_backend_record",
]


class ShellRenderMixin:
    """Rendering surface of the shell."""

    def render_status(self, thread_id: str | None = None) -> str:
        if thread_id is not None:
            return self.render_thread_status(thread_id)

        counts = Counter(thread.status for thread in self._threads.values())
        job_counts = Counter(job.status for job in self._jobs.values())
        lines = [
            "Aurelius Shell",
            f"Session: {self.session_id}",
            f"Workspace: {self.workspace}",
            f"Mode: {self.current_mode}",
            f"Active thread: {self.active_thread_id or 'none'}",
            f"Active workstream: {self.active_workstream_id or 'none'}",
            "Threads:",
        ]
        if self._threads:
            for thread in self._threads.values():
                lines.append(f"  - {self._render_thread_row(thread)}")
        else:
            lines.append("  - none")
        lines.append(
            "Thread counts: "
            + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            if counts
            else "Thread counts: none"
        )
        lines.append("Workstreams:")
        if self._workstreams:
            for workstream in self._workstreams.values():
                lines.append(f"  - {self._render_workstream_row(workstream)}")
        else:
            lines.append("  - none")
        lines.append("Jobs:")
        if self._jobs:
            for job in self._jobs.values():
                lines.append(f"  - {self._render_job_row(job)}")
        else:
            lines.append("  - none")
        if job_counts:
            lines.append(
                "Job counts: "
                + ", ".join(f"{status}={count}" for status, count in sorted(job_counts.items()))
            )
        else:
            lines.append("Job counts: none")
        lines.append(f"Workflow runs: {len(self._workflow_runs)}")
        lines.append(
            f"Tool-call audit entries: {sum(len(entries) for entries in self._tool_calls.values())}"
        )
        lines.append("Backends:")
        backend_names = self._backend_names()
        if backend_names:
            for backend_name in backend_names:
                lines.append(f"  - {backend_name}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def render_thread_status(self, thread_id: str) -> str:
        thread = self._coerce_thread(thread_id)
        skill_labels = (
            ", ".join(f"{skill.name} [{skill.skill_id}]" for skill in thread.skills) or "none"
        )
        approval_ids = (
            ", ".join(
                approval.approval_id
                for approval in self._approvals.values()
                if approval.thread_id == thread.thread_id
            )
            or "none"
        )
        job_ids = (
            ", ".join(
                job.job_id for job in self._jobs.values() if job.thread_id == thread.thread_id
            )
            or "none"
        )
        lines = [
            f"Thread: {thread.thread_id}",
            f"Title: {thread.title}",
            f"Mode: {thread.mode}",
            f"Status: {thread.status}",
            f"Host: {thread.host}",
            f"Session: {thread.session_id or 'none'}",
            f"Workstream id: {thread.workstream_id or 'none'}",
            f"Workstream name: {thread.workstream_name or 'none'}",
            f"Workspace: {thread.workspace or 'none'}",
            f"Channel: {thread.channel or 'none'}",
            f"Parent thread: {thread.parent_thread_id or 'none'}",
            f"Parent checkpoint: {thread.parent_checkpoint_id or 'none'}",
            f"Lineage: {', '.join(thread.lineage) if thread.lineage else 'none'}",
            f"Skills: {skill_labels}",
            f"Approvals: {approval_ids}",
            f"Checkpoints: {', '.join(thread.checkpoints) if thread.checkpoints else 'none'}",
            f"Background jobs: {job_ids}",
            f"Tool calls: {len(self._tool_calls.get(thread.thread_id, ()))}",
            f"Message history: {len(thread.message_history)}",
            f"Memory summary: {thread.memory_summary or 'none'}",
            f"Last model response: {thread.last_model_response or 'none'}",
            f"Active jobs: {', '.join(thread.active_job_ids) if thread.active_job_ids else 'none'}",
            "Instruction stack:",
        ]
        if thread.instruction_stack:
            for layer in thread.instruction_stack:
                lines.append(f"  - {layer}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def render_workstream_status(self, workstream_id: str) -> str:
        workstream = self._coerce_workstream(workstream_id)
        lines = [
            f"Workstream: {workstream.workstream_id}",
            f"Name: {workstream.name}",
            f"Status: {workstream.status}",
            f"Workspace: {workstream.workspace or 'none'}",
            f"Threads: {len(workstream.thread_ids)}",
        ]
        if workstream.thread_ids:
            for thread_id in workstream.thread_ids:
                thread = self._threads.get(thread_id)
                if thread is None:
                    lines.append(f"  - {thread_id} (missing)")
                else:
                    lines.append(f"  - {self._render_thread_row(thread)}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def render_job_status(self, job_id: str) -> str:
        job = self._coerce_job(job_id)
        lines = [
            f"Job: {job.job_id}",
            f"Thread: {job.thread_id}",
            f"Status: {job.status}",
            f"Cancelable: {job.cancelable}",
            f"Description: {job.description}",
            f"Created at: {job.created_at}",
            f"Updated at: {job.updated_at}",
        ]
        if job.result is not None:
            lines.append(f"Result: {job.result!r}")
        if job.metadata:
            lines.append(f"Metadata: {json.dumps(job.metadata, sort_keys=True)}")
        return "\n".join(lines)

    def _render_thread_row(self, thread: TaskThread) -> str:
        skill_count = len(thread.skills)
        checkpoint_count = len(thread.checkpoints)
        return (
            f"{thread.thread_id} | {thread.mode} | {thread.status} | "
            f"{thread.title} | skills={skill_count} checkpoints={checkpoint_count}"
        )

    def _render_workstream_row(self, workstream: Workstream) -> str:
        return (
            f"{workstream.workstream_id} | {workstream.status} | {workstream.name} "
            f"| threads={len(workstream.thread_ids)}"
        )

    def _render_job_row(self, job: BackgroundJob) -> str:
        return f"{job.job_id} | {job.status} | {job.thread_id} | {job.description}"

    def _backend_surface(self):
        import src.backends as backends

        return backends

    def _backend_names(self) -> tuple[str, ...]:
        return self._describe_backends()["names"]

    def _describe_backends(self) -> dict[str, Any]:
        backends = self._backend_surface()
        names = backends.list_backends()
        records = []
        for backend_name in names:
            adapter = backends.get_backend(backend_name)
            records.append(_backend_record(adapter))
        return {
            "count": len(records),
            "names": list(names),
            "backends": records,
        }


def _backend_record(adapter: Any) -> dict[str, Any]:
    """Describe a backend adapter as the shell status and ``backend show`` expect."""
    return {
        "backend_name": adapter.contract.backend_name,
        "adapter_class": type(adapter).__name__,
        "contract": asdict(adapter.contract),
        "runtime_info": adapter.runtime_info(),
    }
