"""Registers for threads, approvals, checkpoints, messages, tool calls and jobs."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from typing import Any

from src.agent.session.models import _BACKGROUND_JOB_STATUSES
from src.agent.session.util import _json_safe
from src.agent.session.util import _require_non_empty
from src.agent.session.util import _utc_now
from src.model.interface_framework import ApprovalRequest
from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import Checkpoint
from src.model.interface_framework import InterfaceFrameworkError
from src.model.interface_framework import MessageEnvelope
from src.model.interface_framework import TaskThread

__all__ = [
    "SessionRecordsMixin",
]


class SessionRecordsMixin:
    """Per-record registration and lookup surface."""

    # ------------------------------------------------------------------
    # thread / approval / checkpoint / message / tool-call records
    # ------------------------------------------------------------------
    def register_thread(
        self,
        session_id: str,
        thread: TaskThread,
        *,
        workstream_id: str | None = None,
    ) -> TaskThread:
        session = self._require_session(session_id)
        if not isinstance(thread, TaskThread):
            raise InterfaceFrameworkError(f"thread must be TaskThread, got {type(thread).__name__}")
        session.threads[thread.thread_id] = thread
        session.active_thread_id = thread.thread_id
        workstream = self._resolve_workstream(session, thread.workstream_id or workstream_id)
        if workstream is not None:
            thread_ids = tuple(dict.fromkeys(workstream.thread_ids + (thread.thread_id,)))
            session.workstreams[workstream.workstream_id] = replace(
                workstream,
                current_thread_id=thread.thread_id,
                thread_ids=thread_ids,
                updated_at=_utc_now(),
            )
            session.active_workstream_id = workstream.workstream_id
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="thread.registered",
            summary=f"Registered thread {thread.title}",
            thread_id=thread.thread_id,
            workstream_id=thread.workstream_id or workstream_id,
            payload={"thread": _json_safe(asdict(thread))},
        )
        return thread

    def update_thread(self, session_id: str, thread: TaskThread) -> TaskThread:
        return self.register_thread(session_id, thread)

    def get_thread(self, session_id: str, thread_id: str) -> TaskThread | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.threads.get(thread_id)

    def list_threads(self, session_id: str) -> list[TaskThread]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return sorted(session.threads.values(), key=lambda item: (item.updated_at, item.thread_id))

    def register_approval(self, session_id: str, approval: ApprovalRequest) -> ApprovalRequest:
        session = self._require_session(session_id)
        session.approvals[approval.approval_id] = approval
        thread = session.threads.get(approval.thread_id)
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                approvals=tuple(dict.fromkeys(thread.approvals + (approval.approval_id,))),
                updated_at=_utc_now(),
            )
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="approval.registered",
            summary=f"Registered approval {approval.approval_id}",
            thread_id=approval.thread_id,
            payload={"approval": _json_safe(asdict(approval))},
        )
        return approval

    def get_approval(self, session_id: str, approval_id: str) -> ApprovalRequest | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.approvals.get(approval_id)

    def register_checkpoint(self, session_id: str, checkpoint: Checkpoint) -> Checkpoint:
        session = self._require_session(session_id)
        session.checkpoints[checkpoint.checkpoint_id] = checkpoint
        thread = session.threads.get(checkpoint.thread_id)
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                checkpoints=tuple(dict.fromkeys(thread.checkpoints + (checkpoint.checkpoint_id,))),
                memory_summary=checkpoint.memory_summary,
                updated_at=_utc_now(),
            )
        workstream = self._resolve_workstream(session, checkpoint.workstream_id)
        if workstream is not None:
            session.workstreams[workstream.workstream_id] = replace(
                workstream,
                checkpoint_ids=tuple(
                    dict.fromkeys(workstream.checkpoint_ids + (checkpoint.checkpoint_id,))
                ),
                updated_at=_utc_now(),
            )
        session.memory_summary = checkpoint.memory_summary
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="checkpoint.registered",
            summary=f"Registered checkpoint {checkpoint.checkpoint_id}",
            thread_id=checkpoint.thread_id,
            workstream_id=checkpoint.workstream_id,
            payload={"checkpoint": _json_safe(asdict(checkpoint))},
        )
        return checkpoint

    def get_checkpoint(self, session_id: str, checkpoint_id: str) -> Checkpoint | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.checkpoints.get(checkpoint_id)

    def register_message(self, session_id: str, message: MessageEnvelope) -> MessageEnvelope:
        session = self._require_session(session_id)
        session.messages[message.envelope_id] = message
        thread = session.threads.get(message.thread_id) if message.thread_id is not None else None
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                message_history=thread.message_history + (message,),
                updated_at=_utc_now(),
            )
        if message.workstream_id is not None:
            workstream = self._resolve_workstream(session, message.workstream_id)
            if workstream is not None:
                session.workstreams[workstream.workstream_id] = replace(
                    workstream,
                    messages=workstream.messages + (message,),
                    updated_at=_utc_now(),
                )
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="message.registered",
            summary=f"Registered message {message.envelope_id}",
            thread_id=message.thread_id,
            workstream_id=message.workstream_id,
            payload={"message": _json_safe(asdict(message))},
        )
        return message

    def get_message(self, session_id: str, envelope_id: str) -> MessageEnvelope | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.messages.get(envelope_id)

    def list_messages(
        self,
        session_id: str,
        *,
        channel_id: str | None = None,
        thread_id: str | None = None,
        workstream_id: str | None = None,
    ) -> list[MessageEnvelope]:
        session = self.get_session(session_id)
        if session is None:
            return []
        messages = list(session.messages.values())
        if channel_id is not None:
            messages = [message for message in messages if message.channel_id == channel_id]
        if thread_id is not None:
            messages = [message for message in messages if message.thread_id == thread_id]
        if workstream_id is not None:
            messages = [message for message in messages if message.workstream_id == workstream_id]
        return sorted(messages, key=lambda item: (item.created_at, item.envelope_id))

    def register_tool_call(
        self, session_id: str, thread_id: str, entry: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        session.tool_calls.setdefault(thread_id, []).append(dict(entry))
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="tool_call.recorded",
            summary=f"Recorded tool call {entry.get('tool_name', 'tool')}",
            thread_id=thread_id,
            payload={"tool_call": _json_safe(entry)},
        )
        return entry

    def list_tool_calls(
        self, session_id: str, thread_id: str | None = None
    ) -> list[dict[str, Any]]:
        session = self.get_session(session_id)
        if session is None:
            return []
        if thread_id is None:
            items: list[dict[str, Any]] = []
            for entries in session.tool_calls.values():
                items.extend(_json_safe(entries))
            return items
        return _json_safe(session.tool_calls.get(thread_id, []))

    def register_background_job(self, session_id: str, job: BackgroundJob) -> BackgroundJob:
        session = self._require_session(session_id)
        session.jobs[job.job_id] = job
        thread = session.threads.get(job.thread_id)
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                active_job_ids=tuple(dict.fromkeys(thread.active_job_ids + (job.job_id,))),
                updated_at=_utc_now(),
            )
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="background_job.registered",
            summary=f"Registered background job {job.job_id}",
            thread_id=job.thread_id,
            payload={"job": _json_safe(asdict(job))},
        )
        return job

    def get_background_job(self, session_id: str, job_id: str) -> BackgroundJob | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.jobs.get(job_id)

    def list_background_jobs(
        self, session_id: str, workstream_id: str | None = None
    ) -> list[BackgroundJob]:
        session = self.get_session(session_id)
        if session is None:
            return []
        jobs = list(session.jobs.values())
        if workstream_id is not None:
            jobs = [job for job in jobs if job.metadata.get("workstream_id") == workstream_id]
        return sorted(jobs, key=lambda item: (item.updated_at, item.job_id))

    def update_background_job(
        self,
        session_id: str,
        job_id: str,
        *,
        status: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> BackgroundJob:
        session = self._require_session(session_id)
        job = session.jobs.get(job_id)
        if job is None:
            raise InterfaceFrameworkError(f"unknown background job: {job_id!r}")
        if status is not None and status not in _BACKGROUND_JOB_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_BACKGROUND_JOB_STATUSES)}, got {status!r}"
            )
        updated = replace(
            job,
            status=_require_non_empty(status, "status") if status is not None else job.status,
            updated_at=_utc_now(),
            result=result if result is not None else job.result,
            metadata={**job.metadata, **({"error": error} if error is not None else {})},
        )
        session.jobs[job_id] = updated
        if updated.status in {"completed", "failed", "canceled"}:
            thread = session.threads.get(updated.thread_id)
            if thread is not None:
                session.threads[thread.thread_id] = replace(
                    thread,
                    active_job_ids=tuple(j for j in thread.active_job_ids if j != job_id),
                    updated_at=_utc_now(),
                )
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="background_job.updated",
            summary=f"Updated background job {updated.job_id} to {updated.status}",
            thread_id=updated.thread_id,
            payload={"job": _json_safe(asdict(updated))},
        )
        return updated

    def cancel_background_job(self, session_id: str, job_id: str) -> BackgroundJob:
        return self.update_background_job(session_id, job_id, status="canceled")
