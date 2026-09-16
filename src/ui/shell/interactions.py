"""Approval, checkpoint, subagent, background-job and channel routing actions."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import replace
from typing import Any
import copy
import uuid

from src.model.interface_framework import ApprovalRequest
from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import Checkpoint
from src.model.interface_framework import MessageEnvelope as FrameworkMessageEnvelope
from src.model.interface_framework import TaskThread
from src.ui.shell.models import MessageEnvelope
from src.ui.shell.util import _dedupe_strings
from src.ui.shell.util import _utc_now

__all__ = [
    "ShellInteractionsMixin",
]


class ShellInteractionsMixin:
    """Shell actions that hand work to the interface framework."""

    def _update_active_job_ids(
        self,
        thread_id: str,
        transform: Callable[[tuple[str, ...]], tuple[str, ...]],
    ) -> None:
        """Rewrite a thread's active job ids in place, when the thread is known."""
        current_thread = self._threads.get(thread_id)
        if current_thread is None:
            return
        self._threads[thread_id] = replace(
            current_thread,
            active_job_ids=transform(current_thread.active_job_ids),
            updated_at=_utc_now(),
        )

    def request_approval(
        self,
        thread: str | TaskThread,
        *,
        category: str,
        action_summary: str,
        affected_resources: Sequence[str] = (),
        reason: str,
        reversible: bool,
        minimum_scope: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApprovalRequest:
        target = self._coerce_thread(thread)
        approval = self.framework.request_approval(
            target,
            category=category,
            action_summary=action_summary,
            affected_resources=tuple(affected_resources),
            reason=reason,
            reversible=reversible,
            minimum_scope=minimum_scope,
            metadata=metadata,
        )
        self._approvals[approval.approval_id] = approval
        current_thread = self._threads.get(target.thread_id)
        if current_thread is not None:
            updated_approvals = _dedupe_strings(current_thread.approvals + (approval.approval_id,))
            self._threads[target.thread_id] = replace(
                current_thread,
                approvals=updated_approvals,
                updated_at=_utc_now(),
            )
        self.active_thread_id = target.thread_id
        return approval

    def checkpoint_thread(
        self,
        thread: str | TaskThread,
        *,
        memory_summary: str,
        last_model_response: str | None = None,
        last_tool_result: Mapping[str, Any] | None = None,
    ) -> Checkpoint:
        target = self._coerce_thread(thread)
        checkpoint = self.framework.checkpoint_thread(
            target,
            memory_summary=memory_summary,
            last_model_response=last_model_response,
            last_tool_result=dict(last_tool_result or {}) or None,
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        self._adopt_thread(self.framework.resume_thread(checkpoint))
        return checkpoint

    def resume_thread(
        self,
        checkpoint: str | Checkpoint,
    ) -> TaskThread:
        target = self._coerce_checkpoint(checkpoint)
        return self._adopt_thread(self.framework.resume_thread(target))

    def spawn_subagent(
        self,
        thread: str | TaskThread,
        *,
        title: str,
        task_prompt: str,
        mode: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskThread:
        parent = self._coerce_thread(thread)
        child = self.framework.spawn_subagent(
            parent,
            title=title,
            task_prompt=task_prompt,
            mode=mode,
            metadata=metadata,
        )
        self._adopt_thread(child)
        self._inherit_workstream(parent.thread_id, child.thread_id)
        return child

    def launch_background_job(
        self,
        thread: str | TaskThread,
        *,
        description: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> BackgroundJob:
        target = self._coerce_thread(thread)
        job = self.framework.launch_background_job(
            target,
            description=description,
            metadata=metadata,
        )
        self._jobs[job.job_id] = job
        self._update_active_job_ids(
            target.thread_id,
            lambda active: _dedupe_strings(active + (job.job_id,)),
        )
        return job

    def cancel_background_job(self, job: str | BackgroundJob) -> BackgroundJob:
        target = self._coerce_job(job)
        canceled = self.framework.cancel_background_job(target)
        self._jobs[canceled.job_id] = canceled
        self._update_active_job_ids(
            canceled.thread_id,
            lambda active: tuple(job_id for job_id in active if job_id != canceled.job_id),
        )
        return canceled

    def route_channel(
        self,
        *,
        channel: str,
        content: str,
        thread: str | TaskThread | None = None,
        sender: str = "aurelius",
        host: str = "cli",
        recipient: str | None = None,
        kind: str = "message",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._coerce_thread(thread) if thread is not None else None
        routing = self.framework.route_channel(
            host=host,
            channel=channel,
            thread=target,
            recipient=recipient,
            metadata=metadata,
        )
        envelope = MessageEnvelope(
            channel=channel,
            thread_id=target.thread_id if target is not None else None,
            sender=sender,
            kind=kind,
            content=content,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self._messages.append(envelope)
        if target is not None:
            self.active_thread_id = target.thread_id
            current_thread = self._threads.get(target.thread_id)
            if current_thread is not None:
                message_history = current_thread.message_history + (
                    FrameworkMessageEnvelope(
                        envelope_id=f"envelope-{uuid.uuid4()}",
                        channel_id=channel,
                        thread_id=target.thread_id,
                        sender=sender,
                        recipient=recipient,
                        kind=kind,
                        payload={"content": content, "metadata": dict(metadata or {})},
                        created_at=envelope.created_at,
                        session_id=self.session_id,
                        workstream_id=current_thread.workstream_id,
                        workspace=current_thread.workspace,
                        metadata=dict(metadata or {}),
                    ),
                )
                self._threads[target.thread_id] = replace(
                    current_thread,
                    message_history=message_history,
                    updated_at=_utc_now(),
                )
            self._append_toolless_step(target.thread_id, envelope)
        return {
            "routing": routing,
            "envelope": asdict(envelope),
        }

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        thread: str | TaskThread | None = None,
        call_id: str | None = None,
        host_step_id: str | None = None,
        status: str = "validated",
    ) -> dict[str, Any]:
        target = self._coerce_thread(thread) if thread is not None else None
        record = self.framework.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
            host_step_id=host_step_id,
            status=status,
            thread=target,
        )
        if target is not None:
            self._tool_calls.setdefault(target.thread_id, []).append(record)
            self.active_thread_id = target.thread_id
            self._append_toolless_step(
                target.thread_id,
                MessageEnvelope(
                    channel=target.channel or "tool_call",
                    thread_id=target.thread_id,
                    sender="tool",
                    kind="tool_call",
                    content=tool_name,
                    created_at=record["recorded_at"],
                    metadata={"arguments": copy.deepcopy(arguments)},
                ),
            )
        return record
