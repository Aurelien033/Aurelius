"""Workflow normalization and step execution for the shell."""

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
from src.model.interface_framework import TaskThread
from src.ui.shell.commands import _normalize_decision
from src.ui.shell.models import AureliusShellError
from src.ui.shell.models import MessageEnvelope
from src.ui.shell.models import WorkflowRun
from src.ui.shell.models import WorkflowStep
from src.ui.shell.models import _KNOWN_WORKFLOW_STEP_KINDS
from src.ui.shell.util import _utc_now

__all__ = [
    "ShellWorkflowMixin",
]


class ShellWorkflowMixin:
    """Workflow execution surface of the shell."""

    def execute_workflow(
        self,
        thread: str | TaskThread,
        workflow: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        approval_resolver: Callable[[ApprovalRequest], Any] | None = None,
    ) -> WorkflowRun:
        current_thread = self._coerce_thread(thread)
        if current_thread.workstream_id is None:
            workstream = (
                self._workstreams.get(self.active_workstream_id)
                if self.active_workstream_id
                else None
            )
            if workstream is None:
                workstream = self.create_workstream(
                    f"{current_thread.title} workstream",
                    workspace=current_thread.workspace,
                )
            current_thread = replace(
                current_thread,
                workstream_id=workstream.workstream_id,
                workstream_name=workstream.name,
                updated_at=_utc_now(),
            )
            self._threads[current_thread.thread_id] = current_thread
            self._attach_thread_to_workstream(workstream.workstream_id, current_thread.thread_id)
        workflow_name, normalized_steps, metadata = self._normalize_workflow(workflow)
        run_id = f"workflow-{uuid.uuid4()}"
        created_at = _utc_now()
        transcript: list[dict[str, Any]] = []
        status = "running"
        halted_step_index: int | None = None
        halted_reason: str | None = None
        final_artifact: dict[str, Any] | None = None

        for step in normalized_steps:
            try:
                step_result = self._execute_workflow_step(
                    current_thread,
                    step,
                    approval_resolver=approval_resolver,
                )
                transcript.append(_transcript_entry(step, "result", step_result))
                if step.kind == "subagent" and isinstance(step_result, dict):
                    child_thread_id = step_result.get("thread", {}).get("thread_id")
                    if isinstance(child_thread_id, str) and child_thread_id in self._threads:
                        current_thread = self._threads[child_thread_id]
                elif step.kind == "checkpoint" and isinstance(step_result, dict):
                    checkpoint_id = step_result.get("checkpoint", {}).get("checkpoint_id")
                    if isinstance(checkpoint_id, str) and checkpoint_id in self._checkpoints:
                        current_thread = self._threads[self._checkpoints[checkpoint_id].thread_id]
                current_thread = self._threads.get(current_thread.thread_id, current_thread)
                if step.kind == "approval":
                    decision = step_result.get("decision", "pending")
                    if decision == "pending":
                        status = "pending_approval"
                        halted_step_index = step.index
                        halted_reason = "approval pending"
                        final_artifact = step_result
                        break
                    if decision == "deny":
                        status = "blocked"
                        halted_step_index = step.index
                        halted_reason = "approval denied"
                        final_artifact = step_result
                        break
                final_artifact = step_result
            except Exception as exc:  # noqa: BLE001 - explicit failure surface
                status = "failed"
                halted_step_index = step.index
                halted_reason = str(exc)
                final_artifact = {"error": str(exc)}
                transcript.append(_transcript_entry(step, "error", str(exc)))
                break
        else:
            if status == "running":
                status = "completed"

        run = WorkflowRun(
            run_id=run_id,
            thread_id=current_thread.thread_id,
            workflow_name=workflow_name,
            status=status,
            steps=normalized_steps,
            transcript=tuple(transcript),
            created_at=created_at,
            updated_at=_utc_now(),
            halted_step_index=halted_step_index,
            halted_reason=halted_reason,
            final_artifact=final_artifact,
            metadata=metadata,
        )
        self._threads[current_thread.thread_id] = replace(
            current_thread,
            status=status,
            updated_at=_utc_now(),
        )
        self.active_thread_id = current_thread.thread_id
        self._workflow_runs[run.run_id] = run
        return run

    def _append_toolless_step(self, thread_id: str, envelope: MessageEnvelope) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return
        steps = list(thread.steps)
        steps.append(
            {
                "kind": envelope.kind,
                "channel": envelope.channel,
                "content": envelope.content,
                "created_at": envelope.created_at,
                "metadata": copy.deepcopy(envelope.metadata),
            }
        )
        self._threads[thread_id] = replace(thread, steps=tuple(steps), updated_at=_utc_now())

    def _normalize_workflow(
        self,
        workflow: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> tuple[str, tuple[WorkflowStep, ...], dict[str, Any]]:
        if isinstance(workflow, Mapping):
            workflow_name = workflow.get("name") or workflow.get("title") or "workflow"
            metadata = dict(workflow.get("metadata", {}))
            raw_steps = workflow.get("steps")
        elif isinstance(workflow, Sequence) and not isinstance(workflow, (str, bytes)):
            workflow_name = "workflow"
            metadata = {}
            raw_steps = workflow
        else:
            raise AureliusShellError(
                "workflow must be a mapping with 'steps' or a sequence of step mappings"
            )
        if raw_steps is None:
            raise AureliusShellError("workflow is missing a steps collection")
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise AureliusShellError("workflow steps must be a sequence of mappings")
        normalized_steps: list[WorkflowStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise AureliusShellError(
                    f"workflow step {index} must be a mapping, got {type(raw_step).__name__}"
                )
            kind_value = raw_step.get("kind", raw_step.get("type"))
            if not isinstance(kind_value, str) or not kind_value.strip():
                raise AureliusShellError(f"workflow step {index} is missing kind/type")
            kind = kind_value.strip().lower()
            if kind not in _KNOWN_WORKFLOW_STEP_KINDS:
                raise AureliusShellError(f"workflow step {index} has unsupported kind {kind!r}")
            payload = {
                key: copy.deepcopy(value)
                for key, value in raw_step.items()
                if key not in {"kind", "type"}
            }
            normalized_steps.append(
                WorkflowStep(
                    index=index,
                    kind=kind,
                    payload=payload,
                    approval_required=kind == "approval",
                )
            )
        return str(workflow_name), tuple(normalized_steps), metadata

    def _execute_workflow_step(
        self,
        thread: TaskThread,
        step: WorkflowStep,
        *,
        approval_resolver: Callable[[ApprovalRequest], Any] | None,
    ) -> dict[str, Any]:
        payload = step.payload
        if step.kind == "message":
            content = payload.get("content", payload.get("text", ""))
            if not isinstance(content, str) or not content.strip():
                raise AureliusShellError("message workflow steps require content")
            channel = payload.get("channel") or thread.channel or "workflow"
            routing = self.route_channel(
                channel=str(channel),
                content=content,
                thread=thread,
                sender=str(payload.get("sender", "workflow")),
                host=thread.host,
                recipient=payload.get("recipient"),
                kind="message",
                metadata=dict(payload.get("metadata", {})),
            )
            return routing
        if step.kind == "approval":
            approval = self.request_approval(
                thread,
                category=str(payload.get("category", "")),
                action_summary=str(payload.get("action_summary", "")),
                affected_resources=tuple(payload.get("affected_resources", ())),
                reason=str(payload.get("reason", "")),
                reversible=bool(payload.get("reversible", True)),
                minimum_scope=str(payload.get("minimum_scope", "")),
                metadata=payload.get("metadata"),
            )
            decision = payload.get("decision")
            if decision is None and approval_resolver is not None:
                decision = approval_resolver(approval)
            normalized = _normalize_decision(decision)
            resolved_approval = replace(
                approval,
                decision=normalized,
                decided_at=_utc_now() if normalized != "pending" else None,
            )
            self._approvals[resolved_approval.approval_id] = resolved_approval
            return {
                "approval": asdict(resolved_approval),
                "decision": normalized,
                "status": _APPROVAL_STEP_STATUS[normalized],
            }
        if step.kind == "tool_call":
            tool_name = _require_step_field(payload, "tool_name", "tool_call")
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                raise AureliusShellError("tool_call workflow steps require arguments dict")
            record = self.record_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                thread=thread,
                call_id=payload.get("call_id"),
                host_step_id=payload.get("host_step_id"),
                status=str(payload.get("status", "validated")),
            )
            return {"tool_call": record}
        if step.kind == "checkpoint":
            memory_summary = _require_step_field(payload, "memory_summary", "checkpoint")
            checkpoint = self.checkpoint_thread(
                thread,
                memory_summary=memory_summary,
                last_model_response=payload.get("last_model_response"),
                last_tool_result=payload.get("last_tool_result"),
            )
            return {"checkpoint": asdict(checkpoint)}
        if step.kind == "background_job":
            description = _require_step_field(payload, "description", "background_job")
            job = self.launch_background_job(
                thread,
                description=description,
                metadata=payload.get("metadata"),
            )
            return {"job": asdict(job)}
        if step.kind == "subagent":
            title = _require_step_field(payload, "title", "subagent")
            task_prompt = _require_step_field(payload, "task_prompt", "subagent")
            child = self.spawn_subagent(
                thread,
                title=title,
                task_prompt=task_prompt,
                mode=payload.get("mode"),
                metadata=payload.get("metadata"),
            )
            return {"thread": asdict(child)}
        raise AureliusShellError(f"unsupported workflow step kind: {step.kind!r}")


_APPROVAL_STEP_STATUS = {"allow": "approved", "deny": "denied", "pending": "pending"}


def _require_step_field(payload: Mapping[str, Any], key: str, step_kind: str) -> str:
    """Return a required non-empty string field of a workflow step payload."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AureliusShellError(f"{step_kind} workflow steps require {key}")
    return value


def _transcript_entry(step: WorkflowStep, key: str, value: Any) -> dict[str, Any]:
    """Build one workflow transcript record for a step outcome."""
    return {
        "step_index": step.index,
        "kind": step.kind,
        "approval_required": step.approval_required,
        key: value,
    }
