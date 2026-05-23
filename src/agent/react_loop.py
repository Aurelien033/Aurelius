"""ReAct-style agent loop for Aurelius.

Implements the Reason+Act paradigm from Yao et al. 2022
("ReAct: Synergizing Reasoning and Acting", arXiv:2210.03629).

The loop is intentionally minimal and model-agnostic:

    plan -> act (emit tool_call) -> observe (tool result) -> reflect -> ...

until either the model emits a final answer, the step budget is
exhausted, or a fatal error occurs. Tool execution is sandboxed by the
caller; the loop's own responsibilities are:

    * prompt construction (system + task + step history)
    * model invocation via a pluggable ``generate_fn``
    * tool-call parsing (via :class:`UnifiedToolCallParser`)
    * argument validation against the registered tool signature
    * wall-clock timeout on tool invocation
    * capturing every failure on the producing :class:`AgentStep`

There are no silent fallbacks. Every failure mode is materialised as an
``AgentStep.error`` string so the caller can audit the full trace.

The module deliberately uses only the Python standard library: no torch,
no transformers, no langchain. This keeps the loop runnable in the
training harness, in evaluation sandboxes, and in serving workers.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.tool_call_parser import (
    ParsedToolCall,
    ToolCallParseError,
    UnifiedToolCallParser,
)
from src.agent.slr_integration import prepare_slr_recall_context
from src.memory.amc_tier2 import AMCTier2Hook
from src.memory.amc_tier3 import AMCTier3Hook, TrustLevel
from src.memory.sdb_runtime import SDBMemoryRuntime
from src.reasoning.stochastic_latent_recall import SLRConfig, default_slr_config
from src.runtime.memory_quarantine import MemoryCandidate, build_memory_quarantine_report
from src.safety.admission_controller import (
    AdmissionAction,
    SafetyAdmissionController,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single turn in the ReAct trace.

    Roles mirror the ChatML convention: ``assistant`` for model output,
    ``tool`` for observation turns synthesised from tool results.
    """

    role: str
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None
    error: str | None = None


@dataclass
class AgentTrace:
    """Full transcript of a :meth:`ReActLoop.run` invocation."""

    task: str = ""
    system_prompt: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    status: str = "no_answer"  # one of {success, budget, error, no_answer}
    steps_used: int = 0
    # AMC Tier 2 usage counters
    tier2_calls: int = 0
    tier2_writes: int = 0
    tier3_promotions: int = 0
    safety_blocked: bool = False
    safety_action: str = ""


# ---------------------------------------------------------------------------
# Final-answer extraction
# ---------------------------------------------------------------------------


_FINAL_XML_RE = re.compile(
    r"<final_answer>(?P<body>.*?)</final_answer>",
    re.DOTALL,
)
# "Final Answer:" (case-insensitive) at the start of a line.
_FINAL_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*final answer[ \t]*:[ \t]*(?P<body>.*)$",
)


def _extract_final_answer(text: str) -> str | None:
    """Return the final-answer body if present, else ``None``.

    Two forms are recognised:

    * ``<final_answer>...</final_answer>`` (anywhere, multiline body)
    * ``Final Answer: ...`` at the start of a line (single-line body)

    The XML form is preferred when both appear. The extracted body is
    stripped of leading/trailing whitespace but otherwise verbatim.
    """
    if not text:
        return None
    m = _FINAL_XML_RE.search(text)
    if m is not None:
        return m.group("body").strip()
    m = _FINAL_PREFIX_RE.search(text)
    if m is not None:
        return m.group("body").strip()
    return None


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class ReActLoop:
    """ReAct reasoning+action loop.

    Parameters
    ----------
    generate_fn:
        Callable taking the current message list and returning the raw
        model string for this step. Must be deterministic for a fixed
        input if the caller wants reproducible traces.
    tool_registry:
        Mapping of tool name to a Python callable. Keyword arguments
        are supplied from the parsed tool call; positional-only tools
        are rejected.
    max_steps:
        Upper bound on the number of model turns. Each assistant turn
        counts as one step regardless of whether it produced a tool
        call.
    max_tool_seconds:
        Wall-clock timeout per tool invocation. Timeouts surface as
        ``AgentStep.error`` containing the substring ``"timeout"``; the
        loop continues.
    parser:
        A parser exposing ``.parse(text) -> list[ParsedToolCall]``.
        Defaults to :class:`UnifiedToolCallParser` which auto-detects
        XML or JSON.
    """

    def __init__(
        self,
        generate_fn: Callable[[list[dict]], str],
        tool_registry: dict[str, Callable[..., Any]],
        max_steps: int = 8,
        max_tool_seconds: float = 5.0,
        parser: Any = None,
        tier2_hook: AMCTier2Hook | None = None,
        tier3_hook: AMCTier3Hook | None = None,
        safety_admission: SafetyAdmissionController | None = None,
        slr_config: SLRConfig | None = None,
        sdb_runtime: SDBMemoryRuntime | None = None,
        session_id: str = "react-session",
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if not isinstance(tool_registry, dict):
            raise TypeError("tool_registry must be a dict")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_tool_seconds <= 0:
            raise ValueError("max_tool_seconds must be > 0")
        self._generate = generate_fn
        self._tools = tool_registry
        self._max_steps = max_steps
        self._tool_timeout = float(max_tool_seconds)
        self._parser = parser if parser is not None else UnifiedToolCallParser()
        self._tier2_hook = tier2_hook
        self._tier2_calls = 0  # number of times recall was inserted
        self._tier2_writes = 0
        self._tier3_hook = tier3_hook
        self._tier3_promotions = 0
        self._safe_adm = safety_admission  # alias used by Tier-2 write gating below
        self._slr_config = slr_config if slr_config is not None else default_slr_config()
        self._sdb_runtime = sdb_runtime
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, task: str, system_prompt: str = "") -> AgentTrace:
        """Drive the loop until terminal condition is reached."""
        if not isinstance(task, str):
            raise TypeError("task must be a str")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a str")

        trace = AgentTrace(task=task, system_prompt=system_prompt)

        # Safety pre-check: assess user input before any generation
        if self._safe_adm is not None:
            try:
                decision = self._safe_adm.assess_input(task)
                if not decision.allowed:
                    trace.safety_blocked = True
                    trace.safety_action = decision.action.value
                    trace.status = "blocked"
                    trace.steps.append(
                        AgentStep(
                            role="assistant",
                            content="",
                            error=f"safety_blocked: {decision.reason}",
                        )
                    )
                    return trace
            except Exception:  # noqa: BLE001
                # Safety controller crash → fail closed; do not proceed to generation
                trace.safety_blocked = True
                trace.safety_action = "error"
                trace.status = "error"
                trace.steps.append(
                    AgentStep(
                        role="assistant",
                        content="",
                        error=(
                            "safety_controller_error: admission controller "
                            "raised an exception during pre-check"
                        ),
                    )
                )
                return trace

        # Handle empty task upfront: we still make one call so callers
        # relying on system_prompt-only flows are supported, but if the
        # task is empty AND there is no system prompt we short-circuit.
        if task.strip() == "" and system_prompt.strip() == "":
            trace.status = "no_answer"
            return trace

        for step_idx in range(self._max_steps):
            messages = self._build_messages(
                task=task,
                system_prompt=system_prompt,
                steps=trace.steps,
                step_idx=step_idx,
            )
            try:
                raw = self._generate(messages)
            except Exception as exc:  # noqa: BLE001 - generate_fn is untrusted
                trace.steps.append(
                    AgentStep(
                        role="assistant",
                        content="",
                        error=f"generate_fn raised: {type(exc).__name__}: {exc}",
                    )
                )
                trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
                trace.status = "error"
                return trace

            if not isinstance(raw, str):
                trace.steps.append(
                    AgentStep(
                        role="assistant",
                        content="",
                        error=f"generate_fn returned non-str: {type(raw).__name__}",
                    )
                )
                trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
                trace.status = "error"
                return trace

            # Terminal: final answer.
            final = _extract_final_answer(raw)
            if final is not None:
                trace.steps.append(AgentStep(role="assistant", content=raw))
                trace.final_answer = final
                trace.status = "success"
                trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
                trace.tier2_calls = self._tier2_calls
                trace.tier2_writes = self._tier2_writes
                trace.tier3_promotions = self._tier3_promotions
                return trace

            # Parse any tool calls in the assistant output.
            assistant_step = AgentStep(role="assistant", content=raw)
            tool_calls: list[ParsedToolCall] = []
            try:
                tool_calls = list(self._parser.parse(raw))
            except ToolCallParseError as exc:
                assistant_step.error = f"tool_call_parse_error: {exc}"
            except Exception as exc:  # noqa: BLE001
                assistant_step.error = f"tool_call_parse_error: {type(exc).__name__}: {exc}"

            if not tool_calls:
                # The model reasoned but neither finalised nor called a
                # tool. Record and continue; next iteration will see this
                # turn as history and can self-correct. If we exhaust
                # budget in this state we return "budget".
                trace.steps.append(assistant_step)
                if self._tier2_hook is not None:
                    if self._write_to_tier2(assistant_step.content, "assistant"):
                        self._tier2_writes += 1
                continue

            # Attach the first tool's bookkeeping to the assistant step
            # for convenience, then emit observation steps for each.
            first = tool_calls[0]
            assistant_step.tool_name = first.name
            assistant_step.tool_input = dict(first.arguments)
            trace.steps.append(assistant_step)

            for call in tool_calls:
                obs = self._dispatch_tool(call)
                trace.steps.append(obs)
                if self._tier2_hook is not None:
                    obs_text = obs.tool_output or obs.error or ""
                    if self._write_to_tier2(obs_text, "tool"):
                        self._tier2_writes += 1

        # Budget exhausted without final answer.
        trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
        trace.tier2_calls = self._tier2_calls
        trace.tier2_writes = self._tier2_writes
        trace.tier3_promotions = self._tier3_promotions
        # Tier 3: promote Tier 2 entries into LTM on budget exhaust
        if self._tier3_hook is not None and self._tier2_hook is not None:
            tier3_errors: list[str] = []
            try:
                promoted = 0
                for ev in list(self._tier2_hook.episodic._entries):
                    try:
                        confidence = ev.importance
                        trust = None
                        if confidence >= 0.8:
                            trust = TrustLevel.TRUSTED
                        elif confidence < 0.3:
                            trust = TrustLevel.UNVERIFIED
                        entry = self._tier3_hook.promote(
                            key=ev.id,
                            value=ev.content,
                            source_tier2_id=ev.id,
                            confidence=confidence,
                            trust_level=trust,
                        )
                        if entry is not None:
                            promoted += 1
                    except Exception as exc:  # noqa: BLE001
                        tier3_errors.append(
                            f"tier3_promote_error:{ev.id}:{type(exc).__name__}:{exc}"
                        )
                # Now run the Tier 3 lifecycle sweep
                try:
                    self._tier3_hook.consolidate()
                except Exception as exc:  # noqa: BLE001
                    tier3_errors.append(f"tier3_consolidate_error:{type(exc).__name__}:{exc}")
                self._tier3_promotions = promoted
                if tier3_errors:
                    if trace.steps:
                        existing = trace.steps[-1].error or ""
                        sep = " | " if existing else ""
                        trace.steps[-1].error = existing + sep + "; ".join(tier3_errors)
                    else:
                        trace.steps.append(
                            AgentStep(
                                role="assistant",
                                content="",
                                error="; ".join(tier3_errors),
                            )
                        )
            except Exception as exc:
                err_msg = f"tier3_outer_error:{type(exc).__name__}:{exc}"
                if trace.steps:
                    existing = trace.steps[-1].error or ""
                    sep = " | " if existing else ""
                    trace.steps[-1].error = existing + sep + err_msg
                else:
                    trace.steps.append(AgentStep(role="assistant", content="", error=err_msg))
        trace.tier3_promotions = self._tier3_promotions
        trace.status = "budget"
        return trace

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        task: str,
        system_prompt: str,
        steps: list[AgentStep],
        step_idx: int = 0,
    ) -> list[dict]:
        """Render the full conversation as a plain list-of-dicts.

        We use dicts rather than :class:`Message` to keep this module
        import-light: the caller's ``generate_fn`` can transform this
        list to whatever wire format its model demands.
        """
        messages: list[dict] = []
        if self._slr_config.enabled and step_idx > 0:
            slr_context = prepare_slr_recall_context(
                config=self._slr_config,
                query_text=task,
                query_id=f"{self._session_id}:step:{step_idx}",
                tier2_hook=self._tier2_hook,
                tier3_hook=self._tier3_hook,
                sdb_runtime=self._sdb_runtime,
                session_id=self._session_id,
                step_idx=step_idx,
            )
            if slr_context is not None:
                messages.append({"role": "user", "content": slr_context.preamble})
        # AMC Tier 2: inject episodic retrieval on steps > 0
        if self._tier2_hook is not None and step_idx > 0:
            recalled = self._tier2_hook.retrieve(task, limit=self._tier2_hook.config.max_retrieved)
            if recalled:
                candidates = [
                    MemoryCandidate(content=entry.content, source=f"tier2:{entry.role}")
                    for entry in recalled
                ]
                ctrl = self._safe_adm
                report = (
                    build_memory_quarantine_report(candidates, controller=ctrl)
                    if ctrl is not None
                    else {
                        "trusted": [{"content": c.content, "source": c.source} for c in candidates],
                        "quarantined": [],
                    }
                )
                trusted = report.get("trusted", [])
                if trusted:
                    lines = ["[Tier-2 episodic memory recall]:"]
                    for rec in trusted:
                        source = rec.get("source", "tier2")
                        body = rec.get("content", "")
                        if body:
                            lines.append(f"- [{source}] {body}")
                    preamble = "\n".join(lines)
                    messages.append({"role": "user", "content": preamble})
                    self._tier2_calls += 1
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        for step in steps:
            if step.role == "assistant":
                messages.append({"role": "assistant", "content": step.content})
            elif step.role == "tool":
                # Surface the observation in a form the model can read.
                body = step.tool_output if step.tool_output is not None else (step.error or "")
                prefix = f"[{step.tool_name}] " if step.tool_name else ""
                messages.append({"role": "tool", "content": f"{prefix}{body}"})
            else:
                # Unknown role: pass through verbatim.
                messages.append({"role": step.role, "content": step.content})
        return messages

    def _dispatch_tool(self, call: ParsedToolCall) -> AgentStep:
        """Validate and execute one tool call, returning an observation.

        Every failure path (unknown tool, bad args, exception, timeout)
        is captured on ``AgentStep.error``; the loop never re-raises.
        """
        step = AgentStep(
            role="tool",
            content="",
            tool_name=call.name,
            tool_input=dict(call.arguments),
        )

        fn = self._tools.get(call.name)
        if fn is None:
            step.error = f"unknown_tool: {call.name!r}"
            return step

        # Best-effort argument validation against the callable signature.
        # This rejects argument names the target does not accept, which
        # blocks trivial "pollute kwargs" injection attempts.
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            # Builtin/C callable with no introspectable signature: skip
            # validation and let the call itself surface TypeErrors.
            sig = None

        kwargs = dict(call.arguments)
        if sig is not None:
            params = sig.parameters
            accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts_var_kw:
                allowed = {
                    name
                    for name, p in params.items()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                unexpected = set(kwargs) - allowed
                if unexpected:
                    step.error = (
                        "invalid_arguments: unexpected keys "
                        f"{sorted(unexpected)!r} for tool {call.name!r}"
                    )
                    return step

        # Execute with wall-clock timeout.
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: fn(**kwargs))
                try:
                    result = future.result(timeout=self._tool_timeout)
                except concurrent.futures.TimeoutError:
                    # The worker thread cannot be killed, but the loop
                    # moves on; the daemon-ish cleanup at executor exit
                    # lets the thread finish in the background.
                    future.cancel()
                    step.error = f"tool_error: timeout after {self._tool_timeout:.3f}s"
                    return step
        except Exception as exc:  # noqa: BLE001 - tool code is untrusted
            step.error = f"tool_error: {type(exc).__name__}: {exc}"
            return step

        # Normalise output to a string so observation rendering is trivial.
        if isinstance(result, str):
            step.tool_output = result
        else:
            try:
                step.tool_output = repr(result)
            except Exception as exc:  # noqa: BLE001
                step.error = f"tool_error: unrepr-able result: {exc}"
                return step
        step.content = step.tool_output
        return step

    def _write_to_tier2(self, content: str, role: str) -> bool:
        """Write *content* to Tier-2 only if the admission controller approves it.

        Returns True when the observation was stored, False when it was
        quarantined or blocked.  When no controller is registered (legacy /
        test mode) all writes are allowed.
        """
        if self._safe_adm is None:
            if self._tier2_hook is not None:
                self._tier2_hook.observe(
                    role,
                    content,
                    surprise=0.5 if role == "assistant" else 0.7,
                )
            return True
        decision = self._safe_adm.assess_memory_candidate(
            content,
            source=f"react_loop:{role}",
        )
        if not decision.allowed or decision.action in (
            AdmissionAction.QUARANTINE,
            AdmissionAction.BLOCK,
        ):
            return False
        if self._tier2_hook is not None:
            self._tier2_hook.observe(
                role,
                content,
                surprise=0.5 if role == "assistant" else 0.7,
            )
        return True


__all__ = ["AgentStep", "AgentTrace", "ReActLoop"]
