"""
ComposerAgent — autonomous coding agent inspired by Cursor Composer 2.5.

This is the main orchestration loop. The agent:
  1. Receives a natural language task
  2. Explores the codebase to understand context
  3. Plans changes across files
  4. Executes edits using the DiffEngine
  5. Verifies with tests/lint (TerminalSandbox)
  6. Iterates until the task is complete or max turns exhausted

The agent uses a pluggable model backend (local or API) and can work
with any LLM. The "Apply Model" specialization is the part that would
benefit from fine-tuning, but the system works with general models too.

Architecture:
  ┌─────────────────────────────────────────────┐
  │                ComposerAgent                 │
  │                                              │
  │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
  │  │  Planner  │  │ Explorer │  │   Editor    │ │
  │  │ (model)   │  │(indexer) │  │  (diffs)    │ │
  │  └──────────┘  └──────────┘  └────────────┘ │
  │        │             │              │         │
  │  ┌─────▼─────────────▼──────────────▼──────┐ │
  │  │           Execution Loop                 │ │
  │  │  plan → explore → edit → verify → repeat │ │
  │  └──────────────────────────────────────────┘ │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .checkpoint_rollback import CheckpointRollback
from .codebase_indexer import CodebaseIndexer
from .context_assembler import ContextAssembler, ContextBudget, ContextRequest
from .diff_engine import DiffApplier, DiffGenerator, DiffParser, FileEdit
from .edit_verifier import EditVerifier
from .multifile_orchestrator import MultiFileEditOrchestrator
from .terminal_sandbox import TerminalSandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class AgentAction(str, Enum):  # noqa: UP042
    """Actions the agent can take in the loop."""

    EXPLORE = "explore"  # search the codebase
    READ = "read"  # read a specific file
    PLAN = "plan"  # produce an edit plan
    EDIT = "edit"  # execute file edits
    RUN = "run"  # run a command
    VERIFY = "verify"  # check results
    COMPLETE = "complete"  # task is done
    ABORT = "abort"  # task cannot be completed


@dataclass
class AgentTurn:
    """A single turn in the agent loop."""

    turn_id: int
    action: AgentAction
    thought: str  # model's reasoning
    payload: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    tokens_used: int = 0
    wall_time_seconds: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    max_turns: int = 20
    max_explore_turns: int = 6
    max_edit_turns: int = 8
    max_tokens_per_turn: int = 8000
    auto_verify: bool = True
    auto_lint: bool = True
    auto_test: bool = True
    test_command: str = "pytest -x"
    lint_command: str = "ruff check ."
    model_fn: Callable[..., str] | None = None  # pluggable model
    # Optional exact token counter (e.g. a tokenizer's encode-len). When None,
    # a ~4-chars/token estimate is used (consistent with the truncation budget).
    token_counter: Callable[[str], int] | None = None


@dataclass
class AgentResult:
    """Final result of an agent session."""

    success: bool
    task_description: str
    turns: list[AgentTurn] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_wall_time: float = 0.0
    final_state: str = ""  # summary for the user
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

COMPOSER_SYSTEM_PROMPT = """You are an expert software engineering agent.
Your job is to understand a coding task, explore the codebase, plan changes,
make precise edits, and verify your work.

## Available tools

You can respond with ONE of these action formats per turn:

### EXPLORE — search the codebase
```
<action>explore</action>
<query>what you want to find</query>
```

### READ — read a specific file
```
<action>read</action>
<file>relative/path/to/file.py</file>
```

### PLAN — produce an edit plan
```
<action>plan</action>
<plan>
- File: path/to/file1.py
  Change: describe what to change and why
- File: path/to/file2.py
  Change: describe what to change and why
</plan>
```

### EDIT — produce diffs for files
```
<action>edit</action>
<diffs>
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,5 +10,7 @@
 context line
-removed line
+added line
 context line
</diffs>
```

### RUN — execute a command
```
<action>run</action>
<command>pytest tests/ -x</command>
```

### COMPLETE — task is done
```
<action>complete</action>
<summary>what was accomplished</summary>
```

## Rules
1. EXPLORE before you EDIT. You must understand the codebase first.
2. PLAN before you EDIT. Think about all files that need changes.
3. Produce minimal, precise unified diffs. Only change what needs changing.
4. After editing, RUN tests or lint to verify.
5. If verification fails, read the error output and fix it.
6. Stay within the repository directory. Do not read or edit files outside it.
7. If you cannot complete the task, use ABORT and explain why.

## Context
Repository: {repo_root}
Task: {task_description}
"""


# ---------------------------------------------------------------------------
# ComposerAgent
# ---------------------------------------------------------------------------


class ComposerAgent:
    """Autonomous coding agent that explores, plans, edits, and verifies.

    Usage:
      agent = ComposerAgent(
          repo_root=Path("/path/to/repo"),
          model_fn=my_model_generate,
      )
      result = agent.run("Add error handling to the API endpoints")
    """

    def __init__(
        self,
        repo_root: Path,
        model_fn: Callable[..., str] | None = None,
        config: AgentConfig | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.model_fn = model_fn
        self.config = config or AgentConfig()

        # Core subsystems
        self._indexer = CodebaseIndexer(self.repo_root)
        self._sandbox = TerminalSandbox(self.repo_root)
        self._diff_applier = DiffApplier(self.repo_root)
        self._context_assembler = ContextAssembler(self.repo_root, self._indexer)
        self._edit_orchestrator = MultiFileEditOrchestrator(self.repo_root, self._indexer)
        self._verifier = EditVerifier(self.repo_root, self._sandbox)
        self._checkpoint = CheckpointRollback(self.repo_root)

        # State
        self._turns: list[AgentTurn] = []
        self._conversation: list[dict] = []
        self._total_tokens = 0
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Execute a coding task autonomously.

        This is the main entry point. It runs the plan→explore→edit→verify
        loop until the task is complete or max turns exhausted.
        """
        self._start_time = time.monotonic()

        # Build index if needed
        if self._indexer.stats.total_files == 0:
            logger.info("Building codebase index...")
            self._indexer.build()

        # Initialize conversation with system prompt
        system_prompt = COMPOSER_SYSTEM_PROMPT.format(
            repo_root=str(self.repo_root),
            task_description=task,
        )
        self._conversation = [{"role": "system", "content": system_prompt}]
        self._add_user_message(f"Please complete this task: {task}")

        turn_id = 0
        explore_turns = 0
        edit_turns = 0
        result = AgentResult(success=False, task_description=task)

        while turn_id < self.config.max_turns:
            turn_id += 1
            turn_start = time.monotonic()

            try:
                response = self._call_model()
            except Exception as exc:
                result.errors.append(f"Model call failed on turn {turn_id}: {exc}")
                logger.exception("Model call failed")
                break

            action, payload = self._parse_response(response)
            turn = AgentTurn(
                turn_id=turn_id,
                action=action,
                thought=response[:500],
                payload=payload,
                wall_time_seconds=time.monotonic() - turn_start,
            )

            # Execute the action
            outcome = self._execute_action(action, payload, result)
            turn.result = outcome
            self._turns.append(turn)
            self._conversation.append({"role": "assistant", "content": response})
            self._conversation.append({"role": "user", "content": outcome})

            if action == AgentAction.EXPLORE:
                explore_turns += 1
                if explore_turns >= self.config.max_explore_turns:
                    self._add_user_message(
                        "You have reached the maximum explore turns. "
                        "Please PLAN your changes and start EDITING."
                    )

            if action == AgentAction.EDIT:
                edit_turns += 1
                if edit_turns >= self.config.max_edit_turns:
                    self._add_user_message(
                        "You have reached the maximum edit turns. "
                        "Please verify your work and COMPLETE."
                    )

            if action == AgentAction.COMPLETE:
                result.success = True
                result.final_state = payload.get("summary", "Task completed.")
                break

            if action == AgentAction.ABORT:
                result.success = False
                result.final_state = payload.get("reason", "Task aborted by agent.")
                break

        result.turns = list(self._turns)
        if not result.files_changed:
            result.files_changed = [
                f
                for t in self._turns
                if t.action == AgentAction.EDIT
                for f in (t.payload.get("files_changed", []))
            ]
        result.total_tokens = self._total_tokens
        result.total_wall_time = time.monotonic() - self._start_time
        return result

    # ------------------------------------------------------------------
    # Internal: model interaction
    # ------------------------------------------------------------------

    def _call_model(self) -> str:
        """Call the model with the current conversation."""
        if self.model_fn is None:
            raise RuntimeError("No model function configured. Set model_fn on ComposerAgent.")

        # Truncate the CONVERSATION (not the rendered string) to fit the budget,
        # keeping the system message + a rolling window of the most recent turns.
        # This preserves the agent's own prior actions (assistant turns) within
        # the window and avoids splitting on a magic "<user>" marker that a diff
        # or code payload could contain.
        convo = self._truncate_conversation(self._conversation)

        prompt_parts = []
        for msg in convo:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<system>\n{content}\n</system>")
            elif role == "user":
                prompt_parts.append(f"<user>\n{content}\n</user>")
            elif role == "assistant":
                prompt_parts.append(f"<assistant>\n{content}\n</assistant>")

        full_prompt = "\n".join(prompt_parts) + "\n<assistant>\n"

        response = self.model_fn(full_prompt)
        self._total_tokens += self._count_tokens(full_prompt) + self._count_tokens(response)
        return response

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Uses ``config.token_counter`` when provided
        (e.g. a real tokenizer), else a ~4-chars/token estimate. NOTE: prior
        code used ``str.split()`` here, which counts WORDS and undercounts
        tokens by ~1.3-1.6x."""
        tc = self.config.token_counter
        if tc is not None:
            try:
                return int(tc(text))
            except Exception:  # pragma: no cover - defensive
                logger.debug("token_counter raised; falling back to char estimate")
        return max(1, len(text) // 4)

    def _truncate_conversation(self, convo: list[dict]) -> list[dict]:
        """Keep the system message + the most recent whole messages that fit the
        character budget (``max_tokens_per_turn * 4``, matching the token
        estimate). Rolling window over BOTH user and assistant turns, so the
        agent retains its own recent actions."""
        if not convo:
            return convo
        budget = self.config.max_tokens_per_turn * 4  # chars (~4 chars/token)
        head = [convo[0]] if convo[0].get("role") == "system" else []
        rest = convo[len(head):]
        used = sum(len(m.get("content", "")) for m in head)
        kept_rev: list[dict] = []
        for m in reversed(rest):
            c = len(m.get("content", ""))
            if used + c > budget and kept_rev:
                break  # keep at least one recent message even if oversized
            kept_rev.append(m)
            used += c
        return head + list(reversed(kept_rev))

    def _add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._conversation.append({"role": "user", "content": content})

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> tuple[AgentAction, dict]:
        """Parse the model's response into an action and payload."""
        # Try structured XML-style format first
        action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
        if action_match:
            action_str = action_match.group(1).strip().lower()
            payload: dict = {}

            if action_str == "explore":
                query = re.search(r"<query>(.*?)</query>", response, re.DOTALL)
                payload["query"] = query.group(1).strip() if query else ""

            elif action_str == "read":
                file_match = re.search(r"<file>(.*?)</file>", response, re.DOTALL)
                payload["file"] = file_match.group(1).strip() if file_match else ""

            elif action_str == "plan":
                plan = re.search(r"<plan>(.*?)</plan>", response, re.DOTALL)
                payload["plan"] = plan.group(1).strip() if plan else ""

            elif action_str == "edit":
                diffs = re.search(r"<diffs>(.*?)</diffs>", response, re.DOTALL)
                payload["diffs"] = diffs.group(1).strip() if diffs else ""

            elif action_str == "run":
                cmd = re.search(r"<command>(.*?)</command>", response, re.DOTALL)
                payload["command"] = cmd.group(1).strip() if cmd else ""

            elif action_str == "complete":
                summary = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
                payload["summary"] = summary.group(1).strip() if summary else "Task completed."

            elif action_str == "abort":
                reason = re.search(r"<reason>(.*?)</reason>", response, re.DOTALL)
                payload["reason"] = reason.group(1).strip() if reason else "Unknown reason."

            try:
                return AgentAction(action_str), payload
            except ValueError:
                pass

        # Fallback: heuristic parsing
        return ComposerAgent._heuristic_parse(response)

    @staticmethod
    def _heuristic_parse(response: str) -> tuple[AgentAction, dict]:
        """Fallback parsing when structured format fails."""
        lower = response.lower()
        payload: dict = {}

        if "```diff" in response or "--- a/" in response:
            diffs_match = re.search(r"```diff\n(.*?)```", response, re.DOTALL)
            if diffs_match:
                payload["diffs"] = diffs_match.group(1)
            else:
                # Try to find --- a/ pattern directly
                diff_match = re.search(r"--- a/.*", response)
                if diff_match:
                    payload["diffs"] = response
            return AgentAction.EDIT, payload

        if "pytest" in lower or "npm test" in lower or "cargo test" in lower:
            payload["command"] = response.strip()
            return AgentAction.RUN, payload

        if "done" in lower or "complete" in lower or "finished" in lower:
            payload["summary"] = response[:500]
            return AgentAction.COMPLETE, payload

        if "cannot" in lower or "unable" in lower or "abort" in lower:
            payload["reason"] = response[:500]
            return AgentAction.ABORT, payload

        # Default: treat as plan/thought
        payload["plan"] = response[:1000]
        return AgentAction.PLAN, payload

    # ------------------------------------------------------------------
    # Internal: action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: AgentAction, payload: dict, result: AgentResult) -> str:
        """Execute an agent action and return a result string for the model."""
        if action == AgentAction.EXPLORE:
            return self._do_explore(payload.get("query", ""))

        elif action == AgentAction.READ:
            return self._do_read(payload.get("file", ""))

        elif action == AgentAction.PLAN:
            return (
                f"Plan received. You have {self.config.max_edit_turns} edit turns remaining. "
                "Proceed with EDIT when ready."
            )

        elif action == AgentAction.EDIT:
            return self._do_edit(payload.get("diffs", ""), result)

        elif action == AgentAction.RUN:
            return self._do_run(payload.get("command", ""))

        elif action == AgentAction.VERIFY:
            return self._do_verify()

        elif action == AgentAction.COMPLETE:
            return "Task marked as complete."

        elif action == AgentAction.ABORT:
            return f"Task aborted: {payload.get('reason', 'No reason given')}"

        return f"Unknown action: {action}"

    def _do_explore(self, query: str) -> str:
        """Search the codebase and return results."""
        if not query:
            return "Please provide a search query."

        hits = self._indexer.search(query, top_k=10)
        if not hits:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}"]
        for h in hits:
            lines.append(f"  {h.file}:{h.line} [{h.source}] {h.snippet[:120]}")

        # Give the model a compact code packet, not just a file list.
        context = self._context_assembler.assemble(
            ContextRequest(task=query, queries=[query]),
            ContextBudget(
                max_tokens=min(self.config.max_tokens_per_turn, 16_000),
                reserve_tokens=2_000,
            ),
        )
        lines.append("")
        lines.append("Assembled context preview:")
        for item in context.items[:5]:
            lines.append(
                f"  {item.file}:{item.start_line}-{item.end_line} "
                f"tokens≈{item.estimated_tokens} reason={item.reason}"
            )
        return "\n".join(lines)

    def _do_read(self, filepath: str) -> str:
        """Read a file and return its contents (truncated if large)."""
        full_path = self.repo_root / filepath
        try:
            content = full_path.read_text()
        except FileNotFoundError:
            return f"File not found: {filepath}"
        except Exception as exc:
            return f"Error reading {filepath}: {exc}"

        # Truncate if too large
        lines = content.split("\n")
        if len(lines) > 500:
            head = lines[:200]
            tail = lines[-200:]
            return (
                f"File: {filepath} ({len(lines)} lines, showing first 200 + last 200)\n\n"
                + "\n".join(head)
                + f"\n\n... ({len(lines) - 400} lines omitted) ...\n\n"
                + "\n".join(tail)
            )

        return f"File: {filepath} ({len(lines)} lines)\n\n{content}"

    def _do_edit(self, diffs_text: str, result: AgentResult) -> str:
        """Parse and apply diffs from model output."""
        if not diffs_text:
            return "No diffs provided. Please use the EDIT action with unified diff format."

        # Parse diffs
        try:
            edits = DiffParser.parse(diffs_text)
        except Exception as exc:
            return (
                f"Failed to parse diffs: {exc}\n\n"
                "Please use standard unified diff format:\n"
                "--- a/path/to/file\n"
                "+++ b/path/to/file\n"
                "@@ -line,count +line,count @@\n"
                " context\n"
                "-removed\n"
                "+added"
            )

        if not edits:
            # Try generating a diff from model output (model gave old+new, not diff)
            edits = self._try_generate_diffs(diffs_text)

        if not edits:
            return "Could not parse any file edits from the provided diffs."

        # Order edits and checkpoint only the files this edit will touch.
        multi = self._edit_orchestrator.order_file_edits(edits)
        changed_files = multi.affected_files
        checkpoint = self._checkpoint.create(
            changed_files, description="ComposerAgent pre-edit checkpoint"
        )
        apply_result = self._diff_applier.apply_multi_file_edit(multi, dry_run=False)

        if apply_result.success:
            self._diff_applier.commit()
            result.files_changed.extend(
                f for f in apply_result.files_changed if f not in result.files_changed
            )

            targeted_commands = self._edit_orchestrator.affected_test_commands(
                apply_result.files_changed
            )
            verification = self._verifier.verify(
                apply_result.files_changed,
                commands=targeted_commands,
                run_lint=self.config.auto_lint,
                run_tests=self.config.auto_test,
                test_command=self.config.test_command,
                lint_command=self.config.lint_command,
            )
            verify_output = "\n" + verification.render(max_output_chars=1200)
            if not verification.success:
                verify_output += (
                    f"\nCheckpoint available for rollback: {checkpoint.checkpoint_id}. "
                    "Edits were left in place so the agent can inspect and repair."
                )
            else:
                verify_output += f"\nCheckpoint retained: {checkpoint.checkpoint_id}"

            return f"Applied edits to: {', '.join(apply_result.files_changed)}{verify_output}"
        else:
            self._checkpoint.restore(checkpoint.checkpoint_id)
            return (
                f"Failed to apply edits: {'; '.join(apply_result.errors)}\n"
                f"Restored checkpoint: {checkpoint.checkpoint_id}"
            )

    def _try_generate_diffs(self, text: str) -> list[FileEdit]:
        """Try to extract file edits when model output is not a standard diff.

        This handles cases where the model outputs "old code" vs "new code"
        with file paths, but not in unified diff format.
        """
        edits: list[FileEdit] = []

        # Pattern: "File: path/to/file.py" followed by code blocks
        file_pattern = re.compile(
            r"(?:File|file|FILE):\s*([^\s\n]+(?:\.py|\.js|\.ts|\.rs|\.go|\.java)[^\n]*)",
            re.IGNORECASE,
        )
        matches = file_pattern.findall(text)
        if not matches:
            # Try ``` blocks with file annotations
            block_pattern = re.compile(
                r"```(?:python|javascript|typescript|rust)?\s*(?:#|//)\s*(?:File|file):\s*([^\n]+)\n(.*?)```",
                re.DOTALL,
            )
            for m in block_pattern.finditer(text):
                filepath = m.group(1).strip()
                new_content = m.group(2).strip()
                full_path = self.repo_root / filepath
                try:
                    old_content = full_path.read_text()
                except FileNotFoundError:
                    old_content = ""
                edit = DiffGenerator.diff_content(filepath, old_content, new_content)
                if edit.hunks:
                    edits.append(edit)

        return edits

    def _do_run(self, command: str) -> str:
        """Execute a command in the sandbox."""
        if not command:
            return "Please provide a command to run."

        result = self._sandbox.run(command)
        parts = [
            f"Command: {command}",
            f"Exit code: {result.exit_code}",
            f"Wall time: {result.wall_time_seconds:.1f}s",
        ]
        if result.was_timeout:
            parts.append("TIMEOUT")
        if result.stdout:
            parts.append(f"\n[stdout]\n{result.stdout[-3000:]}")
        if result.stderr:
            parts.append(f"\n[stderr]\n{result.stderr[-2000:]}")
        return "\n".join(parts)

    def _do_verify(self) -> str:
        """Run verification (tests + lint)."""
        return self._do_run(self.config.test_command)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def turns(self) -> list[AgentTurn]:
        return list(self._turns)

    @property
    def indexer(self) -> CodebaseIndexer:
        return self._indexer

    @property
    def sandbox(self) -> TerminalSandbox:
        return self._sandbox
