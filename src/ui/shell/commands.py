"""Command-line parsing for the Aurelius shell."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import json
import shlex

from src.ui.shell.models import AureliusShellError
from src.ui.shell.render import _backend_record

__all__ = [
    "ShellCommandsMixin",
    "_normalize_decision",
]


class ShellCommandsMixin:
    """Slash-command surface of the shell."""

    def execute_command(self, command_line: str) -> str:
        """Execute a tiny shell command and return the textual result."""
        if not isinstance(command_line, str) or not command_line.strip():
            raise AureliusShellError("command_line must be a non-empty string")
        argv = shlex.split(command_line)
        if not argv:
            raise AureliusShellError("command_line produced no tokens")
        command = argv[0]
        if command == "status":
            return self.render_status()
        if command == "mode" and len(argv) >= 3 and argv[1] == "set":
            policy = self.set_mode(argv[2])
            return json.dumps(
                {
                    "current_mode": self.current_mode,
                    "policy": asdict(policy),
                },
                sort_keys=True,
            )
        if command == "backend" and len(argv) >= 2 and argv[1] == "list":
            return json.dumps(self._describe_backends(), sort_keys=True)
        if command == "backend" and len(argv) >= 3 and argv[1] == "show":
            backend_name = argv[2]
            backends = self._backend_surface()
            try:
                adapter = backends.get_backend(backend_name)
            except Exception as exc:
                raise AureliusShellError(str(exc)) from exc
            return json.dumps(
                {"backend": _backend_record(adapter)},
                sort_keys=True,
            )
        if command == "backend" and len(argv) >= 3 and argv[1] == "engine" and argv[2] == "list":
            return json.dumps(
                {"engine_surface": self.surface_catalog()["engine_adapters"]},
                sort_keys=True,
            )
        if command == "backend" and len(argv) >= 4 and argv[1] == "engine" and argv[2] == "show":
            engine_name = argv[3]
            engine_surface = self.surface_catalog()["engine_adapters"]
            for record in engine_surface["engine_adapters"]:
                if record["backend_name"] == engine_name:
                    return json.dumps({"engine": record}, sort_keys=True)
            raise AureliusShellError(
                f"unknown engine adapter: {engine_name!r}; known: {engine_surface['names']}"
            )
        if command == "skill" and len(argv) >= 2 and argv[1] == "summary":
            return json.dumps(
                {"summary": self.catalog_skill_summary()},
                sort_keys=True,
            )
        if command == "skill" and len(argv) >= 3 and argv[1] == "show":
            return json.dumps(
                {"skill": self.catalog_skill_show(argv[2])},
                sort_keys=True,
            )
        if command == "skill" and len(argv) >= 3 and argv[1] == "search":
            query = " ".join(argv[2:]).strip()
            if not query:
                raise AureliusShellError("skill search requires a non-empty query")
            matches = self.catalog_skill_search(query)
            return json.dumps(
                {
                    "count": len(matches),
                    "skills": matches,
                },
                sort_keys=True,
            )
        if command == "channel" and len(argv) >= 2 and argv[1] == "list":
            return json.dumps(
                {
                    "count": len(self.list_messages()),
                    "messages": list(self.list_messages()),
                },
                sort_keys=True,
            )
        if command == "journal" and len(argv) >= 3 and argv[1] == "branch":
            return json.dumps(
                {"journal": self.journal_branch_summary(argv[2])},
                sort_keys=True,
            )
        if command == "journal" and len(argv) >= 3 and argv[1] == "compaction":
            return json.dumps(
                {"journal": self.journal_compaction_summary(branch_id=argv[2])},
                sort_keys=True,
            )
        if command == "capability" and len(argv) >= 2 and argv[1] == "summary":
            return json.dumps(
                {"capability": self.capability_summary()},
                sort_keys=True,
            )
        if command == "capability" and len(argv) >= 2 and argv[1] == "schema":
            return json.dumps(
                {"schema": self.capability_summary_schema()},
                sort_keys=True,
            )
        if command == "surface" and len(argv) >= 2 and argv[1] == "summary":
            return json.dumps(
                {"surface": self.surface_catalog()},
                sort_keys=True,
            )
        if command == "surface" and len(argv) >= 2 and argv[1] == "schema":
            return json.dumps(
                {"schema": self.surface_catalog_schema()},
                sort_keys=True,
            )
        if command == "thread" and len(argv) >= 3 and argv[1] == "status":
            thread_id = argv[2] if len(argv) > 2 else self.active_thread_id
            if thread_id is None:
                raise AureliusShellError("no active thread selected")
            return self.render_thread_status(thread_id)
        raise AureliusShellError(f"unsupported shell command: {command_line!r}")


def _normalize_decision(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, bool):
        return "allow" if value else "deny"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "allow",
            "allow_once",
            "allow_for_thread",
            "allow_for_scope",
        }:
            return "allow"
        if normalized in {"deny", "block", "blocked"}:
            return "deny"
        if normalized in {"pending", "wait"}:
            return "pending"
    return "pending"
