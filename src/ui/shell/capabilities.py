"""Capability, surface-catalog and journal introspection for the shell."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent.surface_catalog import describe_ui_surface

__all__ = [
    "ShellCapabilitiesMixin",
]


class ShellCapabilitiesMixin:
    """Introspection surface of the shell."""

    def instruction_layers(
        self,
        *,
        workspace: str | Path | None = None,
        skill_ids: Sequence[str] = (),
        mode_name: str | None = None,
        memory_summary: str | None = None,
    ) -> tuple[str, ...]:
        """Expose the active instruction layering model for terminal consumers."""
        return self._build_skill_catalog().instruction_layers_for(
            workspace=workspace,
            repo_root=self.framework.paths.repo_root,
            skill_ids=skill_ids,
            mode_name=mode_name,
            memory_summary=memory_summary,
        )

    def journal_branch_summary(self, branch_id: str = "main") -> dict[str, Any]:
        """Return a persisted journal branch summary for the shell session."""
        runtime = self._build_runtime()
        if runtime.session_manager.get_session(self.session_id) is None:
            return {
                "branch_id": branch_id,
                "name": branch_id,
                "base_entry_id": None,
                "head_entry_id": None,
                "entry_count": 0,
                "latest_entry_id": None,
                "latest_entry_kind": None,
                "compaction_count": 0,
                "latest_compaction_id": None,
                "metadata": {},
            }
        return runtime.journal_branch_summary(self.session_id, branch_id)

    def journal_compaction_summary(
        self,
        *,
        branch_id: str | None = None,
        compaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a persisted journal compaction summary for the shell session."""
        runtime = self._build_runtime()
        if runtime.session_manager.get_session(self.session_id) is None:
            return {
                "compaction_id": None,
                "branch_id": branch_id,
                "policy": None,
                "keep_last_n": 0,
                "dropped_count": 0,
                "retained_count": 0,
                "summary_entry_id": None,
                "facts_count": 0,
                "summary_text": "",
                "created_at": None,
                "metadata": {},
            }
        return runtime.journal_compaction_summary(
            self.session_id,
            compaction_id=compaction_id,
            branch_id=branch_id,
        )

    def capability_summary(self) -> dict[str, Any]:
        """Return a runtime-backed capability summary for the shell session."""
        return self._build_runtime().capability_summary(self.session_id)

    def capability_summary_schema(self) -> dict[str, Any]:
        """Return the versioned schema for capability summaries."""
        return self._build_runtime().capability_summary_schema()

    def surface_catalog(self) -> dict[str, Any]:
        """Return a runtime-backed surface catalog with UI coverage added."""
        catalog = self._build_runtime().surface_catalog()
        catalog["ui"] = describe_ui_surface()
        return catalog

    def surface_catalog_schema(self) -> dict[str, Any]:
        """Return the versioned schema for surface catalogs."""
        return self._build_runtime().surface_catalog_schema()
