"""Skill discovery, catalog and attachment helpers for the shell."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
import copy

from agent.skill_catalog import SkillCatalog
from src.model.interface_framework import SkillBundle
from src.model.interface_framework import TaskThread
from src.ui.shell.models import AureliusShellError
from src.ui.shell.models import SkillRecord

__all__ = [
    "ShellSkillsMixin",
    "_DEFAULT_SKILL_ROOT_NAMES",
    "_normalize_skill_scope",
    "_parse_skill_markdown",
]


class ShellSkillsMixin:
    """Local-first skill catalog surface exposed by the shell."""

    def discover_skills(
        self,
        roots: Sequence[str | Path] | None = None,
    ) -> tuple[SkillRecord, ...]:
        candidate_roots = self._skill_roots(roots)
        discovered: dict[str, SkillRecord] = {}
        for root in candidate_roots:
            if not root.exists():
                continue
            skill_files = self._skill_files_under(root)
            for skill_file in skill_files:
                record = self._build_skill_record(root, skill_file)
                discovered.setdefault(record.skill_id, record)
        return tuple(sorted(discovered.values(), key=lambda record: record.skill_id))

    def attach_skill_ids(
        self,
        thread: str | TaskThread,
        skill_ids: Sequence[str],
        *,
        roots: Sequence[str | Path] | None = None,
    ) -> TaskThread:
        catalog = {record.skill_id: record for record in self.discover_skills(roots)}
        bundles: list[SkillBundle] = []
        for skill_id in skill_ids:
            if skill_id in catalog:
                bundles.append(self._bundle_from_skill_record(catalog[skill_id]))
            else:
                bundles.append(SkillBundle(skill_id=skill_id, name=skill_id))
        return self.attach_skills(thread, bundles)

    def catalog_skill_summary(self) -> dict[str, Any]:
        """Return the local-first skill catalog provenance summary."""
        return self._build_skill_catalog().provenance_summary()

    def catalog_skill_show(self, skill_id: str) -> dict[str, Any]:
        """Return one catalog skill record as a JSON-safe mapping."""
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise AureliusShellError("skill_id must be a non-empty string")
        entry = self._build_skill_catalog().get(skill_id)
        if entry is None:
            raise AureliusShellError(f"unknown skill: {skill_id!r}")
        return asdict(entry)

    def catalog_skill_search(self, query: str) -> list[dict[str, Any]]:
        """Search the local-first skill catalog."""
        if not isinstance(query, str) or not query.strip():
            raise AureliusShellError("query must be a non-empty string")
        return [asdict(entry) for entry in self._build_skill_catalog().search(query)]

    def _build_skill_catalog(self) -> SkillCatalog:
        return SkillCatalog(self.framework.paths.repo_root)

    def _normalize_skill_inputs(
        self,
        skill_inputs: Sequence[str | SkillBundle | SkillRecord],
    ) -> tuple[SkillBundle, ...]:
        normalized: list[SkillBundle] = []
        seen: set[str] = set()
        for item in skill_inputs:
            if isinstance(item, SkillBundle):
                bundle = item
            elif isinstance(item, SkillRecord):
                bundle = self._bundle_from_skill_record(item)
            elif isinstance(item, str) and item.strip():
                bundle = SkillBundle(skill_id=item, name=item)
            else:
                raise AureliusShellError(
                    "skill inputs must be strings, SkillBundle instances, or SkillRecord instances"
                )
            if bundle.skill_id in seen:
                continue
            seen.add(bundle.skill_id)
            normalized.append(bundle)
        return tuple(normalized)

    def _bundle_from_skill_record(self, record: SkillRecord) -> SkillBundle:
        return SkillBundle(
            skill_id=record.skill_id,
            name=record.name,
            scope=_normalize_skill_scope(record.scope),
            provenance=record.provenance,
            source_path=record.source_path,
            metadata=copy.deepcopy(record.metadata),
        )

    def _skill_roots(self, roots: Sequence[str | Path] | None) -> tuple[Path, ...]:
        if roots is not None:
            return tuple(Path(root).expanduser().resolve() for root in roots)
        repo_root = self.framework.paths.repo_root
        candidate_roots = [repo_root / name for name in _DEFAULT_SKILL_ROOT_NAMES]
        candidate_roots.extend(
            [
                Path.home() / ".codex" / "skills",
                Path.home() / ".agents" / "skills",
            ]
        )
        return tuple(dict.fromkeys(path.resolve() for path in candidate_roots))

    def _skill_files_under(self, root: Path) -> tuple[Path, ...]:
        if root.is_file():
            return (root,) if root.name == "SKILL.md" else ()
        if not root.is_dir():
            return ()
        return tuple(sorted(path for path in root.rglob("SKILL.md") if path.is_file()))

    def _build_skill_record(self, root: Path, skill_file: Path) -> SkillRecord:
        try:
            text = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise AureliusShellError(f"failed to read skill bundle at {skill_file}: {exc}") from exc
        if not text.strip():
            raise AureliusShellError(f"skill bundle is empty: {skill_file}")
        skill_id = self._skill_id_for_file(root, skill_file)
        name, summary = _parse_skill_markdown(text, skill_id)
        provenance = (
            "repo-local"
            if skill_file.resolve().is_relative_to(self.framework.paths.repo_root)
            else "global"
        )
        scope = "repo" if provenance == "repo-local" else "global"
        return SkillRecord(
            skill_id=skill_id,
            name=name,
            source_path=str(skill_file),
            provenance=provenance,
            summary=summary,
            scope=scope,
            metadata={
                "root": str(root),
                "bundle_path": str(skill_file),
            },
        )

    def _skill_id_for_file(self, root: Path, skill_file: Path) -> str:
        parent = skill_file.parent
        try:
            rel = parent.resolve().relative_to(root.resolve())
        except ValueError:
            rel = parent.name
        if isinstance(rel, Path):
            parts = rel.parts
        elif isinstance(rel, str):
            parts = (rel,)
        else:
            parts = ()
        if parts:
            return "/".join(parts)
        return parent.name


_DEFAULT_SKILL_ROOT_NAMES = (
    "skills",
    ".codex/skills",
    ".agents/skills",
)


def _normalize_skill_scope(scope: str) -> str:
    normalized = str(scope or "thread").strip().lower()
    if normalized in {"workspace", "local"}:
        return "repo"
    if normalized in {"global", "org", "repo", "thread"}:
        return normalized
    return "thread"


def _parse_skill_markdown(text: str, skill_id: str) -> tuple[str, str]:
    title = ""
    summary_lines: list[str] = []
    in_summary = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_summary:
                break
            continue
        if not title and line.startswith("#"):
            title = line.lstrip("#").strip()
            continue
        if title and not in_summary:
            in_summary = True
            summary_lines.append(line)
        elif in_summary:
            summary_lines.append(line)
    if not title:
        title = skill_id.replace("/", " ").replace("-", " ").title()
    summary = " ".join(summary_lines).strip()
    return title, summary
