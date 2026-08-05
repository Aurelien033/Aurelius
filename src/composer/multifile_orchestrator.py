"""
MultiFileEditOrchestrator — dependency-aware planning for repository edits.

The first ComposerAgent could apply multi-file diffs, but it did not reason
about edit ordering, dependency blast radius, or verification scope. This module
creates an explicit edit plan from candidate files and the indexer's import
metadata.

This is deliberately model-agnostic: a language model can propose candidate
files, but the ordering and risk accounting are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .codebase_indexer import CodebaseIndexer
from .diff_engine import FileEdit, MultiFileEdit


@dataclass
class EditTarget:
    """A file selected for possible editing."""

    file: str
    reason: str
    priority: float = 0.5
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    risk: str = "medium"  # low / medium / high
    suggested_verifiers: list[str] = field(default_factory=list)


@dataclass
class EditPlan:
    """Ordered multi-file edit plan."""

    task: str
    targets: list[EditTarget]
    ordered_files: list[str]
    risk_summary: dict[str, int]
    verification_commands: list[str]

    def render(self) -> str:
        lines = [f"# Edit plan: {self.task}", ""]
        lines.append("## Ordered files")
        for i, filepath in enumerate(self.ordered_files, 1):
            target = next((t for t in self.targets if t.file == filepath), None)
            reason = target.reason if target else "selected"
            risk = target.risk if target else "medium"
            lines.append(f"{i}. {filepath} — risk={risk}; {reason}")
        lines.append("")
        lines.append("## Verification")
        for cmd in self.verification_commands:
            lines.append(f"- `{cmd}`")
        lines.append("")
        lines.append("## Risk summary")
        for k, v in sorted(self.risk_summary.items()):
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


class MultiFileEditOrchestrator:
    """Plan and order edits across related files."""

    def __init__(self, repo_root: Path, indexer: CodebaseIndexer):
        self.repo_root = Path(repo_root).resolve()
        self.indexer = indexer

    def plan(self, task: str, candidate_files: list[str]) -> EditPlan:
        """Create a deterministic edit plan from candidate files.

        Ordering rule:
          1. Source/library files before tests/docs
          2. Lower-level dependencies before importers
          3. High priority before low priority
          4. Smaller blast radius before larger blast radius when tied
        """
        if self.indexer.stats.total_files == 0:
            self.indexer.build()

        targets: list[EditTarget] = []
        for filepath in self._dedupe_existing(candidate_files):
            entry = self.indexer.get_entry(filepath)
            imports = entry.imports if entry else []
            imported_by = self._files_importing_file(filepath)
            risk = self._risk_for_file(filepath, imported_by)
            verifiers = self._verifiers_for_file(filepath)
            priority = self._priority_for_file(filepath, imported_by)
            targets.append(
                EditTarget(
                    file=filepath,
                    reason=self._reason_for_file(filepath, imported_by),
                    priority=priority,
                    imports=imports,
                    imported_by=imported_by,
                    risk=risk,
                    suggested_verifiers=verifiers,
                )
            )

        ordered = [t.file for t in sorted(targets, key=self._sort_key)]
        risk_summary = {"low": 0, "medium": 0, "high": 0}
        for target in targets:
            risk_summary[target.risk] = risk_summary.get(target.risk, 0) + 1

        commands = self._merge_verifiers(targets)
        return EditPlan(
            task=task,
            targets=targets,
            ordered_files=ordered,
            risk_summary=risk_summary,
            verification_commands=commands,
        )

    def order_file_edits(self, edits: list[FileEdit]) -> MultiFileEdit:
        """Order concrete FileEdits using the same dependency heuristic."""
        plan = self.plan("apply concrete edits", [e.filepath for e in edits])
        by_file = {e.filepath: e for e in edits}
        ordered_edits = [by_file[f] for f in plan.ordered_files if f in by_file]
        return MultiFileEdit(
            file_edits=ordered_edits, description="dependency-ordered multi-file edit"
        )

    def affected_test_commands(self, files: list[str]) -> list[str]:
        """Return test commands likely to cover changed files."""
        plan = self.plan("verification planning", files)
        return plan.verification_commands

    def _dedupe_existing(self, files: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for f in files:
            clean = str(Path(f))
            if clean in seen:
                continue
            if (
                (self.repo_root / clean).exists()
                or clean.startswith("tests/")
                or clean.startswith("src/")
            ):
                result.append(clean)
                seen.add(clean)
        return result

    def _files_importing_file(self, filepath: str) -> list[str]:
        module_guess = self._module_name_for_file(filepath)
        if not module_guess:
            return []
        importers = set(self.indexer.files_importing(module_guess))
        # Also check stem-only imports; common in local modules.
        importers.update(self.indexer.files_importing(Path(filepath).stem))
        importers.discard(filepath)
        return sorted(importers)

    def _module_name_for_file(self, filepath: str) -> str:
        path = Path(filepath)
        if path.suffix != ".py":
            return path.stem
        parts = list(path.with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        return ".".join(parts)

    def _risk_for_file(self, filepath: str, imported_by: list[str]) -> str:
        if filepath.endswith(("pyproject.toml", "Cargo.toml", "package.json")):
            return "high"
        if filepath.startswith("src/") and len(imported_by) >= 8:
            return "high"
        if filepath.startswith("tests/") or filepath.startswith("docs/"):
            return "low"
        if len(imported_by) >= 3:
            return "medium"
        return "low"

    def _priority_for_file(self, filepath: str, imported_by: list[str]) -> float:
        score = 0.5
        if filepath.startswith("src/"):
            score += 0.25
        if filepath.startswith("tests/"):
            score -= 0.10
        if filepath.startswith("docs/"):
            score -= 0.20
        score += min(0.20, len(imported_by) * 0.02)
        if Path(filepath).name in {"__init__.py", "pyproject.toml", "Cargo.toml"}:
            score += 0.05
        return score

    def _reason_for_file(self, filepath: str, imported_by: list[str]) -> str:
        if filepath.startswith("tests/"):
            return "test coverage / regression gate"
        if filepath.startswith("docs/"):
            return "documentation surface"
        if imported_by:
            return f"source module imported by {len(imported_by)} files"
        return "source or support file"

    def _verifiers_for_file(self, filepath: str) -> list[str]:
        path = Path(filepath)
        if path.suffix == ".py":
            commands = ["python3 -m py_compile " + filepath]
            if filepath.startswith("src/"):
                candidate = Path("tests") / path.relative_to("src").parent / f"test_{path.stem}.py"
                if (self.repo_root / candidate).is_file():
                    commands.append(f"python3 -m pytest {candidate} -q")
            if filepath.startswith("tests/"):
                commands.append(f"python3 -m pytest {filepath} -q")
            return commands
        if path.suffix == ".rs":
            return ["cargo test --quiet"]
        if path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
            return ["npm test -- --runInBand"]
        return []

    def _merge_verifiers(self, targets: list[EditTarget]) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()
        for target in targets:
            for cmd in target.suggested_verifiers:
                if cmd not in seen:
                    commands.append(cmd)
                    seen.add(cmd)
        # If no targeted tests exist, compile Python files and leave full-suite as optional.
        if not commands:
            py_files = [t.file for t in targets if t.file.endswith(".py")]
            if py_files:
                commands.extend(f"python3 -m py_compile {f}" for f in py_files[:10])
        return commands

    def _sort_key(self, target: EditTarget) -> tuple[int, int, float, int, str]:
        is_test_or_doc = 1 if target.file.startswith(("tests/", "docs/")) else 0
        is_config = (
            1 if Path(target.file).name in {"pyproject.toml", "Cargo.toml", "package.json"} else 0
        )
        blast_radius = len(target.imported_by)
        return (is_test_or_doc, is_config, -target.priority, blast_radius, target.file)
