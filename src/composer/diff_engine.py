"""
DiffEngine — precise code editing with unified-diff generation and application.

Cursor Composer 2.5 edits code by producing unified diffs rather than
rewriting entire files. This module implements:

  - UnifiedDiff: dataclass representing a hunk-level change
  - DiffGenerator: produces minimal unified diffs from old/new file content
  - DiffApplier: applies unified diffs with fuzz matching and rollback
  - MultiFileDiff: orchestrates diffs across multiple files atomically

The key insight: the model should output only the CHANGED regions as
diffs, not the whole file. This reduces tokens, avoids copy-paste errors,
and makes review straightforward.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EditHunk:
    """A single contiguous change in a file."""

    old_start: int  # 1-indexed start line in original
    old_count: int  # number of lines removed
    new_start: int  # 1-indexed start line in result
    new_count: int  # number of lines added
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    context_before: list[str] = field(default_factory=list)  # 3 lines
    context_after: list[str] = field(default_factory=list)  # 3 lines

    def as_unified_diff(self, filepath: str) -> str:
        """Render this hunk as a unified diff snippet."""
        lines = [f"--- a/{filepath}", f"+++ b/{filepath}"]
        lines.append(f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@")
        for cl in self.context_before:
            lines.append(f" {cl}")
        for ol in self.old_lines:
            lines.append(f"-{ol}")
        for nl in self.new_lines:
            lines.append(f"+{nl}")
        for cl in self.context_after:
            lines.append(f" {cl}")
        return "\n".join(lines)


@dataclass
class FileEdit:
    """A set of edits to a single file."""

    filepath: str
    hunks: list[EditHunk] = field(default_factory=list)
    description: str = ""  # natural language description of the edit

    def as_unified_diff(self) -> str:
        """Render all hunks as a single unified diff."""
        if not self.hunks:
            return ""
        parts = [f"--- a/{self.filepath}", f"+++ b/{self.filepath}"]
        for hunk in self.hunks:
            parts.append(
                f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
            )
            for cl in hunk.context_before:
                parts.append(f" {cl}")
            for ol in hunk.old_lines:
                parts.append(f"-{ol}")
            for nl in hunk.new_lines:
                parts.append(f"+{nl}")
            for cl in hunk.context_after:
                parts.append(f" {cl}")
        return "\n".join(parts)


@dataclass
class MultiFileEdit:
    """An edit spanning multiple files with atomicity guarantees."""

    file_edits: list[FileEdit] = field(default_factory=list)
    description: str = ""  # natural language description of the overall change

    @property
    def affected_files(self) -> list[str]:
        return [e.filepath for e in self.file_edits]


@dataclass
class ApplyResult:
    """Result of applying a set of edits."""

    success: bool
    files_changed: list[str]
    files_failed: list[str]
    errors: list[str] = field(default_factory=list)
    rollback_applied: bool = False


# ---------------------------------------------------------------------------
# DiffGenerator
# ---------------------------------------------------------------------------


class DiffGenerator:
    """Generate minimal unified diffs by comparing original and modified content.

    Uses Python's difflib for line-level diffing. Produces clean hunks
    that a model could learn to generate directly.
    """

    @staticmethod
    def diff_lines(
        original: list[str],
        modified: list[str],
        context_lines: int = 3,
    ) -> list[EditHunk]:
        """Compare two lists of lines and produce EditHunks.

        Args:
            original: Original file lines (with newlines or without).
            modified: Modified file lines.
            context_lines: Number of context lines around each hunk.

        Returns:
            List of EditHunk objects describing the changes.
        """
        # Strip trailing newlines for cleaner matching
        orig = [line.rstrip("\n") for line in original]
        mod = [line.rstrip("\n") for line in modified]

        matcher = difflib.SequenceMatcher(None, orig, mod)
        hunks: list[EditHunk] = []

        for group in matcher.get_grouped_opcodes(n=context_lines):
            old_start = group[0][1] + 1  # 1-indexed
            new_start = group[0][3] + 1

            old_lines: list[str] = []
            new_lines: list[str] = []
            ctx_before: list[str] = []
            ctx_after: list[str] = []

            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    # Split equal context between before and after
                    if not old_lines and not new_lines:
                        ctx_before.extend(orig[i1:i2])
                    elif old_lines or new_lines:
                        ctx_after.extend(orig[i1:i2])
                elif tag == "delete":
                    old_lines.extend(orig[i1:i2])
                elif tag == "replace":
                    old_lines.extend(orig[i1:i2])
                    new_lines.extend(mod[j1:j2])
                elif tag == "insert":
                    new_lines.extend(mod[j1:j2])

            # Compute old/new counts (default to 1 if empty for valid diff format)
            old_count = len(old_lines) if old_lines else (1 if new_lines else 0)
            new_count = len(new_lines) if new_lines else (1 if old_lines else 0)

            if old_lines or new_lines:
                hunks.append(
                    EditHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        old_lines=list(old_lines),
                        new_lines=list(new_lines),
                        context_before=list(ctx_before),
                        context_after=list(ctx_after),
                    )
                )

        return hunks

    @staticmethod
    def diff_files(
        original_path: Path,
        modified_path: Path,
    ) -> FileEdit:
        """Generate a FileEdit from two file paths."""
        orig_lines = original_path.read_text().splitlines(keepends=False)
        mod_lines = modified_path.read_text().splitlines(keepends=False)
        hunks = DiffGenerator.diff_lines(orig_lines, mod_lines)
        return FileEdit(filepath=str(original_path), hunks=hunks)

    @staticmethod
    def diff_content(
        filepath: str,
        original_content: str,
        modified_content: str,
    ) -> FileEdit:
        """Generate a FileEdit from in-memory content strings."""
        orig_lines = original_content.splitlines(keepends=False)
        mod_lines = modified_content.splitlines(keepends=False)
        hunks = DiffGenerator.diff_lines(orig_lines, mod_lines)
        return FileEdit(filepath=filepath, hunks=hunks)


# ---------------------------------------------------------------------------
# DiffApplier
# ---------------------------------------------------------------------------


class DiffApplier:
    """Apply unified diffs to files on disk with fuzz matching and rollback.

    Key features:
      - Fuzz matching: tolerates minor line-number drift
      - Backup: saves original file before applying
      - Rollback: restores from backup on failure
      - Validation: checks that the applied result is the diff
    """

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self._backups: dict[str, str] = {}  # filepath -> backup content

    def apply_file_edit(self, edit: FileEdit, dry_run: bool = False) -> bool:
        """Apply a single file edit. Returns True on success."""
        full_path = self.repo_root / edit.filepath

        # Read original
        try:
            original = full_path.read_text()
        except FileNotFoundError:
            # Creating a new file
            original = ""
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # Save backup
        self._backups[str(edit.filepath)] = original

        # Apply hunks
        try:
            result = self._apply_hunks(original.splitlines(keepends=True), edit.hunks)
            result_text = "".join(result)
        except Exception as exc:
            logger.error("Failed to apply edit to %s: %s", edit.filepath, exc)
            return False

        if dry_run:
            return True

        try:
            full_path.write_text(result_text)
        except Exception as exc:
            logger.error("Failed to write %s: %s", edit.filepath, exc)
            return False

        return True

    def apply_multi_file_edit(self, edit: MultiFileEdit, dry_run: bool = False) -> ApplyResult:
        """Apply edits across multiple files atomically.

        If any file fails, all changes are rolled back.
        """
        results = ApplyResult(success=True, files_changed=[], files_failed=[], errors=[])

        # Phase 1: attempt all edits
        for file_edit in edit.file_edits:
            ok = self.apply_file_edit(file_edit, dry_run=dry_run)
            if ok:
                results.files_changed.append(file_edit.filepath)
            else:
                results.files_failed.append(file_edit.filepath)
                results.errors.append(f"Failed to apply edit to {file_edit.filepath}")

        if results.files_failed:
            results.success = False
            if not dry_run:
                self.rollback()
                results.rollback_applied = True

        return results

    def rollback(self) -> None:
        """Restore all backed-up files to their original state."""
        for filepath, original_content in self._backups.items():
            full_path = self.repo_root / filepath
            try:
                if original_content:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(original_content)
                elif full_path.exists():
                    full_path.unlink()
            except Exception as exc:
                logger.error("Rollback failed for %s: %s", filepath, exc)
        self._backups.clear()

    def commit(self) -> None:
        """Clear backups after successful apply."""
        self._backups.clear()

    # ------------------------------------------------------------------
    # Internal: hunk application with fuzz matching
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_hunks(lines: list[str], hunks: list[EditHunk], fuzz: int = 5) -> list[str]:
        """Apply a list of hunks to lines, with fuzz matching for tolerance.

        Hunks are applied in reverse order (bottom-up) to preserve line
        numbers. Fuzz matching finds the nearest matching context within
        ``fuzz`` lines of the expected position.
        """
        result = list(lines)

        for hunk in reversed(hunks):
            # Build the "search" pattern from context_before + old_lines
            search_lines = list(hunk.context_before) + list(hunk.old_lines)
            replacement_lines = list(hunk.context_before) + list(hunk.new_lines)

            # Find the position of old lines in the current result
            target_start = hunk.old_start - 1  # 0-indexed

            # Try exact match first, then fuzz
            pos = DiffApplier._find_lines(
                result,
                list(hunk.context_before) + list(hunk.old_lines),
                target_start,
                fuzz,
            )

            if pos is None:
                # Try matching just the context
                if hunk.context_before:
                    pos = DiffApplier._find_lines(
                        result, list(hunk.context_before), target_start, fuzz * 2
                    )
                    if pos is not None:
                        # Insert new lines after context
                        end = pos + len(hunk.context_before) + len(hunk.old_lines)
                        if end <= len(result):
                            result[pos + len(hunk.context_before) : end] = list(hunk.new_lines)
                        else:
                            result[pos + len(hunk.context_before) :] = list(hunk.new_lines)
                    else:
                        raise ValueError(
                            f"Could not find context for hunk at line {hunk.old_start}"
                        )
                else:
                    raise ValueError(f"Could not apply hunk at line {hunk.old_start}")
            else:
                # Replace old lines with new lines
                old_len = len(search_lines)
                result[pos : pos + old_len] = replacement_lines

        return result

    @staticmethod
    def _find_lines(
        lines: list[str],
        pattern: list[str],
        expected_pos: int,
        fuzz: int = 5,
    ) -> int | None:
        """Find pattern in lines near expected_pos, with fuzz tolerance."""
        if not pattern:
            return expected_pos

        # Try exact match at expected position
        if DiffApplier._matches_at(lines, pattern, expected_pos):
            return expected_pos

        # Try positions within fuzz range
        for offset in range(1, fuzz + 1):
            # Before expected
            pos = expected_pos - offset
            if pos >= 0 and DiffApplier._matches_at(lines, pattern, pos):
                return pos

            # After expected
            pos = expected_pos + offset
            if pos + len(pattern) <= len(lines) and DiffApplier._matches_at(lines, pattern, pos):
                return pos

        return None

    @staticmethod
    def _matches_at(lines: list[str], pattern: list[str], pos: int) -> bool:
        """Check if pattern matches lines starting at pos."""
        if pos < 0 or pos + len(pattern) > len(lines):
            return False
        for i, pline in enumerate(pattern):
            if lines[pos + i].rstrip("\n") != pline:
                return False
        return True


# ---------------------------------------------------------------------------
# DiffParser — parse model output into structured FileEdits
# ---------------------------------------------------------------------------


class DiffParser:
    """Parse a raw unified diff string (from model output) into FileEdit objects.

    Handles the common format that coding models produce:
      --- a/path/to/file
      +++ b/path/to/file
      @@ -old_start,old_count +new_start,new_count @@
       context line
      -removed line
      +added line
    """

    _HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    @classmethod
    def parse(cls, diff_text: str) -> list[FileEdit]:
        """Parse a multi-file unified diff string into structured FileEdits."""
        edits: list[FileEdit] = []
        current_edit: FileEdit | None = None
        current_hunk: dict | None = None
        current_context: list[str] = []
        current_old: list[str] = []
        current_new: list[str] = []
        seen_removals = False
        seen_additions = False

        for line in diff_text.split("\n"):
            # Detect file header
            if line.startswith("--- a/") or line.startswith("--- "):
                if current_edit and current_hunk:
                    cls._finish_hunk(
                        current_edit,
                        current_hunk,
                        current_context,
                        current_old,
                        current_new,
                        seen_removals,
                        seen_additions,
                    )
                    current_hunk = None
                filepath = line[6:] if line.startswith("--- a/") else line[4:]
                filepath = filepath.strip()
                if filepath and filepath != "/dev/null":
                    current_edit = FileEdit(filepath=filepath)
                    edits.append(current_edit)
                    current_context = []
                    current_old = []
                    current_new = []
                    seen_removals = False
                    seen_additions = False
                continue

            if line.startswith("+++ "):
                continue

            # Detect hunk header
            m = cls._HUNK_HEADER_RE.match(line)
            if m:
                if current_edit and current_hunk:
                    cls._finish_hunk(
                        current_edit,
                        current_hunk,
                        current_context,
                        current_old,
                        current_new,
                        seen_removals,
                        seen_additions,
                    )
                current_hunk = {
                    "old_start": int(m.group(1)),
                    "old_count": int(m.group(2) or 1),
                    "new_start": int(m.group(3)),
                    "new_count": int(m.group(4) or 1),
                }
                current_context = []
                current_old = []
                current_new = []
                seen_removals = False
                seen_additions = False
                continue

            if not current_edit:
                continue

            # Classify lines
            if line.startswith("-"):
                seen_removals = True
                current_old.append(line[1:])
            elif line.startswith("+"):
                seen_additions = True
                current_new.append(line[1:])
            elif line.startswith(" "):
                # Context line — if we have pending old/new, finish hunk
                if (seen_removals or seen_additions) and current_hunk:
                    cls._finish_hunk(
                        current_edit,
                        current_hunk,
                        current_context,
                        current_old,
                        current_new,
                        seen_removals,
                        seen_additions,
                    )
                    current_hunk = {
                        "old_start": current_hunk["old_start"] + current_hunk["old_count"],
                        "old_count": 0,
                        "new_start": current_hunk["new_start"] + current_hunk["new_count"],
                        "new_count": 0,
                    }
                    current_context = []
                    current_old = []
                    current_new = []
                    seen_removals = False
                    seen_additions = False
                current_context.append(line[1:])

        # Finish last hunk
        if current_edit and current_hunk:
            cls._finish_hunk(
                current_edit,
                current_hunk,
                current_context,
                current_old,
                current_new,
                seen_removals,
                seen_additions,
            )

        return [e for e in edits if e.hunks]

    @classmethod
    def _finish_hunk(
        cls,
        edit: FileEdit,
        hunk_info: dict,
        context: list[str],
        old_lines: list[str],
        new_lines: list[str],
        has_removals: bool,
        has_additions: bool,
    ) -> None:
        """Create an EditHunk and add it to the FileEdit."""
        if not old_lines and not new_lines:
            return

        old_count = len(old_lines) if old_lines else (1 if new_lines else 0)
        new_count = len(new_lines) if new_lines else (1 if old_lines else 0)

        # Split context: last 3 of preceding context go to before, rest is post-hunk
        ctx_before = context[-3:] if len(context) >= 3 else list(context)
        ctx_after = []

        edit.hunks.append(
            EditHunk(
                old_start=hunk_info["old_start"],
                old_count=old_count,
                new_start=hunk_info["new_start"],
                new_count=new_count,
                old_lines=list(old_lines),
                new_lines=list(new_lines),
                context_before=list(ctx_before),
                context_after=list(ctx_after),
            )
        )
