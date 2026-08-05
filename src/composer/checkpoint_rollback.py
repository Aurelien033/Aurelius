"""
CheckpointRollback — git-aware file-scoped checkpoints for Composer edits.

Using `git stash` blindly in this repository is unsafe because the working tree
may already contain many unrelated untracked and modified files from ongoing
research. This checkpoint manager is therefore git-aware but file-scoped:

  - captures git status and per-file metadata
  - stores exact pre-edit file bytes for only the files Composer will touch
  - restores only those files on rollback
  - never stages, commits, stashes, resets, or rewrites history

This gives the safety property Composer needs without trampling parallel work.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointFile:
    """Metadata for one file captured in a checkpoint."""

    path: str
    existed: bool
    size_bytes: int = 0
    sha256: str = ""
    backup_relpath: str = ""


@dataclass
class Checkpoint:
    """A file-scoped checkpoint."""

    checkpoint_id: str
    repo_root: str
    created_at: float
    files: list[CheckpointFile]
    git_branch: str = ""
    git_status_short: str = ""
    description: str = ""


class CheckpointRollback:
    """Create and restore file-scoped checkpoints."""

    def __init__(self, repo_root: Path, checkpoint_dir: Path | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint_dir = checkpoint_dir or (self.repo_root / ".aurelius_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create(self, files: list[str], description: str = "") -> Checkpoint:
        """Create a checkpoint for the given files."""
        checkpoint_id = self._new_checkpoint_id(files, description)
        root = self.checkpoint_dir / checkpoint_id
        backups = root / "files"
        backups.mkdir(parents=True, exist_ok=True)

        captured: list[CheckpointFile] = []
        for rel in self._dedupe(files):
            abs_path = self.repo_root / rel
            if abs_path.exists():
                digest = self._sha256(abs_path)
                backup_rel = str(Path("files") / rel)
                backup_abs = root / backup_rel
                backup_abs.parent.mkdir(parents=True, exist_ok=True)
                if abs_path.is_file():
                    shutil.copy2(abs_path, backup_abs)
                    size = abs_path.stat().st_size
                else:
                    size = 0
                captured.append(
                    CheckpointFile(
                        path=rel,
                        existed=True,
                        size_bytes=size,
                        sha256=digest,
                        backup_relpath=backup_rel,
                    )
                )
            else:
                captured.append(CheckpointFile(path=rel, existed=False))

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            repo_root=str(self.repo_root),
            created_at=time.time(),
            files=captured,
            git_branch=self._git(["branch", "--show-current"]),
            git_status_short=self._git(["status", "--short"]),
            description=description,
        )
        (root / "manifest.json").write_text(
            json.dumps(self._checkpoint_to_dict(checkpoint), indent=2)
        )
        return checkpoint

    def restore(self, checkpoint_id: str) -> Checkpoint:
        """Restore files from a checkpoint."""
        checkpoint = self.load(checkpoint_id)
        root = self.checkpoint_dir / checkpoint_id
        for file_meta in checkpoint.files:
            abs_path = self.repo_root / file_meta.path
            if file_meta.existed:
                backup = root / file_meta.backup_relpath
                if backup.exists():
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, abs_path)
            else:
                if abs_path.exists() and abs_path.is_file():
                    abs_path.unlink()
        return checkpoint

    def load(self, checkpoint_id: str) -> Checkpoint:
        manifest = self.checkpoint_dir / checkpoint_id / "manifest.json"
        data = json.loads(manifest.read_text())
        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            repo_root=data["repo_root"],
            created_at=data["created_at"],
            files=[CheckpointFile(**f) for f in data["files"]],
            git_branch=data.get("git_branch", ""),
            git_status_short=data.get("git_status_short", ""),
            description=data.get("description", ""),
        )

    def list_checkpoints(self) -> list[Checkpoint]:
        """List available checkpoints newest first."""
        checkpoints: list[Checkpoint] = []
        for manifest in self.checkpoint_dir.glob("*/manifest.json"):
            try:
                checkpoints.append(self.load(manifest.parent.name))
            except Exception:  # noqa: BLE001, S112
                continue
        return sorted(checkpoints, key=lambda c: c.created_at, reverse=True)

    def cleanup(self, keep_latest: int = 20) -> None:
        """Delete old checkpoints, keeping the newest N."""
        checkpoints = self.list_checkpoints()
        for cp in checkpoints[keep_latest:]:
            root = self.checkpoint_dir / cp.checkpoint_id
            shutil.rmtree(root, ignore_errors=True)

    def _new_checkpoint_id(self, files: list[str], description: str) -> str:
        payload = json.dumps(
            {"t": time.time_ns(), "files": sorted(files), "description": description}
        )
        return "cp_" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _dedupe(self, files: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for f in files:
            rel = os.path.normpath(f)
            if rel.startswith("..") or os.path.isabs(rel):
                raise ValueError(f"Checkpoint path must be repo-relative: {f}")
            if rel not in seen:
                seen.add(rel)
                result.append(rel)
        return result

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def _git(self, args: list[str]) -> str:
        try:
            result = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _checkpoint_to_dict(self, checkpoint: Checkpoint) -> dict:
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "repo_root": checkpoint.repo_root,
            "created_at": checkpoint.created_at,
            "files": [file.__dict__ for file in checkpoint.files],
            "git_branch": checkpoint.git_branch,
            "git_status_short": checkpoint.git_status_short,
            "description": checkpoint.description,
        }
