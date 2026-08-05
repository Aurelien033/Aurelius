"""
ContextAssembler — token-budgeted repository context selection for Composer.

A Cursor/Composer-style coding agent does not become good by dumping the whole
repo into the model. It becomes good by selecting the smallest causally useful
context: task-relevant files, symbol windows, tests, dependency neighbors, and
recently edited files. This module makes that context selection explicit and
measurable.

Design goals:
  - deterministic ranking, not opaque embedding magic
  - exact byte/line accounting and explicit token estimates
  - safe truncation by file windows rather than arbitrary string slicing
  - no repo mutation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .codebase_indexer import CodebaseIndexer, SearchHit


@dataclass
class ContextBudget:
    """Budget controls for an assembled prompt context."""

    max_tokens: int = 32_000
    reserve_tokens: int = 4_000
    chars_per_token: float = 4.0
    max_file_tokens: int = 4_000
    symbol_window_lines: int = 80

    @property
    def usable_tokens(self) -> int:
        return max(0, self.max_tokens - self.reserve_tokens)

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return int(math.ceil(len(text) / self.chars_per_token))


@dataclass
class ContextRequest:
    """Inputs for context assembly."""

    task: str
    explicit_files: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    include_tests: bool = True
    include_import_neighbors: bool = True


@dataclass
class ContextItem:
    """A selected context block."""

    file: str
    start_line: int
    end_line: int
    text: str
    reason: str
    score: float
    estimated_tokens: int
    symbol: str | None = None


@dataclass
class AssembledContext:
    """Final prompt-ready context packet."""

    request: ContextRequest
    budget: ContextBudget
    items: list[ContextItem]
    total_estimated_tokens: int
    omitted_files: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            "# Aurelius Composer Context",
            f"Task: {self.request.task}",
            f"Estimated context tokens: {self.total_estimated_tokens}/{self.budget.usable_tokens}",
            "",
        ]
        for item in self.items:
            header = (
                f"## {item.file}:{item.start_line}-{item.end_line} "
                f"score={item.score:.3f} reason={item.reason}"
            )
            if item.symbol:
                header += f" symbol={item.symbol}"
            parts.append(header)
            parts.append("```")
            parts.append(item.text.rstrip())
            parts.append("```")
            parts.append("")
        if self.omitted_files:
            parts.append("# Omitted files due to budget")
            for f in self.omitted_files:
                parts.append(f"- {f}")
        return "\n".join(parts)


class ContextAssembler:
    """Assemble a compact, ranked context packet for a coding task."""

    def __init__(self, repo_root: Path, indexer: CodebaseIndexer):
        self.repo_root = Path(repo_root).resolve()
        self.indexer = indexer

    def assemble(
        self, request: ContextRequest, budget: ContextBudget | None = None
    ) -> AssembledContext:
        budget = budget or ContextBudget()
        if self.indexer.stats.total_files == 0:
            self.indexer.build()

        candidates: dict[tuple[str, int, int, str | None], ContextItem] = {}

        def add_item(item: ContextItem) -> None:
            key = (item.file, item.start_line, item.end_line, item.symbol)
            existing = candidates.get(key)
            if existing is None or item.score > existing.score:
                candidates[key] = item

        # 1. Explicit files are highest priority.
        for filepath in request.explicit_files:
            item = self._whole_or_head(filepath, reason="explicit", score=1.00, budget=budget)
            if item:
                add_item(item)

        # 2. Recent files are highly relevant but lower than explicit files.
        for filepath in request.recent_files:
            item = self._whole_or_head(filepath, reason="recent", score=0.85, budget=budget)
            if item:
                add_item(item)

        # 3. Search query hits from task and provided queries.
        search_queries = [request.task, *request.queries]
        for q_i, query in enumerate(search_queries):
            if not query.strip():
                continue
            for hit in self.indexer.search(query, top_k=12):
                score = max(0.20, hit.score - 0.02 * q_i)
                item = self._hit_window(
                    hit, reason=f"query:{query[:48]}", score=score, budget=budget
                )
                if item:
                    add_item(item)

        # 4. Test neighbors for source files.
        if request.include_tests:
            for filepath in list({item.file for item in candidates.values()}):
                for test_path in self._candidate_test_paths(filepath):
                    item = self._whole_or_head(
                        test_path, reason=f"test-neighbor:{filepath}", score=0.72, budget=budget
                    )
                    if item:
                        add_item(item)

        # 5. Import neighbors, useful for API contracts.
        if request.include_import_neighbors:
            for filepath in list({item.file for item in candidates.values()}):
                entry = self.indexer.get_entry(filepath)
                if not entry:
                    continue
                for imp in entry.imports[:8]:
                    neighbors = self.indexer.files_importing(imp)[:4]
                    for neighbor in neighbors:
                        if neighbor != filepath:
                            item = self._whole_or_head(
                                neighbor,
                                reason=f"import-neighbor:{imp}",
                                score=0.50,
                                budget=budget,
                            )
                            if item:
                                add_item(item)

        ranked = sorted(
            candidates.values(), key=lambda x: (x.score, -x.estimated_tokens), reverse=True
        )
        selected: list[ContextItem] = []
        omitted: list[str] = []
        used = 0
        for item in ranked:
            if used + item.estimated_tokens <= budget.usable_tokens:
                selected.append(item)
                used += item.estimated_tokens
            else:
                omitted.append(item.file)

        return AssembledContext(
            request=request,
            budget=budget,
            items=selected,
            total_estimated_tokens=used,
            omitted_files=sorted(set(omitted)),
        )

    def _hit_window(
        self, hit: SearchHit, reason: str, score: float, budget: ContextBudget
    ) -> ContextItem | None:
        path = self.repo_root / hit.file
        if not path.is_file():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None

        symbol_name = hit.symbol
        start = max(1, hit.line - budget.symbol_window_lines // 2)
        end = min(len(lines), hit.line + budget.symbol_window_lines // 2)

        # If the hit is inside a known symbol, prefer the symbol's line span.
        for symbol in self.indexer.symbols_for_file(hit.file):
            if symbol.line_start <= hit.line <= symbol.line_end:
                start = max(1, symbol.line_start - 5)
                end = min(len(lines), symbol.line_end + 5)
                symbol_name = symbol.name
                break

        return self._make_item(hit.file, lines, start, end, reason, score, budget, symbol_name)

    def _whole_or_head(
        self, filepath: str, reason: str, score: float, budget: ContextBudget
    ) -> ContextItem | None:
        path = self.repo_root / filepath
        if not path.is_file():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        if not lines:
            return None

        whole_text = "\n".join(lines)
        whole_tokens = budget.estimate_tokens(whole_text)
        if whole_tokens <= budget.max_file_tokens:
            return self._make_item(filepath, lines, 1, len(lines), reason, score, budget, None)

        # File too large: include head plus tail, but count as one item.
        approx_lines = max(20, int(budget.max_file_tokens * budget.chars_per_token / 80))
        head_count = approx_lines // 2
        tail_count = approx_lines - head_count
        selected_lines = (
            lines[:head_count] + ["... omitted middle of large file ..."] + lines[-tail_count:]
        )
        text = "\n".join(selected_lines)
        return ContextItem(
            file=filepath,
            start_line=1,
            end_line=len(lines),
            text=text,
            reason=f"{reason}:head-tail",
            score=score * 0.92,
            estimated_tokens=budget.estimate_tokens(text),
        )

    def _make_item(
        self,
        filepath: str,
        lines: list[str],
        start_line: int,
        end_line: int,
        reason: str,
        score: float,
        budget: ContextBudget,
        symbol: str | None,
    ) -> ContextItem:
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        numbered = [
            f"{i + 1}|{line}" for i, line in enumerate(lines[start_idx:end_idx], start=start_idx)
        ]
        text = "\n".join(numbered)
        return ContextItem(
            file=filepath,
            start_line=start_line,
            end_line=end_line,
            text=text,
            reason=reason,
            score=score,
            estimated_tokens=budget.estimate_tokens(text),
            symbol=symbol,
        )

    def _candidate_test_paths(self, filepath: str) -> list[str]:
        """Return plausible test files for a source file."""
        path = Path(filepath)
        candidates: list[str] = []
        if filepath.startswith("src/"):
            rel = Path(filepath[4:])
            stem = rel.stem
            parent = rel.parent
            candidates.append(str(Path("tests") / parent / f"test_{stem}.py"))
            candidates.append(str(Path("tests") / f"test_{stem}.py"))
        if filepath.startswith("crates/"):
            candidates.append(str(path.parent / "tests.rs"))
        return [c for c in candidates if (self.repo_root / c).is_file()]
