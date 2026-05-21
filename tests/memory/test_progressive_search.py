"""Unit tests for src.memory.progressive_search.

Covers the 3-layer progressive search pipeline and stdlib-only import constraint.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.memory.progressive_search import (
    IndexEntry,
    ProgressiveSearchError,
    ProgressiveSearcher,
    SearchResult,
    DEFAULT_PROGRESSIVE_SEARCHER,
    PROGRESSIVE_SEARCH_REGISTRY,
)

# ── stdlib-only constraint ────────────────────────────────────────────────────

_SRC = pathlib.Path(__file__).parents[2] / "src" / "memory" / "progressive_search.py"
_STDLIB = {"collections", "dataclasses", "typing", "re", '__future__'}


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported - _STDLIB, f"Non-stdlib imports: {imported - _STDLIB}"


# ── IndexEntry / SearchResult dataclasses ─────────────────────────────────────

class TestDataClasses:
    def test_index_entry_defaults(self) -> None:
        e = IndexEntry(entry_id="a", summary_tags=["tag"], timestamp=1.0, layer=0)
        assert e.access_count == 0

    def test_search_result(self) -> None:
        r = SearchResult(
            entry_id="x", score=0.9,
            timeline_context=["a: [0]: tag"],
            full_content="hello",
        )
        assert r.full_content == "hello"


# ── ProgressiveSearcher construction ─────────────────────────────────────────

class TestConstruction:
    def test_empty_index(self) -> None:
        ps = ProgressiveSearcher()
        assert ps.search("anything") == []

    def test_seed_index(self) -> None:
        entries = [
            IndexEntry("e1", ["alpha"], 1.0, 0),
            IndexEntry("e2", ["beta"], 2.0, 1),
        ]
        ps = ProgressiveSearcher(index=entries)
        assert len(ps._index) == 2

    def test_default_singleton(self) -> None:
        assert isinstance(DEFAULT_PROGRESSIVE_SEARCHER, ProgressiveSearcher)

    def test_registry(self) -> None:
        assert "default" in PROGRESSIVE_SEARCH_REGISTRY


# ── index_entry / set_full_content / remove ───────────────────────────────────

class TestIndexMutation:
    def setup_method(self) -> None:
        self.ps = ProgressiveSearcher()

    def test_index_and_find(self) -> None:
        self.ps.index_entry(IndexEntry("e1", ["alpha"], 1.0, 0))
        r = self.ps.search("alpha")
        assert len(r) == 1

    def test_index_overwrite(self) -> None:
        self.ps.index_entry(IndexEntry("e1", ["alpha"], 1.0, 0))
        self.ps.index_entry(IndexEntry("e1", ["beta"], 2.0, 0))
        assert len(self.ps._index) == 1

    def test_set_full_content(self) -> None:
        self.ps.index_entry(IndexEntry("e1", ["alpha"], 1.0, 0))
        self.ps.set_full_content("e1", "the full text")
        r = self.ps.search("alpha", fetch_full=True)
        assert r[0].full_content == "the full text"

    def test_remove_existing(self) -> None:
        self.ps.index_entry(IndexEntry("e1", ["alpha"], 1.0, 0))
        assert self.ps.remove("e1") is True
        assert self.ps.search("alpha") == []

    def test_remove_missing(self) -> None:
        assert self.ps.remove("__ghost__") is False


# ── search pipeline ───────────────────────────────────────────────────────────

class TestSearch:
    def setup_method(self) -> None:
        self.ps = ProgressiveSearcher([
            IndexEntry("e1", ["python", "async"], 1.0, 0, access_count=5),
            IndexEntry("e2", ["rust", "ownership"], 2.0, 1, access_count=2),
            IndexEntry("e3", ["python", "beginner"], 3.0, 0, access_count=10),
        ])
        self.ps.set_full_content("e1", "Python async patterns explained")
        self.ps.set_full_content("e2", "Rust ownership deep dive")
        self.ps.set_full_content("e3", "Python for beginners")

    def test_keyword_overlap_scores(self) -> None:
        results = self.ps.search("python async")
        assert len(results) >= 1
        ids = [r.entry_id for r in results]
        assert "e1" in ids

    def test_no_match_returns_empty(self) -> None:
        assert self.ps.search("quantum") == []

    def test_empty_query_returns_empty(self) -> None:
        assert self.ps.search("   ") == []

    def test_top_k_limit(self) -> None:
        results = self.ps.search("python", top_k=1)
        assert len(results) <= 1

    def test_fetch_full_content(self) -> None:
        results = self.ps.search("rust", fetch_full=True)
        assert results[0].full_content is not None
        assert "Rust" in results[0].full_content

    def test_fetch_full_false(self) -> None:
        results = self.ps.search("rust", fetch_full=False)
        assert results[0].full_content is None

    def test_timeline_context_populated(self) -> None:
        results = self.ps.search("python", timeline_radius=1)
        assert len(results[0].timeline_context) > 0

    def test_timeline_radius_zero(self) -> None:
        results = self.ps.search("python", timeline_radius=0)
        assert results[0].timeline_context == []

    def test_negative_top_k_raises(self) -> None:
        with pytest.raises(ProgressiveSearchError):
            self.ps.search("x", top_k=-1)   # type: ignore[arg-type]

    def test_negative_radius_raises(self) -> None:
        with pytest.raises(ProgressiveSearchError):
            self.ps.search("x", timeline_radius=-1)  # type: ignore[arg-type]

    def test_access_count_affects_ranking(self) -> None:
        # e3 has same keyword match as e1 but higher access_count → ranks higher
        results = self.ps.search("python", top_k=2)
        ids = [r.entry_id for r in results]
        assert ids.index("e3") < ids.index("e1")


# ── stats ─────────────────────────────────────────────────────────────────────

class TestStats:
    def test_empty(self) -> None:
        ps = ProgressiveSearcher()
        s = ps.stats()
        assert s["index_size"] == 0

    def test_populated(self) -> None:
        ps = ProgressiveSearcher([
            IndexEntry("e1", ["a", "b"], 1.0, 0, access_count=3),
            IndexEntry("e2", ["c"], 2.0, 1, access_count=5),
        ])
        ps.set_full_content("e1", "text")
        s = ps.stats()
        assert s["index_size"] == 2
        assert s["avg_tags_per_entry"] == 1.5
        assert s["full_content_store_size"] == 1
        assert s["avg_access_count"] == 4.0
