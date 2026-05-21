"""Unit tests for src.memory.layered_memory.

Covers the 5-layer store, TTL eviction, deduplication, retrieval, promotion,
and the stdlib-only import constraint.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import datetime, timezone, timedelta

import pytest

from src.memory.layered_memory import (
    LayeredMemory,
    LayeredMemoryEntry,
    LayeredMemoryError,
    MemoryLayer,
    DEFAULT_LAYERED_MEMORY,
    LAYERED_MEMORY_REGISTRY,
)

# ── stdlib-only constraint ────────────────────────────────────────────────────

_SRC = pathlib.Path(__file__).parents[2] / "src" / "memory" / "layered_memory.py"
_STDLIB = {"collections", "dataclasses", "uuid", "datetime", '__future__'}


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported - _STDLIB, f"Non-stdlib imports: {imported - _STDLIB}"


# ── LayeredMemoryEntry ───────────────────────────────────────────────────────

class TestLayeredMemoryEntry:
    def test_auto_entry_id(self) -> None:
        e = LayeredMemoryEntry(content="hello")
        assert len(e.entry_id) == 8
        assert e.content == "hello"

    def test_explicit_entry_id(self) -> None:
        e = LayeredMemoryEntry(entry_id="abc", content="hi")
        assert e.entry_id == "abc"

    def test_default_timestamp_is_recent(self) -> None:
        before = datetime.now(timezone.utc)
        e = LayeredMemoryEntry(content="x")
        after = datetime.now(timezone.utc)
        assert before <= e.timestamp <= after

    def test_defaults(self) -> None:
        e = LayeredMemoryEntry()
        assert e.access_count == 0
        assert e.importance_score == 0.5


# ── LayeredMemory default construction ───────────────────────────────────────

class TestDefaultConstruction:
    def test_five_layers(self) -> None:
        lm = LayeredMemory()
        assert len(lm._layers) == 5

    def test_layer_names(self) -> None:
        lm = LayeredMemory()
        names = {layer.name for layer in lm._layers.values()}
        assert "L0 Meta Rules" in names
        assert "L1 Insight Index" in names
        assert "L4 Session Archive" in names

    def test_custom_layers(self) -> None:
        layers = [MemoryLayer(level=0, name="only", max_entries=10)]
        lm = LayeredMemory(layers=layers)
        assert len(lm._layers) == 1

    def test_default_singleton(self) -> None:
        assert isinstance(DEFAULT_LAYERED_MEMORY, LayeredMemory)

    def test_registry_has_default(self) -> None:
        assert "default" in LAYERED_MEMORY_REGISTRY


# ── store ────────────────────────────────────────────────────────────────────

class TestStore:
    def test_store_string(self) -> None:
        lm = LayeredMemory()
        entry = lm.store("raw content", "L1 Insight Index")
        assert entry.content == "raw content"
        assert entry.layer == "L1 Insight Index"

    def test_store_entry_object(self) -> None:
        lm = LayeredMemory()
        e = LayeredMemoryEntry(content="obj", layer="L2 Global Facts")
        stored = lm.store(e, "L2 Global Facts")
        assert stored is e

    def test_store_increments_len(self) -> None:
        lm = LayeredMemory()
        lm.store("a", "L0 Meta Rules")
        lm.store("b", "L0 Meta Rules")
        assert len(lm) == 2

    def test_store_unknown_layer_raises(self) -> None:
        lm = LayeredMemory()
        with pytest.raises(LayeredMemoryError):
            lm.store("x", "NoSuchLayer")

    def test_store_kwargs_passed_through(self) -> None:
        lm = LayeredMemory()
        e = lm.store("doc", "L1 Insight Index", importance_score=0.9)
        assert e.importance_score == 0.9


# ── retrieve ─────────────────────────────────────────────────────────────────

class TestRetrieve:
    def setup_method(self) -> None:
        self.lm = LayeredMemory()
        self.lm.store("apple pie recipe", "L1 Insight Index")
        self.lm.store("banana bread", "L2 Global Facts")

    def test_case_insensitive(self) -> None:
        results = self.lm.retrieve("APPLE")
        assert len(results) == 1

    def test_specific_layer(self) -> None:
        results = self.lm.retrieve("banana", layer_name="L2 Global Facts")
        assert len(results) == 1

    def test_no_match(self) -> None:
        results = self.lm.retrieve("zebra")
        assert results == []

    def test_increments_access_count(self) -> None:
        results = self.lm.retrieve("apple")
        assert results[0].access_count == 1
        self.lm.retrieve("apple")
        assert results[0].access_count == 2

    def test_all_layers_when_none_specified(self) -> None:
        results = self.lm.retrieve("banana")
        assert len(results) == 1  # L2 matches


# ── evict_expired ─────────────────────────────────────────────────────────────

class TestTTLEviction:
    def test_ttl_expiry(self) -> None:
        lm = LayeredMemory()
        old = LayeredMemoryEntry(
            content="stale",
            layer="L1 Insight Index",
            timestamp=datetime.now(timezone.utc) - timedelta(days=8),
        )
        lm.store(old, "L1 Insight Index")
        assert len(lm) == 1
        removed = lm.evict_expired()
        assert removed == 1
        assert len(lm) == 0

    def test_no_ttl_no_expiry(self) -> None:
        lm = LayeredMemory()
        lm.store("forever", "L0 Meta Rules")
        removed = lm.evict_expired()
        assert removed == 0

    def test_mixed_layers(self) -> None:
        lm = LayeredMemory()
        lm.store("fresh", "L1 Insight Index")
        old = LayeredMemoryEntry(
            content="stale",
            layer="L1 Insight Index",
            timestamp=datetime.now(timezone.utc) - timedelta(days=8),
        )
        lm.store(old, "L1 Insight Index")
        removed = lm.evict_expired()
        assert removed == 1
        assert len(lm) == 1


# ── promote ──────────────────────────────────────────────────────────────────

class TestPromote:
    def setup_method(self) -> None:
        self.lm = LayeredMemory()

    def test_promote_with_access(self) -> None:
        entry = self.lm.store("skill", "L2 Global Facts")
        entry.access_count = 10
        result = self.lm.promote(entry.entry_id)
        assert result is True
        assert entry.layer == "L1 Insight Index"

    def test_promote_high_importance_bypasses_access_threshold(self) -> None:
        entry = self.lm.store("vital", "L2 Global Facts")
        entry.importance_score = 0.95
        assert self.lm.promote(entry.entry_id) is True
        assert entry.layer == "L1 Insight Index"

    def test_promote_not_found_raises(self) -> None:
        with pytest.raises(LayeredMemoryError):
            self.lm.promote("__ghost__")

    def test_promote_from_l0_returns_false(self) -> None:
        entry = self.lm.store("meta", "L0 Meta Rules")
        entry.access_count = 100
        result = self.lm.promote(entry.entry_id)
        assert result is False

    def test_promote_below_threshold_fails(self) -> None:
        entry = self.lm.store("weak", "L2 Global Facts")
        entry.access_count = 1
        entry.importance_score = 0.3
        result = self.lm.promote(entry.entry_id)
        assert result is False
        assert entry.layer == "L2 Global Facts"


# ── dump_layer ────────────────────────────────────────────────────────────────

class TestDumpLayer:
    def test_returns_list_copy(self) -> None:
        lm = LayeredMemory()
        lm.store("a", "L0 Meta Rules")
        dumped = lm.dump_layer("L0 Meta Rules")
        assert len(dumped) == 1
        dumped.append(LayeredMemoryEntry(content="extra"))
        assert len(lm.dump_layer("L0 Meta Rules")) == 1  # original unaffected


# ── search ───────────────────────────────────────────────────────────────────

class TestSearch:
    def setup_method(self) -> None:
        self.lm = LayeredMemory()
        self.lm.store("python async pattern", "L2 Global Facts", importance_score=0.9)
        self.lm.store("rust ownership guide", "L2 Global Facts", importance_score=0.5)

    def test_keyword_match(self) -> None:
        results = self.lm.search("python")
        assert len(results) >= 1
        assert "python" in results[0].content.lower()

    def test_no_match_returns_empty(self) -> None:
        results = self.lm.search("quantum computing")
        assert results == []

    def test_top_k_respected(self) -> None:
        results = self.lm.search("guide", top_k=1)
        assert len(results) <= 1

    def test_importance_affects_ranking(self) -> None:
        # python entry has importance_score=0.9 → should rank first
        results = self.lm.search("guide")
        # only one entry matches 'guide'
        assert results[0].content.startswith("rust")


# ── len ──────────────────────────────────────────────────────────────────────

class TestLen:
    def test_empty(self) -> None:
        assert len(LayeredMemory()) == 0

    def test_after_stores(self) -> None:
        lm = LayeredMemory()
        lm.store("a", "L0 Meta Rules")
        lm.store("b", "L1 Insight Index")
        assert len(lm) == 2
