"""Unit tests for src.cli.pipeline_processor.

Covers all 7 chain operations, the lambda `pipeline()` factory, edge cases,
iterability, repr, + the stdlib-only import constraint.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.cli.pipeline_processor import Pipeline, pipeline


_STDLIB = {"collections", "typing", "dataclasses", "__future__", "collections.abc"}
_SRC = pathlib.Path(__file__).parents[2] / "src" / "cli" / "pipeline_processor.py"


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported - _STDLIB), f"Non-stdlib: {imported - _STDLIB}"


# ── helpers ─────────────────────────────────────────────────────────────────

class TestBasic:
    """Smoke tests for the 7 ops."""

    def setup_method(self):
        self.p = Pipeline([1, 2, 3, 4, 5])

    def test_collect(self):
        assert self.p.collect() == [1, 2, 3, 4, 5]

    def test_filter(self):
        assert self.p.filter(lambda x: x > 2).collect() == [3, 4, 5]

    def test_filter_nothing(self):
        assert self.p.filter(lambda _: False).collect() == []

    def test_map(self):
        assert Pipeline([1, 2, 3]).map(lambda x: x * 2).collect() == [2, 4, 6]

    def test_sort(self):
        assert Pipeline([3, 1, 4, 1, 5]).sort().collect() == [1, 1, 3, 4, 5]

    def test_sort_reverse(self):
        assert Pipeline([1, 3, 2]).sort(reverse=True).collect() == [3, 2, 1]

    def test_sort_by_key(self):
        words = Pipeline(["beta", "alpha", "gamma"]).sort(key=len).collect()
        assert words == ["beta", "alpha", "gamma"]

    def test_head(self):
        assert Pipeline([1, 2, 3, 4, 5]).head(3).collect() == [1, 2, 3]

    def test_head_too_many(self):
        assert Pipeline([1, 2]).head(10).collect() == [1, 2]

    def test_head_zero(self):
        assert Pipeline([1, 2, 3]).head(0).collect() == []

    def test_head_negative(self):
        assert Pipeline([1, 2, 3]).head(-1).collect() == []

    def test_tail(self):
        assert Pipeline([1, 2, 3, 4, 5]).tail(2).collect() == [4, 5]

    def test_tail_too_many(self):
        assert Pipeline([1, 2]).tail(10).collect() == [1, 2]

    def test_tail_zero(self):
        assert Pipeline([1, 2, 3]).tail(0).collect() == []

    def test_dedup_consecutive(self):
        assert Pipeline([1, 1, 2, 2, 2, 3]).dedup().collect() == [1, 2, 3]

    def test_dedup_non_consecutive(self):
        assert Pipeline([1, 2, 1]).dedup().collect() == [1, 2, 1]

    def test_dedup_empty(self):
        assert Pipeline([]).dedup().collect() == []

    def test_group_by(self):
        result = Pipeline([1, 2, 3, 4, 5, 6]).group_by(lambda x: x % 2)
        assert result == {0: [2, 4, 6], 1: [1, 3, 5]}

    def test_group_by_string(self):
        result = Pipeline(["ab", "cd", "ef"]).group_by(lambda s: s[0])
        assert result == {"a": ["ab"], "c": ["cd"], "e": ["ef"]}


# ── chaining ─────────────────────────────────────────────────────────────────

class TestChaining:
    def test_filter_then_map(self):
        r = Pipeline(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).collect()
        assert r == [0, 6, 12, 18, 24]

    def test_filter_then_sort(self):
        r = Pipeline([3, 1, 4, 1, 5, 9]).filter(lambda x: x % 2 == 1).sort().collect()
        assert r == [1, 1, 3, 5, 9]

    def test_head_then_tail(self):
        r = Pipeline(range(11)).head(8).tail(3).collect()
        assert r == [5, 6, 7]

    def test_sort_then_dedup(self):
        r = Pipeline([1, 2, 1, 3, 2]).sort().dedup().collect()
        assert r == [1, 2, 3]

    def test_group_by_then_len(self):
        result = Pipeline(range(6)).group_by(lambda x: x % 3)
        assert len(result) == 3
        assert result[0] == [0, 3]


# ── factory + repr ───────────────────────────────────────────────────────────

class TestFactory:
    def test_pipeline_factory(self):
        r = pipeline([1, 2, 3]).filter(lambda x: x > 1).collect()
        assert r == [2, 3]

    def test_len(self):
        assert len(Pipeline([1, 2, 3])) == 3
        assert len(Pipeline([])) == 0

    def test_iter(self):
        assert list(Pipeline([1, 2, 3])) == [1, 2, 3]

    def test_repr(self):
        r = repr(Pipeline([1, 2, 3]))
        assert "Pipeline" in r
        assert "[1, 2, 3]" in r

    def test_to_pipeline(self):
        r = Pipeline([1, 2, 3]).map(lambda x: x + 1).to_pipeline().collect()
        assert r == [2, 3, 4]
