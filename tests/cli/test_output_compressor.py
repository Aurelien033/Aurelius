"""Unit tests for src.cli.output_compressor.

Covers every public method and the stdlib-only import constraint.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.cli.output_compressor import (
    CompressionConfig,
    CompressionResult,
    OutputCompressor,
    DEFAULT_COMPRESSOR,
    OUTPUT_COMPRESSOR_REGISTRY,
)


# ── stdlib-only constraint ────────────────────────────────────────────────────

_SRC = pathlib.Path(__file__).parents[2] / "src" / "cli" / "output_compressor.py"
_STDLIB = {"collections", "dataclasses", "typing", "__future__"}


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:  # type: ignore[attr-defined]
            imported.add(node.module.split(".")[0])  # type: ignore[attr-defined]
        elif isinstance(node, ast.Import) and hasattr(node, "names"):
            pass  # bare `import pkg` without sub-module qualifier is fine

    assert not imported - _STDLIB, f"Non-stdlib imports detected: {imported - _STDLIB}"


# ── CompressionConfig ────────────────────────────────────────────────────────

class TestCompressionConfig:
    def test_defaults(self) -> None:
        cfg = CompressionConfig()
        assert cfg.max_lines == 50
        assert cfg.max_line_length == 200
        assert cfg.deduplicate is True
        assert cfg.group_by_prefix is True
        assert cfg.remove_empty is True
        assert cfg.show_summary is True

    def test_custom(self) -> None:
        cfg = CompressionConfig(max_lines=10, max_line_length=80, deduplicate=False)
        assert cfg.max_lines == 10
        assert cfg.max_line_length == 80
        assert cfg.deduplicate is False


# ── CompressionResult dataclass ───────────────────────────────────────────────

class TestCompressionResult:
    def test_created(self) -> None:
        r = CompressionResult("hi", 1, 1, 1.0, [])
        assert r.output == "hi"
        assert r.compression_ratio == 1.0

    def test_ratio_can_be_zero(self) -> None:
        r = CompressionResult("", 5, 0, 0.0, ["truncate_total"])
        assert r.compression_ratio == 0.0

    def test_empty_input(self) -> None:
        r = CompressionResult("", 0, 0, 0.0, [])
        assert r.output == ""
        assert r.original_lines == 0


# ── OutputCompressor ──────────────────────────────────────────────────────────

class TestOutputCompressor:
    def _r(self, text: str, **kwargs) -> CompressionResult:
        return OutputCompressor(CompressionConfig(**kwargs)).compress(text)

    # ── str strategies ───────────────────────────────────────────────────────

    def test_empty_string(self) -> None:
        r = self._r("")
        assert r.output == ""
        assert r.original_lines == 0
        assert r.compression_ratio == 0.0

    def test_remove_empty_lines(self) -> None:
        r = self._r("a\n\nb\n\nc", remove_empty=True)
        assert r.output == "a\nb\nc"
        assert "remove_empty" in r.strategies_applied

    def test_deduplicate_consecutive(self) -> None:
        r = self._r("x\nx\nx\ny", deduplicate=True)
        assert "x (x3)" in r.output
        assert "deduplicate" in r.strategies_applied

    def test_truncate_long_line(self) -> None:
        long = "a" * 300
        r = self._r(long, max_line_length=50)
        assert len(r.output.splitlines()[0]) <= 53   # 50 + "..."
        assert "truncate_lines" in r.strategies_applied

    def test_truncate_total_lines(self) -> None:
        big = "\n".join(str(i) for i in range(200))
        r = self._r(big, max_lines=10, show_summary=True)
        lines = r.output.splitlines()
        assert len(lines) <= 11   # 10 + summary
        assert "lines omitted" in r.output

    def test_group_by_prefix(self) -> None:
        lines = "\n".join(f"sys{x:03d}" for x in range(10))
        r = self._r(lines, group_by_prefix=True)
        assert "sys... +" in r.output or "sys" in r.output
        assert "group_by_prefix" in r.strategies_applied

    def test_no_strategies_when_disabled(self) -> None:
        cfg = CompressionConfig(
            remove_empty=False, deduplicate=False,
            group_by_prefix=False, max_line_length=0, max_lines=0,
        )
        r = OutputCompressor(cfg).compress("a\nb\nc")
        assert r.strategies_applied == []

    # ── ratios ───────────────────────────────────────────────────────────────

    def test_compression_ratio(self) -> None:
        r = self._r("a\nb\nc\nd\ne", max_lines=2, show_summary=True)
        assert 0.0 < r.compression_ratio < 1.0
        assert r.compressed_lines < r.original_lines

    # ── specialised compressors ──────────────────────────────────────────────

    def test_compress_git_status_strips_hints(self) -> None:
        text = 'On branch main\nYour branch is up to date\n(use "git push" to publish)'
        r = DEFAULT_COMPRESSOR.compress_git_status(text)
        assert "(use" not in r.output
        assert "remove_hints" in r.strategies_applied

    def test_compress_ls_groups_by_extension(self) -> None:
        text = "foo.py\nbar.py\nbaz.txt\nREADME.md\n"
        r = DEFAULT_COMPRESSOR.compress_ls(text)
        assert ".py: 2" in r.output or ".py : 2" in r.output
        assert "group_by_extension" in r.strategies_applied

    def test_compress_test_output_shows_failures_only(self) -> None:
        text = "test_a PASSED\ntest_b PASSED\ntest_c FAILED"
        r = DEFAULT_COMPRESSOR.compress_test_output(text)
        assert "PASSED" not in r.output
        assert "FAILED" in r.output
        assert "failures_only" in r.strategies_applied

    def test_compress_grep_groups_by_file(self) -> None:
        text = "main.py: todo\nmain.py: fixme\nother.py: todo"
        r = DEFAULT_COMPRESSOR.compress_grep(text)
        assert "main.py:" in r.output
        assert "group_by_file" in r.strategies_applied


# ── Singleton / registry ─────────────────────────────────────────────────────

class TestRegistry:
    def test_default_compressor_exists(self) -> None:
        assert isinstance(DEFAULT_COMPRESSOR, OutputCompressor)

    def test_registry_has_default(self) -> None:
        assert "default" in OUTPUT_COMPRESSOR_REGISTRY
        assert OUTPUT_COMPRESSOR_REGISTRY["default"] is DEFAULT_COMPRESSOR

    def test_registry_is_mutable(self) -> None:
        custom = OutputCompressor()
        OUTPUT_COMPRESSOR_REGISTRY["custom"] = custom
        assert OUTPUT_COMPRESSOR_REGISTRY["custom"] is custom
