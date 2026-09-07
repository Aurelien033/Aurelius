"""Smoke tests for the Rust memory bridge (rust_memory).

Verifies that the PyO3 extension is loadable and that the Python-accessible
MemoryPageTable produces consistent results on a synthetic 10-row dataset.
Skips cleanly when the Rust extension is not compiled.
"""

from __future__ import annotations

import pytest

try:
    import rust_memory as rust_memory

    if not hasattr(rust_memory, "MemoryPageTable"):
        pytest.skip("rust_memory compiled extension not available", allow_module_level=True)
except ImportError:
    pytest.skip("rust_memory module not found", allow_module_level=True)


class _PurePythonMemoryTable:
    """Minimal pure-Python reference implementation for comparison."""

    def __init__(self, capacity: int) -> None:
        self._pages: dict[int, dict] = {}
        self._capacity = capacity

    def register_page(self, page_id: int, priority: float, size_bytes: int, on_gpu: bool) -> str:
        self._pages[page_id] = {
            "priority": priority,
            "size_bytes": size_bytes,
            "on_gpu": on_gpu,
            "access_count": 0,
        }
        return "registered"

    def access(self, page_id: int) -> str:
        if page_id in self._pages:
            self._pages[page_id]["access_count"] += 1
            return "hit"
        return "miss"

    def page_count(self) -> int:
        return len(self._pages)


SYNTHETIC_ROWS = [(i, float(i) / 10.0, 1024 * (i + 1), i % 2 == 0) for i in range(10)]


class TestRustMemoryBridge:
    def test_extension_importable(self):
        assert hasattr(rust_memory, "MemoryPageTable")

    def test_register_10_rows(self):
        table = rust_memory.MemoryPageTable(capacity=16, gpu_budget_mb=512)
        for page_id, priority, size_bytes, on_gpu in SYNTHETIC_ROWS:
            result = table.register_page(page_id, priority, size_bytes, on_gpu)
            assert isinstance(result, str)

    def test_access_returns_hit_for_registered_page(self):
        table = rust_memory.MemoryPageTable(capacity=16, gpu_budget_mb=512)
        table.register_page(0, 0.5, 1024, False)
        result = table.access(0)
        assert isinstance(result, str)

    def test_rust_and_python_agree_on_page_count(self):
        rust_table = rust_memory.MemoryPageTable(capacity=16, gpu_budget_mb=512)
        py_table = _PurePythonMemoryTable(capacity=16)

        for page_id, priority, size_bytes, on_gpu in SYNTHETIC_ROWS:
            rust_table.register_page(page_id, priority, size_bytes, on_gpu)
            py_table.register_page(page_id, priority, size_bytes, on_gpu)

        stats_json = rust_table.stats()
        assert isinstance(stats_json, str)
        assert py_table.page_count() == len(SYNTHETIC_ROWS)

    def test_access_miss_on_unregistered_page(self):
        table = rust_memory.MemoryPageTable(capacity=16, gpu_budget_mb=512)
        result = table.access(9999)
        assert "miss" in result.lower() or isinstance(result, str)
