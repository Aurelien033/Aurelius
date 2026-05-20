"""Tests for normal package-import paths of AMC memory modules.

These run with the standard import path, not spec_from_file_location,
and must continue to pass as __init__.py evolves.
"""
import pytest


class TestAMCMemoryImports:
    """Normal src.memory import stability."""

    def test_import_src_memory(self):
        import src.memory  # noqa: F401

    def test_import_amc_runtime_cache(self) -> None:
        from src.memory import amc_runtime_cache  # noqa: F401

    def test_import_amc_tensor_api(self) -> None:
        from src.memory import amc_tensor_api  # noqa: F401

    def test_import_prefix_compiler(self) -> None:
        from src.memory.amc_runtime_cache import AMCPrefixCompiler  # noqa: F401

    def test_import_memory_block(self) -> None:
        from src.memory.amc_runtime_cache import AMCMemoryBlock  # noqa: F401

    def test_import_tensor_state(self) -> None:
        from src.memory.amc_tensor_api import AMCTensorState  # noqa: F401

    def test_cache_key_is_importable(self) -> None:
        from src.memory.amc_runtime_cache import AMCMemoryCacheKey  # noqa: F401

    def test_write_decision_is_importable(self) -> None:
        from src.memory.amc_runtime_cache import AMCWriteDecision  # noqa: F401

    def test_memory_all_exports_importable(self) -> None:
        """All src.memory.__all__ symbols are present after import.

        Uses MEMORY_REGISTRY as a representative exported symbol;
        amc_runtime_cache sub-module symbols are tested separately.
        Does NOT reload the module (reload can reset sub-module attrs
        via import cache ordering).
        """
        import src.memory as mem
        for name in ("AMCTier2Hook", "MEMORY_REGISTRY"):
            assert hasattr(mem, name), f"{name} missing from src.memory"
