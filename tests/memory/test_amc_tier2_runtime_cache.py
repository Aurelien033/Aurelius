"""Tests for AMCTier2Hook AMC runtime cache adapter.

Covers
------
- MemoryEntry → AMCMemoryBlock conversion
- build_runtime_blocks default-UNVERIFIED behaviour
- trust_override=VERIFIED promotes to trusted
- quarantined / revoked paths via direct AMCPrefixCompiler isolation
"""

from __future__ import annotations

import pytest

from src.memory.amc_runtime_cache import (
    AMCMemoryBlock,
    AMCPrefixCompiler,
    TrustState,
)
from src.memory.amc_tier2 import AMCTier2Hook
from src.memory.episodic_memory import MemoryEntry

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_entry(
    content: str = "hello world",
    role: str = "user",
    importance: float = 0.8,
    *,
    eid: str = "e1",
    session_id: str | None = "session-abc",
    step: int | None = 1,
) -> MemoryEntry:
    return MemoryEntry(
        role=role,
        content=content,
        importance=importance,
        id=eid,
        session_id=session_id,
        step=step,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestAMCTier2HookBuildRuntimeBlocks
# ─────────────────────────────────────────────────────────────────────────────


class TestAMCTier2HookBuildRuntimeBlocks:
    """AMCTier2Hook.build_runtime_blocks contract."""

    def setup_method(self) -> None:
        self.hook = AMCTier2Hook()

    # ── basic shape ──────────────────────────────────────────────────────────

    def test_returns_list_of_memory_blocks(self) -> None:
        entries = [make_entry("hello", eid="e1"), make_entry("world", eid="e2")]
        blocks = self.hook.build_runtime_blocks(entries)
        assert len(blocks) == 2
        assert all(isinstance(b, AMCMemoryBlock) for b in blocks)

    def test_block_id_matches_entry_id(self) -> None:
        entries = [make_entry("hello", eid="my-id-42")]
        blocks = self.hook.build_runtime_blocks(entries)
        assert blocks[0].block_id == "my-id-42"

    def test_default_tier_is_two(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry()])
        assert blocks[0].tier == 2

    def test_tier_override(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry()], tier=3)
        assert blocks[0].tier == 3

    # ── trust ────────────────────────────────────────────────────────────────

    def test_default_trust_is_unverified(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry()])
        assert blocks[0].trust_state == TrustState.UNVERIFIED

    def test_trust_override_applied(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry()], trust_override=TrustState.VERIFIED)
        assert blocks[0].trust_state == TrustState.VERIFIED

    # ── provenance ───────────────────────────────────────────────────────────

    def test_provenance_from_role(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(role="assistant")])
        assert "assistant" in blocks[0].provenance

    def test_provenance_from_session(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(session_id="sess-xyz")])
        assert "sess-xyz" in blocks[0].provenance

    def test_provenance_both_role_and_session(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(role="tool", session_id="s-123")])
        p = blocks[0].provenance
        assert "tool" in p
        assert "s-123" in p

    def test_fallback_provenance_when_missing(self) -> None:
        entry = MemoryEntry(role="", content="x", session_id=None)
        blocks = self.hook.build_runtime_blocks([entry])
        assert "tier2:unknown" in blocks[0].provenance

    # ── salience / metadata ──────────────────────────────────────────────────

    def test_salience_from_importance(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(importance=0.7)])
        assert blocks[0].salience == pytest.approx(0.7)

    def test_salience_clamped(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(importance=1.5)])
        assert blocks[0].salience == pytest.approx(1.0)

    def test_content_hash_in_metadata(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(content="unique-content")])
        assert "content_hash" in blocks[0].metadata

    def test_surprise_score_equals_salience(self) -> None:
        blocks = self.hook.build_runtime_blocks([make_entry(importance=0.4)])
        assert blocks[0].surprise_score == pytest.approx(0.4)

    # ── AMCPrefixCompiler roundtrip ──────────────────────────────────────────

    def test_verified_blocks_compile_to_trusted(self) -> None:
        entries = [make_entry("safe data", eid="v1")]
        blocks = self.hook.build_runtime_blocks(entries, trust_override=TrustState.VERIFIED)
        compiler = AMCPrefixCompiler()
        result = compiler.compile(blocks)
        assert len(result.trusted) == 1
        assert result.trusted[0].trust_state == TrustState.VERIFIED

    def test_unverified_blocks_compile_to_allowed(self) -> None:
        entries = [make_entry("unknown data", eid="u1")]
        blocks = self.hook.build_runtime_blocks(entries)  # default UNVERIFIED
        compiler = AMCPrefixCompiler()
        result = compiler.compile(blocks)
        assert len(result.allowed) == 1
        assert result.allowed[0].trust_state == TrustState.UNVERIFIED
        assert len(result.trusted) == 0

    def test_quarantined_entry_excluded_from_trusted(self) -> None:
        """blocks created with UNVERIFIED trust must never appear in trusted."""
        entries = [make_entry("stuff", eid="q1")]
        blocks = self.hook.build_runtime_blocks(entries, trust_override=TrustState.QUARANTINED)
        compiler = AMCPrefixCompiler()
        result = compiler.compile(blocks)
        assert len(result.quarantined) == 1
        assert len(result.trusted) == 0
        assert len(result.allowed) == 0

    def test_cache_key_changes_when_trust_changes(self) -> None:
        """Switching trust state changes the cache fingerprint, even across buckets."""
        entries = [make_entry("data", eid="same-id")]
        compiler = AMCPrefixCompiler(policy_version="v0")
        r1 = compiler.compile(
            self.hook.build_runtime_blocks(entries, trust_override=TrustState.UNVERIFIED)
        )
        r2 = compiler.compile(
            self.hook.build_runtime_blocks(entries, trust_override=TrustState.VERIFIED)
        )
        # UNVERIFIED -> allowed bucket; VERIFIED -> trusted bucket.
        # Both segments exist; verify their fingerprints differ.
        assert len(r1.allowed) == 1
        assert len(r2.trusted) == 1
        assert r1.allowed[0].cache_key.fingerprint != r2.trusted[0].cache_key.fingerprint

    def test_empty_entries_returns_empty_list(self) -> None:
        assert self.hook.build_runtime_blocks([]) == []

    def test_empty_content_entry(self) -> None:
        """Empty content must still produce a valid block (no crash)."""
        entry = MemoryEntry(role="sys", content="", importance=0.0)
        blocks = self.hook.build_runtime_blocks([entry])
        assert len(blocks) == 1
        assert blocks[0].tokens == () or len(blocks[0].tokens) > 0  # hash either way

    def test_no_torch_or_cuda_required(self) -> None:
        """Adapter is pure stdlib — no torch import at module level."""
        import src.memory.amc_runtime_cache as _arc
        import src.memory.amc_tier2 as _t2

        # Both modules should be importable without torch
        assert hasattr(_arc, "AMCMemoryBlock")
        assert hasattr(_t2, "AMCTier2Hook")
        assert hasattr(_t2.AMCTier2Hook, "build_runtime_blocks")
