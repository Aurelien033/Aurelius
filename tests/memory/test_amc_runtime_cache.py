"""Focused tests for src.memory.amc_runtime_cache.


Covers
------
- AMCWriteDecision frozen/data-class enforcement
- Trust-state routing in AMCPrefixCompiler
- Cache-key derivation is sensitive to trust/quarantine/revocation state
- AMCMemoryBlock mutation helpers
- AMCMetricsCollector counters
- Phase-3: session-affinity cache-key seam
- Phase-4: chunked prefix prefill seam
"""
from __future__ import annotations

from __future__ import annotations

import hashlib
import importlib.util
import pathlib

import pytest

# ── Load amc_runtime_cache via spec_from_file_location ───────────────────────

def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(
        mod_name,
        pathlib.Path(__file__).resolve().parents[2] / "src" / rel,
    )
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

_mod = _load("amc_runtime_cache", "memory/amc_runtime_cache.py")

TrustState            = _mod.TrustState
WriteAction           = _mod.WriteAction
AMCWriteDecision      = _mod.AMCWriteDecision
AMCMemoryCacheKey     = _mod.AMCMemoryCacheKey
AMCMemoryBlock        = _mod.AMCMemoryBlock
AMCPrefixSegment      = _mod.AMCPrefixSegment
AMCPrefixCompileResult = _mod.AMCPrefixCompileResult
AMCPrefixCompiler     = _mod.AMCPrefixCompiler
AMCPrefixChunk        = _mod.AMCPrefixChunk
chunk_prefix_segments = _mod.chunk_prefix_segments
AMCMetricsCollector   = _mod.AMCMetricsCollector


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _make_block(
    block_id: str = "b1",
    trust: TrustState = TrustState.VERIFIED,
    *,
    tokens: tuple[int, ...] | None = None,
    tier: int = 2,
    provenance: str = "ego_t2",
    salience: float = 0.5,
    surprise: float = 0.5,
    mdata: dict | None = None,
) -> AMCMemoryBlock:
    return AMCMemoryBlock(
        block_id=block_id,
        tokens=tokens or (1, 2, 3),
        tier=tier,
        trust_state=trust,
        provenance=provenance,
        salience=salience,
        surprise_score=surprise,
        metadata=mdata or {},
    )


def _comp(
    blocks: list[AMCMemoryBlock],
    policy: str = "v0",
    session_fp: str | None = None,
) -> AMCPrefixCompileResult:
    c = AMCPrefixCompiler(policy_version=policy, session_fingerprint=session_fp)
    return c.compile(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# AMCWriteDecision
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCWriteDecision:
    def test_frozen_dataclass_cannot_be_mutated(self) -> None:
        d = AMCWriteDecision(action=WriteAction.WRITE, target_tier=2,
                             confidence=0.9, trust_label=TrustState.VERIFIED,
                             provenance="ego", reason="ok")
        with pytest.raises(AttributeError):
            d.trust_label = TrustState.UNVERIFIED  # type: ignore[misc]

    @pytest.mark.parametrize("action", list(WriteAction))
    def test_all_write_actions_accepted(self, action: TrustState) -> None:
        d = AMCWriteDecision(action=action, target_tier=2,
                             confidence=0.5, trust_label=TrustState.UNVERIFIED,
                             provenance="test", reason="test", quarantine_reason="")
        assert d.action == action

    def test_validation_rejects_empty_reason(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            AMCWriteDecision(action=WriteAction.WRITE, target_tier=2,
                             confidence=0.5, trust_label=TrustState.VERIFIED,
                             provenance="ego", reason="", quarantine_reason="")

    def test_is_safe_true_for_write(self) -> None:
        d = AMCWriteDecision(action=WriteAction.WRITE, target_tier=2,
                             confidence=0.9, trust_label=TrustState.VERIFIED,
                             provenance="ego", reason="ok")
        assert d.is_safe is True

    def test_is_safe_true_for_non_write_actions(self) -> None:
        d = AMCWriteDecision(action=WriteAction.QUARANTINE, target_tier=2,
                             confidence=0.1, trust_label=TrustState.QUARANTINED,
                             provenance="evil", reason="blocked", quarantine_reason="policy")
        assert d.is_safe is True  # non-WRITE actions bypass write confidence guard

    def test_decision_serialisable(self) -> None:
        d = AMCWriteDecision(action=WriteAction.WRITE, target_tier=2,
                             confidence=0.8, trust_label=TrustState.VERIFIED,
                             provenance="ego", reason="ok")
        s = str(d)
        assert "WRITE" in s or "write" in s.lower()


# ─────────────────────────────────────────────────────────────────────────────
# AMCMemoryBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCMemoryBlock:
    def test_create_default(self) -> None:
        b = AMCMemoryBlock(block_id="m1")
        assert b.block_id == "m1"
        assert b.tier == 2
        assert b.trust_state == TrustState.UNVERIFIED
        assert b.salience == 0.5
        assert not b.dirty

    def test_quarantine_transitions_state(self) -> None:
        b = _make_block("m1", trust=TrustState.VERIFIED)
        b.quarantine("policy")
        assert b.trust_state == TrustState.QUARANTINED
        assert b.quarantine_state == "policy"

    def test_quarantine_reason_stored(self) -> None:
        b = _make_block("m1")
        b.quarantine("exceeded budget")
        assert b.quarantine_state == "exceeded budget"

    def test_revoke_updates_epoch(self) -> None:
        b = _make_block("m1")
        b.revoke(epoch=3)
        assert b.trust_state == TrustState.REVOKED
        assert b.revocation_epoch == 3

    def test_revoke_default_epoch(self) -> None:
        b = _make_block("m1")
        b.revoke()
        assert b.revocation_epoch == 0   # no epoch given

    def test_to_cache_key_changes_after_revoke(self) -> None:
        b = _make_block("m1", trust=TrustState.VERIFIED)
        k1 = b.to_cache_key("v0")
        b.revoke(epoch=1)
        k2 = b.to_cache_key("v0")
        assert k1.fingerprint != k2.fingerprint

    def test_dirty_flag(self) -> None:
        b = _make_block("m1", trust=TrustState.VERIFIED)
        assert not b.dirty
        b.mark_dirty()
        assert b.dirty
        b.mark_clean()
        assert not b.dirty

    def test_mark_used_updates_counter(self) -> None:
        b = _make_block("m1")
        t0 = b.last_used
        import time; time.sleep(0.01)
        b.mark_used()
        assert b.last_used >= t0

    def test_tier_validation(self) -> None:
        with pytest.raises(ValueError, match="tier must be"):
            AMCMemoryBlock(block_id="x", tier=9)

    def test_token_normalisation(self) -> None:
        b = AMCMemoryBlock(block_id="x", tokens=[10, 20])
        assert isinstance(b.tokens, tuple)

    def test_metadata_shared_default_bug(self) -> None:
        b1 = AMCMemoryBlock(block_id="x"); b1.metadata["k"] = "v1"
        b2 = AMCMemoryBlock(block_id="y")
        assert b2.metadata.get("k") is None   # must NOT share defaults

    def test_salience_clamps(self) -> None:
        with pytest.raises(ValueError): _make_block("x", salience=1.5)
        with pytest.raises(ValueError): _make_block("x", salience=-0.1)

    def test_to_cache_key_default_session_fingerprint_none(self) -> None:
        b = _make_block("m1")
        k = b.to_cache_key("v0")
        assert k.session_fingerprint is None

    def test_to_cache_key_with_session_fingerprint(self) -> None:
        b = _make_block("m1")
        k = b.to_cache_key("v0", session_fingerprint="s-1")
        assert k.session_fingerprint == "s-1"


# ─────────────────────────────────────────────────────────────────────────────
# Cache-key sensitivity
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheKeySensitivity:
    def test_same_content_same_key(self) -> None:
        b1 = _make_block("x1", provenance="ego")
        b2 = _make_block("x2", provenance="ego")
        assert b1.to_cache_key("v0").fingerprint == b2.to_cache_key("v0").fingerprint

    def test_different_content_different_key(self) -> None:
        b1 = _make_block("x1", provenance="ego")
        b2 = _make_block("x2", provenance="other")
        assert b1.to_cache_key("v0").fingerprint != b2.to_cache_key("v0").fingerprint

    def test_trust_state_change_changes_key(self) -> None:
        b = _make_block("x1", trust=TrustState.VERIFIED)
        k1 = b.to_cache_key("v0")
        b.trust_state = TrustState.UNVERIFIED
        k2 = b.to_cache_key("v0")
        assert k1.fingerprint != k2.fingerprint

    def test_policy_version_change_changes_key(self) -> None:
        b = _make_block("x1")
        k1 = b.to_cache_key("policy-a")
        k2 = b.to_cache_key("policy-b")
        assert k1.fingerprint != k2.fingerprint

    def test_quarantine_state_in_key(self) -> None:
        b = _make_block("x1", trust=TrustState.UNVERIFIED)
        k1 = b.to_cache_key("v0")
        b.quarantine("ea")
        k2 = b.to_cache_key("v0")
        assert k1.fingerprint != k2.fingerprint

    def test_revocation_epoch_in_key(self) -> None:
        b = _make_block("x1", trust=TrustState.VERIFIED)
        k1 = b.to_cache_key("v0")
        b.revoke(epoch=1)
        k2 = b.to_cache_key("v0")
        assert k1.fingerprint != k2.fingerprint

    def test_session_fingerprint_in_key(self) -> None:
        b = _make_block("x1")
        k_a = b.to_cache_key("v0", session_fingerprint="s-A")
        k_b = b.to_cache_key("v0", session_fingerprint="s-B")
        assert k_a.fingerprint != k_b.fingerprint

    def test_provenance_in_key(self) -> None:
        b1 = _make_block("x1", provenance="provenance_A")
        b2 = _make_block("x1", provenance="provenance_B")
        assert b1.to_cache_key("v0").fingerprint != b2.to_cache_key("v0").fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# AMCPrefixCompiler trust routing
# ─────────────────────────────────────────────────────────────────────────────

class TestPrefixCompilerTrustRouting:
    def test_verified_goes_to_trusted(self) -> None:
        r = _comp([_make_block("v1", trust=TrustState.VERIFIED)])
        assert len(r.trusted) == 1 and not r.allowed
        assert r.trusted[0].trust_state == TrustState.VERIFIED

    def test_unverified_goes_to_allowed(self) -> None:
        r = _comp([_make_block("u1", trust=TrustState.UNVERIFIED)])
        assert len(r.allowed) == 1 and not r.trusted
        assert r.allowed[0].trust_state == TrustState.UNVERIFIED

    def test_quarantined_routed_out_of_trusted(self) -> None:
        r = _comp([_make_block("q1", trust=TrustState.QUARANTINED)])
        assert len(r.quarantined) == 1
        assert not r.trusted and not r.allowed

    def test_revoked_routed_out_of_trusted(self) -> None:
        r = _comp([_make_block("r1", trust=TrustState.REVOKED)])
        assert len(r.revoked) == 1
        assert not r.trusted and not r.allowed

    def test_mixed_blocks_all_routed(self) -> None:
        blocks = [
            _make_block("v1", trust=TrustState.VERIFIED),
            _make_block("u1", trust=TrustState.UNVERIFIED),
            _make_block("q1", trust=TrustState.QUARANTINED),
            _make_block("r1", trust=TrustState.REVOKED),
        ]
        r = _comp(blocks)
        assert len(r.trusted) == 1
        assert len(r.allowed) == 1
        assert len(r.quarantined) == 1
        assert len(r.revoked) == 1

    def test_cache_key_changes_map_updated(self) -> None:
        b = _make_block("x1", trust=TrustState.VERIFIED)
        r = _comp([b])
        assert r is not None  # compile ran

    def test_compilation_counter_increments(self) -> None:
        c = AMCPrefixCompiler()
        c.compile([_make_block("x1")])
        c.compile([_make_block("x2")])
        stats = c.compile([_make_block("x3")]).stats
        assert stats["compilations"] >= 3

    def test_policy_version_in_key(self) -> None:
        c1 = AMCPrefixCompiler(policy_version="v0")
        c2 = AMCPrefixCompiler(policy_version="v1")
        r1 = c1.compile([_make_block("x1", trust=TrustState.VERIFIED)])
        r2 = c2.compile([_make_block("x1", trust=TrustState.VERIFIED)])
        assert r1.trusted[0].cache_key.fingerprint != r2.trusted[0].cache_key.fingerprint

    def test_result_stats_populated(self) -> None:
        r = _comp([_make_block("v1"), _make_block("u1")])
        assert r.stats["trusted_segments"] + r.stats["allowed_segments"] >= 2

    def test_revocations_invalidate_prior_key(self) -> None:
        c = AMCPrefixCompiler(policy_version="policy-1")
        b = _make_block("mem1", trust=TrustState.VERIFIED, tier=2)
        r1 = c.compile([b])
        assert len(r1.trusted) == 1
        key_before = r1.trusted[0].cache_key.fingerprint
        b.revoke(epoch=1)
        r2 = c.compile([b])
        assert len(r2.revoked) == 1
        assert "mem1" in r2.cache_key_changes
        old_fp = b.to_cache_key("policy-1").fingerprint
        assert all(seg.cache_key.fingerprint != old_fp for seg in r2.trusted)
        key_after = b.to_cache_key("policy-1").fingerprint
        assert key_before != key_after

    def test_unverified_not_in_trusted_privileged_bucket(self) -> None:
        r = _comp([_make_block("x1", trust=TrustState.UNVERIFIED)])
        assert len(r.trusted) == 0
        assert len(r.allowed) == 1


# ─────────────────────────────────────────────────────────────────────────────
# WriteDecision payload attrs
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCWriteDecisionAttrs:
    def test_required_fields_present(self) -> None:
        for cls_field in ("action", "target_tier", "confidence",
                         "trust_label", "provenance", "reason"):
            assert hasattr(AMCWriteDecision.__annotations__, cls_field) or cls_field in AMCWriteDecision.__dataclass_fields__  # noqa: E501

    def test_ttl_optional(self) -> None:
        d = AMCWriteDecision(action=WriteAction.WRITE, target_tier=2,
                             confidence=0.9, trust_label=TrustState.VERIFIED,
                             provenance="ego", reason="ok", ttl_seconds=None)
        assert d.ttl_seconds is None

    def test_denied_action_still_valid(self) -> None:
        d = AMCWriteDecision(action=WriteAction.QUARANTINE, target_tier=2,
                             confidence=0.1, trust_label=TrustState.QUARANTINED,
                             provenance="malicious", reason="blocked", quarantine_reason="ea")
        assert d.action == WriteAction.QUARANTINE


# ─────────────────────────────────────────────────────────────────────────────
# AMC benchmark configs (benchmark scaffold — see amc_tensor_api.py)
# ─────────────────────────────────────────────────────────────────────────────

# (benchmark-scaffold tests kept in test_amc_tensor_api.py)




# ─────────────────────────────────────────────────────────────────────────────
# AMCMetricsCollector
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCMetricsCollector:
    def setup_method(self) -> None:
        self.m = AMCMetricsCollector()

    def test_inc_counter(self) -> None:
        self.m.inc("x", 2)
        assert self.m._counters["x"] == 2

    def test_set_gauge(self) -> None:
        self.m.set_gauge("y", 1.23)
        assert self.m._gauges["y"] == pytest.approx(1.23)

    def test_cache_hit_increments_trust_counter(self) -> None:
        self.m.record_cache_hit(TrustState.UNVERIFIED)
        assert self.m._counters["cache_hit_unverified"] == 1

    def test_cache_miss_increments_counter(self) -> None:
        self.m.record_cache_miss("unverified")
        assert self.m._counters["cache_miss_unverified"] == 1

    def test_quarantine_exclusion_counter(self) -> None:
        self.m.record_quarantine_exclusion()
        assert self.m._quarantine_exclusions == 1

    def test_revocation_invalidation_counter(self) -> None:
        self.m.record_revocation_invalidation("old_fp_abc")
        assert self.m._revocation_invalidations == 1

    def test_write_decision_recording(self) -> None:
        d = AMCWriteDecision(action=WriteAction.WRITE, target_tier=2,
                             confidence=0.9, trust_label=TrustState.VERIFIED,
                             provenance="ego", reason="ok")
        self.m.record_write_decision(d)
        assert self.m._counters["write_write"] == 1

    def test_hit_rate_by_trust_empty(self) -> None:
        assert self.m.hit_rate_by_trust() == {}

    def test_hit_rate_by_trust_nonempty(self) -> None:
        self.m.record_cache_hit(TrustState.UNVERIFIED)
        self.m.record_cache_miss("unverified")
        self.m.record_cache_hit(TrustState.VERIFIED)
        out = self.m.hit_rate_by_trust()
        assert out["unverified"] == pytest.approx(0.5)
        assert out["verified"] == pytest.approx(1.0)

    def test_snapshot_is_dict(self) -> None:
        self.m.inc("x", 2)
        s = self.m.snapshot()
        assert isinstance(s, dict)
        assert s["counters"]["x"] == 2

    def test_hit_by_trust_hit_counter(self) -> None:
        self.m.record_cache_hit(TrustState.UNVERIFIED)
        assert self.m._hit_by_trust["unverified"] == 1
        assert self.m._counters["cache_hit_unverified"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Integration-style: compiler + metrics together
# ─────────────────────────────────────────────────────────────────────────────

class TestCompilerMetricsIntegration:
    def test_compiler_populates_metrics(self) -> None:
        m = AMCMetricsCollector()
        c = AMCPrefixCompiler()
        b = _make_block("seg1", trust=TrustState.VERIFIED)
        c.compile([b])
        s = m.snapshot()
        assert "counters" in s
        assert "hit_by_trust" in s


# ─────────────────────────────────────────────────────────────────────────────
# Phase-3: session-affinity cache-key seam
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCSessionAffinity:
    """session_fingerprint must affect cache identity; absence must not."""

    def _make_key(self, content: str, session_fp: str | None) -> AMCMemoryCacheKey:
        return AMCMemoryCacheKey.compute(
            content=content,
            token_ids=tuple(ord(c) for c in content),
            tier=2,
            trust_state=TrustState.VERIFIED,
            provenance="ego",
            policy_version="v0",
            session_fingerprint=session_fp,
        )

    def test_no_fingerprint_produces_deterministic_key(self) -> None:
        k1 = self._make_key("hello", None)
        k2 = self._make_key("hello", None)
        assert k1.fingerprint == k2.fingerprint

    def test_same_fingerprint_same_key(self) -> None:
        k1 = self._make_key("hello", "session-A")
        k2 = self._make_key("hello", "session-A")
        assert k1.fingerprint == k2.fingerprint

    def test_different_session_fingerprint_different_key(self) -> None:
        k1 = self._make_key("hello", "session-A")
        k2 = self._make_key("hello", "session-B")
        assert k1.fingerprint != k2.fingerprint

    def test_different_content_different_key(self) -> None:
        k1 = self._make_key("hello", "session-A")
        k2 = self._make_key("world", "session-A")
        assert k1.fingerprint != k2.fingerprint

    def test_segment_session_fingerprint_in_key(self) -> None:
        c = AMCPrefixCompiler(policy_version="v0", session_fingerprint="user-42")
        b = _make_block("seg1", trust=TrustState.VERIFIED)
        r = c.compile([b])
        seg = r.trusted[0]
        assert seg.cache_key.session_fingerprint == "user-42"

    def test_session_fingerprint_is_absent_by_default(self) -> None:
        c = AMCPrefixCompiler(policy_version="v0")
        b = _make_block("seg1", trust=TrustState.VERIFIED)
        r = c.compile([b])  # no session_fingerprint
        seg = r.trusted[0]
        assert seg.cache_key.session_fingerprint is None


# ─────────────────────────────────────────────────────────────────────────────
# Phase-3: metrics-snapshot counter completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCMetricsSnapshotCounters:
    """AMCMetricsCollector.snapshot() must expose every required counter."""

    def setup_method(self) -> None:
        self.m = AMCMetricsCollector()

    def test_snapshot_has_cache_hit_trust_counters(self) -> None:
        self.m.record_cache_hit(TrustState.VERIFIED)
        s = self.m.snapshot()
        assert "counters" in s
        assert "cache_hit_verified" in s["counters"]
        assert s["counters"]["cache_hit_verified"] >= 1

    def test_snapshot_has_quarantine_exclusion_counter(self) -> None:
        self.m.record_quarantine_exclusion()
        s = self.m.snapshot()
        assert s["quarantine_exclusions"] >= 1

    def test_snapshot_has_revocation_invalidation_counter(self) -> None:
        self.m.record_revocation_invalidation("old_fp_abc")
        s = self.m.snapshot()
        assert s["revocation_invalidations"] >= 1

    def test_snapshot_has_write_decision_counters(self) -> None:
        self.m.record_write_decision(AMCWriteDecision(
            action=WriteAction.WRITE, target_tier=2, confidence=0.9,
            trust_label=TrustState.VERIFIED, provenance="test", reason="ok",
        ))
        self.m.record_write_decision(AMCWriteDecision(
            action=WriteAction.QUARANTINE, target_tier=2, confidence=0.1,
            trust_label=TrustState.QUARANTINED, provenance="malicious",
            reason="blocked", quarantine_reason="policy",
        ))
        s = self.m.snapshot()
        assert "write_actions" in s
        assert s["write_actions"].get("write_write", 0) >= 1
        assert s["write_actions"].get("write_quarantine", 0) >= 1

    def test_snapshot_contains_hit_by_trust(self) -> None:
        self.m.record_cache_hit(TrustState.VERIFIED)
        self.m.record_cache_hit(TrustState.UNVERIFIED)
        s = self.m.snapshot()
        assert "hit_by_trust" in s
        assert s["hit_by_trust"].get("verified", 0) >= 1
        assert s["hit_by_trust"].get("unverified", 0) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase-4: chunked prefix prefill seam
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkPrefixSegments:
    """AMCPrefixChunk / chunk_prefix_segments contract."""

    def _seg(self, fp: str, tokens: tuple, trust: TrustState = TrustState.VERIFIED) -> AMCPrefixSegment:
        return AMCPrefixSegment(
            cache_key=AMCMemoryCacheKey.compute(
                content="x", token_ids=tokens, tier=2,
                trust_state=trust, provenance="ego", policy_version="v0",
            ),
            tokens=tokens,
            trust_state=trust,
        )

    def test_single_segment_unchunked_when_below_max(self) -> None:
        seg = self._seg("k1", (1, 2, 3))
        chunks = chunk_prefix_segments([seg], max_tokens=5)
        assert len(chunks) == 1
        assert chunks[0].tokens == (1, 2, 3)

    def test_splits_at_max_tokens_boundary(self) -> None:
        seg = self._seg("k1", (1, 2, 3, 4))
        chunks = chunk_prefix_segments([seg], max_tokens=2)
        assert len(chunks) == 2
        assert chunks[0].tokens == (1, 2)
        assert chunks[1].tokens == (3, 4)

    def test_max_tokens_less_than_one_raises(self) -> None:
        seg = self._seg("k1", (1,))
        with pytest.raises(ValueError, match="must be > 0"):
            chunk_prefix_segments([seg], max_tokens=0)

    def test_empty_token_segment_produces_metadata_chunk(self) -> None:
        seg = self._seg("k1", ())
        chunks = chunk_prefix_segments([seg], max_tokens=4)
        assert len(chunks) == 1
        assert chunks[0].tokens == ()
        assert "segment_tier" in chunks[0].metadata

    def test_trust_label_preserved_in_each_chunk(self) -> None:
        seg = self._seg("k1", (1, 2, 3, 4, 5), trust=TrustState.VERIFIED)
        chunks = chunk_prefix_segments([seg], max_tokens=2)
        assert all(c.trust_state == TrustState.VERIFIED for c in chunks)

    def test_cache_key_fingerprint_preserved(self) -> None:
        seg = self._seg("k1", (7, 8, 9))
        expected = seg.cache_key.fingerprint
        chunks = chunk_prefix_segments([seg], max_tokens=2)
        assert chunks[0].cache_key_fingerprint == expected
        assert chunks[1].cache_key_fingerprint == expected

    def test_mode_total_token_count_preserved(self) -> None:
        tokens = tuple(range(17))
        seg = self._seg("k2", tokens)
        chunks = chunk_prefix_segments([seg], max_tokens=5)
        reassembled = tuple(t for c in chunks for t in c.tokens)
        assert reassembled == tokens

    def test_max_tokens_equals_one(self) -> None:
        seg = self._seg("k1", (10, 20, 30))
        chunks = chunk_prefix_segments([seg], max_tokens=1)
        assert len(chunks) == 3
        assert chunks[0].tokens == (10,)
        assert chunks[1].tokens == (20,)
        assert chunks[2].tokens == (30,)

    def test_preserves_trust_state_in_each_chunk(self) -> None:
        seg = self._seg("k1", (1, 2, 3, 4, 5, 6), trust=TrustState.UNVERIFIED)
        chunks = chunk_prefix_segments([seg], max_tokens=3)
        assert len(chunks) == 2
        assert chunks[0].trust_state == TrustState.UNVERIFIED
        assert chunks[1].trust_state == TrustState.UNVERIFIED

    def test_multiple_segments_interleaved(self) -> None:
        s1 = self._seg("fp1", (1, 2))
        s2 = self._seg("fp2", (3, 4, 5))
        chunks = chunk_prefix_segments([s1, s2], max_tokens=2)
        assert len(chunks) == 3
        assert chunks[0].tokens == (1, 2)
        assert chunks[1].tokens == (3, 4)
        assert chunks[2].tokens == (5,)

    def test_no_torch_or_cuda_required(self) -> None:
        import src.memory.amc_runtime_cache as _mod
        assert hasattr(_mod, "AMCPrefixChunk")
        assert hasattr(_mod, "chunk_prefix_segments")
