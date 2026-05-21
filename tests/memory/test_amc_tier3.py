"""Tests for src.memory.amc_tier3 — AMC Tier 3 long-term store."""

from __future__ import annotations

import time

import pytest

from src.memory.amc_tier3 import (
    AMCTier3Config,
    AMCTier3Hook,
    DecayPolicy,
    Tier3Entry,
    TrustLevel,
)

# ── DecayPolicy ───────────────────────────────────────────────────────────────


class TestDecayPolicy:
    def test_default_values(self):
        dp = DecayPolicy()
        assert dp.half_life_seconds == 86400.0
        assert dp.max_age_seconds == 2592000.0

    def test_custom_values(self):
        dp = DecayPolicy(half_life_seconds=3600.0, max_age_seconds=86400.0)
        assert dp.half_life_seconds == 3600.0
        assert dp.max_age_seconds == 86400.0


# ── Tier3Entry ────────────────────────────────────────────────────────────────


class TestTier3Entry:
    def test_basic_fields(self):
        now = time.monotonic()
        e = Tier3Entry(
            key="k",
            value="v",
            trust_level=TrustLevel.TRUSTED,
            confidence=0.9,
            created_at=now,
        )
        assert e.key == "k"
        assert e.value == "v"
        assert e.trust_level is TrustLevel.TRUSTED
        assert e.confidence == 0.9

    def test_default_unverified(self):
        e = Tier3Entry(key="k", value="v")
        assert e.trust_level is TrustLevel.UNVERIFIED
        assert e.confidence == 1.0

    def test_decayed_importance_quarantined_is_zero(self):
        now = time.monotonic()
        e = Tier3Entry(
            key="k",
            value="v",
            trust_level=TrustLevel.QUARANTINED,
            confidence=1.0,
            created_at=now,
        )
        assert e.decayed_importance(now) == 0.0

    def test_decayed_importance_unverified_decays(self):
        now = time.monotonic()
        past = now - 86400  # 1 day ago
        e = Tier3Entry(key="k", value="v", confidence=1.0, created_at=past)
        assert e.decayed_importance(now) < 1.0

    def test_decayed_importance_fresh_equals_confidence(self):
        now = time.monotonic()
        e = Tier3Entry(key="k", value="v", confidence=0.7, created_at=now)
        assert e.decayed_importance(now) == pytest.approx(0.7, abs=1e-6)

    def test_is_expired_after_max_age(self):
        now = time.monotonic()
        past = now - 2600000.0  # approximately 30 days
        e = Tier3Entry(key="k", value="v", created_at=past)
        assert e.is_expired(now) is True

    def test_is_expired_fresh(self):
        now = time.monotonic()
        e = Tier3Entry(key="k", value="v", created_at=now)
        assert e.is_expired(now) is False

    def test_verify_promotes_trusted_for_high_confidence(self):
        e = Tier3Entry(key="k", value="v", trust_level=TrustLevel.UNVERIFIED)
        e.verify(confidence=0.85)
        assert e.trust_level is TrustLevel.TRUSTED
        assert e.confidence == 0.85
        assert e.last_verified_at is not None

    def test_verify_stays_unverified_for_low_confidence(self):
        e = Tier3Entry(key="k", value="v")
        e.verify(confidence=0.35)
        assert e.trust_level is TrustLevel.UNVERIFIED

    def test_revoke_sets_revoked(self):
        e = Tier3Entry(key="k", value="v")
        e.revoke()
        assert e.trust_level is TrustLevel.REVOKED


# ── AMCTier3Config ────────────────────────────────────────────────────────────


class TestAMCTier3Config:
    def test_defaults(self):
        c = AMCTier3Config()
        assert c.min_confidence == 0.6
        assert c.quarantine_threshold == 0.3
        assert c.max_entries == 10000

    def test_rejects_invalid_min_confidence(self):
        with pytest.raises(ValueError, match="min_confidence"):
            AMCTier3Config(min_confidence=1.5)

    def test_rejects_invalid_quarantine_threshold(self):
        with pytest.raises(ValueError, match="quarantine_threshold"):
            AMCTier3Config(quarantine_threshold=-0.1)

    def test_rejects_zero_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            AMCTier3Config(max_entries=0)


# ── AMCTier3Hook — promote & quarantine ──────────────────────────────────────


class TestPromoteAndQuarantine:
    def setup_method(self):
        self.hook = AMCTier3Hook(AMCTier3Config(max_entries=100, quarantine_threshold=0.3))

    def test_promote_trusted_entry(self):
        entry = self.hook.promote(key="fact:name", value="Ada", confidence=0.9)
        assert entry is not None
        assert entry.trust_level is TrustLevel.TRUSTED
        stats = self.hook.stats()
        assert stats.trusted == 1  # promoted with confidence >= 0.7 → auto-trusted

    def test_promote_unverified_entry(self):
        entry = self.hook.promote(key="fact:city", value="Paris", confidence=0.6)
        assert entry is not None
        assert entry.trust_level is TrustLevel.UNVERIFIED

    def test_promote_below_quarantine_threshold_returns_quarantined(self):
        entry = self.hook.promote(key="untrusted", value="false", confidence=0.1)
        assert entry is not None
        assert entry.trust_level is TrustLevel.QUARANTINED
        assert "untrusted" in self.hook._quarantine

    def test_quarantine_directly(self):
        entry = self.hook.quarantine(key="mystery", value="???", confidence=0.0)
        assert entry.trust_level is TrustLevel.QUARANTINED
        assert "mystery" in self.hook._quarantine

    def test_promote_stored_in_active(self):
        self.hook.promote(key="active", value="yes")
        assert "active" in self.hook._store

    def test_promote_returns_none_when_key_empty(self):
        # Empty key doesn't raise; it goes through
        entry = self.hook.promote(key="", value="v")
        assert entry is not None


# ── AMCTier3Hook — consolidate ───────────────────────────────────────────────


class TestConsolidation:
    def setup_method(self):
        self.hook = AMCTier3Hook(AMCTier3Config(quarantine_threshold=0.3, min_confidence=0.5))

    def test_promotes_quarantined_above_threshold(self):
        self.hook.quarantine(key="q1", value="val", confidence=0.7)
        result = self.hook.consolidate()
        assert result.promoted == 1
        assert "q1" in self.hook._store
        assert "q1" not in self.hook._quarantine

    def test_keeps_quarantined_below_min_confidence(self):
        self.hook.quarantine(key="q_low", value="val", confidence=0.2)
        result = self.hook.consolidate()
        assert result.promoted == 0
        assert "q_low" in self.hook._quarantine

    def test_prunes_expired_quarantined(self):
        past = time.monotonic() - 2600000.0
        entry = Tier3Entry(
            key="old_q",
            value="gone",
            trust_level=TrustLevel.QUARANTINED,
            created_at=past,
        )
        self.hook._quarantine["old_q"] = entry
        result = self.hook.consolidate()
        assert result.expired_pruned >= 1
        assert "old_q" not in self.hook._quarantine

    def test_prunes_expired_active(self):
        past = time.monotonic() - 2600000.0
        entry = Tier3Entry(
            key="old_active",
            value="gone",
            trust_level=TrustLevel.TRUSTED,
            created_at=past,
        )
        self.hook._store["old_active"] = entry
        result = self.hook.consolidate()
        assert result.expired_pruned >= 1
        assert "old_active" not in self.hook._store

    def test_result_is_dict_serializable(self):
        self.hook.quarantine(key="x", value="y", confidence=0.6)
        result = self.hook.consolidate()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "promoted" in d
        assert "quarantined" in d


# ── AMCTier3Hook — prioritize ────────────────────────────────────────────────


class TestPrioritize:
    def setup_method(self):
        self.hook = AMCTier3Hook(AMCTier3Config())

    def test_returns_trusted_over_unverified(self):
        self.hook.promote(key="unv", value="low", confidence=0.55)
        self.hook.promote(
            key="trusted", value="high", confidence=0.95, trust_level=TrustLevel.TRUSTED
        )
        top = self.hook.prioritize(limit=2)
        keys = [e.key for e in top]
        assert "trusted" in keys
        assert "unv" in keys

    def test_excludes_quarantined(self):
        self.hook.promote(key="active", value="safe", confidence=0.7)
        self.hook.quarantine(key="forbidden", value="bad", confidence=0.1)
        results = self.hook.prioritize(limit=10)
        keys = [e.key for e in results]
        assert "active" in keys
        assert "forbidden" not in keys

    def test_excludes_revoked(self):
        self.hook.promote(key="k", value="v")
        entry = self.hook._store["k"]
        entry.revoke()
        assert self.hook.prioritize() == []

    def test_respects_limit(self):
        for i in range(10):
            self.hook.promote(key=f"k{i}", value=f"v{i}", confidence=0.9)
        results = self.hook.prioritize(limit=3)
        assert len(results) == 3

    def test_sort_by_decayed_importance(self):
        now = time.monotonic()
        past = now - 86400  # 1 day decayed
        entry_old = self.hook.promote(key="old", value="x", confidence=1.0)
        entry_new = self.hook.promote(key="new", value="y", confidence=0.9)
        assert entry_old is not None and entry_new is not None
        # Mutate created_at after promotion to simulate older entry
        entry_old.created_at = past
        entry_new.created_at = now
        results = self.hook.prioritize()
        assert results[0].key == "new"  # higher decayed importance


# ── AMCTier3Hook — verify_and_promote ───────────────────────────────────────


class TestVerifyAndPromote:
    def setup_method(self):
        self.hook = AMCTier3Hook(AMCTier3Config())

    def test_verifies_and_promotes_quarantined_entry(self):
        self.hook.quarantine(key="to_verify", value="truth", confidence=0.4)
        entry = self.hook.verify_and_promote("to_verify", confidence=1.0)
        assert entry is not None
        assert entry.trust_level is TrustLevel.TRUSTED
        assert "to_verify" in self.hook._store
        assert "to_verify" not in self.hook._quarantine

    def test_returns_none_for_nonexistent_key(self):
        result = self.hook.verify_and_promote("ghost")
        assert result is None


# ── AMCTier3Hook — stats ─────────────────────────────────────────────────────


class TestTier3Stats:
    def setup_method(self):
        self.hook = AMCTier3Hook(AMCTier3Config())

    def test_empty_stats(self):
        s = self.hook.stats()
        assert s.total_entries == 0
        assert s.trusted == 0
        assert s.quarantined == 0

    def test_tracks_trusted_and_quarantined(self):
        self.hook.promote(key="t1", value="v", trust_level=TrustLevel.TRUSTED)
        self.hook.quarantine(key="q1", value="v", confidence=0.1)
        s = self.hook.stats()
        assert s.trusted == 1
        assert s.unverified == 0
        assert s.quarantined == 1

    def test_stats_to_dict(self):
        self.hook.promote(key="t1", value="v", trust_level=TrustLevel.TRUSTED)
        d = self.hook.stats().to_dict()
        assert "trusted" in d
        assert "avg_confidence" in d


# ── AMCTier3Config validation ────────────────────────────────────────────────


class TestConfigValidation:
    def test_quarantine_threshold_above_min_confidence_rejected(self):
        with pytest.raises(ValueError, match="quarantine_threshold"):
            AMCTier3Config(quarantine_threshold=0.9, min_confidence=0.8)
