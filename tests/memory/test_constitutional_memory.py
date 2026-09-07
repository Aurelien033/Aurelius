"""Tests for ConstitutionalMemory — all tamper attempts must be BLOCKED."""

from __future__ import annotations

import pytest

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState
from src.memory.amc_tier3 import AMCTier3Hook, TrustLevel
from src.memory.constitutional_memory import DEFAULT_PRINCIPLES, ConstitutionalMemory


@pytest.fixture
def tier3() -> AMCTier3Hook:
    return AMCTier3Hook()


@pytest.fixture
def constitutional(tier3: AMCTier3Hook) -> ConstitutionalMemory:
    return ConstitutionalMemory(tier3)


def test_injects_all_default_principles(constitutional: ConstitutionalMemory) -> None:
    assert constitutional.principle_count == len(DEFAULT_PRINCIPLES)


def test_all_principles_are_trusted(
    constitutional: ConstitutionalMemory, tier3: AMCTier3Hook
) -> None:
    for index in range(len(DEFAULT_PRINCIPLES)):
        key = f"constitutional:{index}"
        entry = tier3._store.get(key)
        assert entry is not None, f"{key} not in store"
        assert entry.trust_level == TrustLevel.TRUSTED


def test_integrity_verify_passes_when_untouched(constitutional: ConstitutionalMemory) -> None:
    valid, issues = constitutional.verify_integrity()
    assert valid, f"integrity issues: {issues}"


def test_integrity_fails_on_content_tamper(
    constitutional: ConstitutionalMemory, tier3: AMCTier3Hook
) -> None:
    entry = tier3._store["constitutional:0"]
    entry.value = "tampered"
    valid, issues = constitutional.verify_integrity()
    assert not valid
    assert any("modified" in issue for issue in issues)


def test_integrity_fails_on_missing(
    constitutional: ConstitutionalMemory, tier3: AMCTier3Hook
) -> None:
    del tier3._store["constitutional:3"]
    valid, issues = constitutional.verify_integrity()
    assert not valid
    assert any("missing" in issue for issue in issues)


def test_attempt_revoke_is_blocked(constitutional: ConstitutionalMemory) -> None:
    assert constitutional.attempt_revoke("constitutional:0", source="test")
    entry = constitutional.tier3._store.get("constitutional:0")
    assert entry is not None
    assert entry.trust_level == TrustLevel.TRUSTED
    assert constitutional.violation_count() == 1


def test_attempt_delete_is_blocked(
    constitutional: ConstitutionalMemory, tier3: AMCTier3Hook
) -> None:
    assert constitutional.attempt_delete("constitutional:0", source="test")
    assert "constitutional:0" in tier3._store
    assert constitutional.violation_count() == 1


def test_attempt_quarantine_is_blocked(constitutional: ConstitutionalMemory) -> None:
    assert constitutional.attempt_quarantine("constitutional:0", source="test")
    entry = constitutional.tier3._store.get("constitutional:0")
    assert entry is not None
    assert entry.trust_level != TrustLevel.QUARANTINED
    assert constitutional.violation_count() == 1


def test_attempt_modify_is_blocked(
    constitutional: ConstitutionalMemory, tier3: AMCTier3Hook
) -> None:
    assert constitutional.attempt_modify("constitutional:0", "new content", source="attacker")
    assert tier3._store["constitutional:0"].value == DEFAULT_PRINCIPLES[0]
    assert constitutional.violation_count() == 1


def test_retrieve_all_returns_all_principles(constitutional: ConstitutionalMemory) -> None:
    retrieved = constitutional.retrieve_all()
    assert len(retrieved) == len(DEFAULT_PRINCIPLES)


def test_inject_into_blocks_prepends(constitutional: ConstitutionalMemory) -> None:
    original = [AMCMemoryBlock(block_id="other", tier=2, salience=0.5, surprise_score=0.1)]
    result = constitutional.inject_into_blocks(original)
    assert len(result) == len(DEFAULT_PRINCIPLES) + 1
    assert result[0].block_id.startswith("constitutional:")
    assert result[-1].block_id == "other"


def test_injected_blocks_are_verified_trust(constitutional: ConstitutionalMemory) -> None:
    blocks = constitutional.inject_into_blocks([])
    for block in blocks:
        assert block.trust_state == TrustState.VERIFIED
        assert block.tier == 3
        assert block.provenance == "constitutional"


def test_get_violations_returns_audit_trail(constitutional: ConstitutionalMemory) -> None:
    constitutional.attempt_revoke("constitutional:0", source="user")
    constitutional.attempt_modify("constitutional:1", "bad", source="agent")
    violations = constitutional.get_violations()
    assert len(violations) == 2
    assert violations[0].action == "revoke"
    assert violations[1].action == "modify"


def test_custom_principles_extend_defaults(tier3: AMCTier3Hook) -> None:
    extra = ("Custom principle A", "Custom principle B")
    combined = DEFAULT_PRINCIPLES + extra
    constitutional = ConstitutionalMemory(tier3, principles=combined)
    assert constitutional.principle_count == len(combined)


def test_idempotent_construction(tier3: AMCTier3Hook) -> None:
    ConstitutionalMemory(tier3)
    before = set(tier3._store.keys())
    ConstitutionalMemory(tier3)
    assert set(tier3._store.keys()) == before


def test_stats_includes_violations(constitutional: ConstitutionalMemory) -> None:
    constitutional.attempt_revoke("constitutional:0", source="probe")
    stats = constitutional.stats()
    assert stats["violation_count"] == 1
    assert stats["principle_count"] == len(DEFAULT_PRINCIPLES)
