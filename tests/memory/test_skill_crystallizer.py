"""Tests for SkillCrystallizer (T25)."""

from __future__ import annotations

import pytest

from src.memory.amc_tier3 import AMCTier3Hook, TrustLevel
from src.memory.skill_crystallizer import CrystallizationProposal, SkillCrystallizer


@pytest.fixture
def tier3() -> AMCTier3Hook:
    return AMCTier3Hook()


@pytest.fixture
def crystallizer(tier3: AMCTier3Hook) -> SkillCrystallizer:
    return SkillCrystallizer(
        tier3,
        generate_fn=lambda _prompt: "General principle: prefer AMC hybrid memory architecture.",
        retrieval_threshold=3,
    )


def _trusted_entry(tier3: AMCTier3Hook, key: str, value: str, confidence: float = 0.8) -> None:
    entry = tier3.promote(key=key, value=value, confidence=confidence)
    assert entry is not None


def test_retrieval_count_increments(crystallizer: SkillCrystallizer, tier3: AMCTier3Hook) -> None:
    _trusted_entry(tier3, "fact:a", "User prefers MLA layers on even indices.")
    entry = tier3._store["fact:a"]
    crystallizer.record_retrieval(entry, "architecture question")
    crystallizer.record_retrieval(entry, "layer layout")
    assert crystallizer._retrieval_counts["fact:a"] == 2


def test_scan_returns_proposals_over_threshold(
    crystallizer: SkillCrystallizer,
    tier3: AMCTier3Hook,
) -> None:
    _trusted_entry(tier3, "fact:b", "Promotion gates use Gumbel-Softmax.")
    entry = tier3._store["fact:b"]
    for index in range(3):
        crystallizer.record_retrieval(entry, f"query-{index}")
    proposals = crystallizer.scan_for_crystallization()
    assert len(proposals) == 1
    assert proposals[0].source_key == "fact:b"
    assert proposals[0].retrieval_count == 3


def test_crystallize_produces_new_entry(
    crystallizer: SkillCrystallizer, tier3: AMCTier3Hook
) -> None:
    _trusted_entry(tier3, "fact:c", "Tier-2 stores episodic surprises.")
    proposal = CrystallizationProposal(
        source_key="fact:c",
        source_value="Tier-2 stores episodic surprises.",
        retrieval_count=3,
        history=({"query": "memory tiers", "timestamp": 1.0},),
    )
    entry = crystallizer.crystallize(proposal)
    assert entry is not None
    assert entry.key == "crystal:fact:c"
    assert entry.key in tier3._store


def test_crystallized_entry_has_elevated_confidence(
    crystallizer: SkillCrystallizer,
    tier3: AMCTier3Hook,
) -> None:
    _trusted_entry(tier3, "fact:d", "Surprise head uses detached hidden states.", confidence=0.75)
    proposal = CrystallizationProposal(
        source_key="fact:d",
        source_value=tier3._store["fact:d"].value,
        retrieval_count=4,
        history=(),
    )
    entry = crystallizer.crystallize(proposal)
    assert entry is not None
    assert entry.confidence == pytest.approx(0.8)


def test_crystallize_resets_count(crystallizer: SkillCrystallizer, tier3: AMCTier3Hook) -> None:
    _trusted_entry(tier3, "fact:e", "Constitutional memory is permanent.")
    crystallizer._retrieval_counts["fact:e"] = 5
    proposal = CrystallizationProposal(
        source_key="fact:e",
        source_value="Constitutional memory is permanent.",
        retrieval_count=5,
        history=(),
    )
    crystallizer.crystallize(proposal)
    assert crystallizer._retrieval_counts["fact:e"] == 0


def test_run_cycle_processes_all_ready(
    crystallizer: SkillCrystallizer, tier3: AMCTier3Hook
) -> None:
    _trusted_entry(tier3, "fact:f", "Replay only commits durable events.")
    _trusted_entry(tier3, "fact:g", "KV cache respects trust levels.")
    for key in ("fact:f", "fact:g"):
        entry = tier3._store[key]
        for index in range(3):
            crystallizer.record_retrieval(entry, f"{key}-{index}")
    results = crystallizer.run_cycle()
    assert len(results) == 2
    assert all(entry.key.startswith("crystal:") for entry in results)


def test_crystallization_respects_trust_level(
    crystallizer: SkillCrystallizer,
    tier3: AMCTier3Hook,
) -> None:
    entry = tier3.promote(
        key="fact:low",
        value="Low confidence note",
        confidence=0.2,
    )
    assert entry is not None
    assert entry.trust_level == TrustLevel.QUARANTINED
    crystallizer._retrieval_counts["fact:low"] = 10
    assert crystallizer.scan_for_crystallization() == []
