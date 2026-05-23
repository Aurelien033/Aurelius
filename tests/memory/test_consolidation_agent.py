"""Tests for ConsolidationAgent (T24)."""

from __future__ import annotations

import json

import pytest

from src.memory.amc_tier2 import AMCTier2Hook
from src.memory.amc_tier3 import AMCTier3Hook, TrustLevel
from src.memory.consolidation_agent import (
    ConsolidationAgent,
    ConsolidationReport,
    reflect_and_consolidate,
)


@pytest.fixture
def tier2() -> AMCTier2Hook:
    return AMCTier2Hook()


@pytest.fixture
def tier3() -> AMCTier3Hook:
    return AMCTier3Hook()


@pytest.fixture
def agent(tier2: AMCTier2Hook, tier3: AMCTier3Hook) -> ConsolidationAgent:
    return ConsolidationAgent(tier2, tier3)


SAMPLE_OUTPUT = """
FACT: User prefers AMC hybrid architecture with MLA on even layers.
CONFIDENCE: 0.92
TAGS: architecture,preference
CONTRADICTS: none

FACT: Hello there!
CONFIDENCE: 0.99
TAGS: greeting
CONTRADICTS: none

FACT: 42
CONFIDENCE: 0.8
TAGS: number
CONTRADICTS: none
"""


def test_reflect_parses_fact_format(agent: ConsolidationAgent) -> None:
    proposals = agent.parse_proposals(SAMPLE_OUTPUT)
    assert len(proposals) == 3
    assert proposals[0].fact.startswith("User prefers AMC")
    assert proposals[0].confidence == pytest.approx(0.92)


def test_reflect_promotes_to_tier3(agent: ConsolidationAgent, tier3: AMCTier3Hook) -> None:
    report = agent.reflect_from_output(SAMPLE_OUTPUT)
    assert len(report.promoted) == 1
    assert report.promoted[0].key in tier3._store
    assert "AMC hybrid" in str(report.promoted[0].value)


def test_reflect_quarantines_contradictions(agent: ConsolidationAgent, tier3: AMCTier3Hook) -> None:
    existing = tier3.promote(key="old_fact", value="Old architecture choice", confidence=0.9)
    assert existing is not None
    output = f"""
FACT: User switched to full SSM stack.
CONFIDENCE: 0.88
TAGS: architecture
CONTRADICTS: old_fact
"""
    report = agent.reflect_from_output(output)
    assert len(report.quarantined) == 1
    assert tier3._store["old_fact"].trust_level == TrustLevel.QUARANTINED
    assert len(report.promoted) == 1


def test_reflect_preserves_high_confidence(agent: ConsolidationAgent) -> None:
    report = agent.reflect_from_output(SAMPLE_OUTPUT)
    assert report.promoted[0].confidence >= 0.85
    assert report.promoted[0].trust_level == TrustLevel.TRUSTED


def test_reflect_skips_greetings(agent: ConsolidationAgent) -> None:
    report = agent.reflect_from_output(SAMPLE_OUTPUT)
    assert any("Hello" in proposal.fact for proposal in report.rejected)


def test_reflect_skips_transient_numbers(agent: ConsolidationAgent) -> None:
    report = agent.reflect_from_output(SAMPLE_OUTPUT)
    assert any(proposal.fact.strip() == "42" for proposal in report.rejected)


def test_reflect_empty_transcript_no_promotions(
    agent: ConsolidationAgent,
) -> None:
    report = agent.reflect([])
    assert report.promoted == []
    assert report.quarantined == []


def test_consolidation_report_serializable(agent: ConsolidationAgent) -> None:
    report = agent.reflect_from_output(SAMPLE_OUTPUT)
    payload = report.to_dict()
    json.dumps(payload)
    assert payload["promoted_count"] == 1
    assert payload["rejected_count"] == 2


def test_reflect_and_consolidate_entrypoint(tier2: AMCTier2Hook, tier3: AMCTier3Hook) -> None:
    report = reflect_and_consolidate(
        [{"role": "user", "content": "Use AMC."}],
        tier2_hook=tier2,
        tier3_hook=tier3,
        generate_fn=lambda _prompt: SAMPLE_OUTPUT,
    )
    assert isinstance(report, ConsolidationReport)
    assert len(report.promoted) == 1
