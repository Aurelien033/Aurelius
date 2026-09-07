"""Tests for RC-AMC compiled memory artifact schemas (P0.2 / OC-8)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.memory.compiled_artifacts import (
    AdmissionStatus,
    CompiledArtifactError,
    CompiledMemoryArtifact,
    ConsolidatedFact,
    ConsolidatedFactStatus,
    EntityCard,
    EvidenceSpan,
    FactCandidate,
    MemoryWritePlan,
    ReflectionQAPair,
    RelationshipEdge,
    Sensitivity,
    build_compiled_artifact,
    stable_hash,
)
from src.memory.sdb_runtime import REDACTED


def _now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC)


def _evidence(
    *,
    evidence_id: str = "ev-1",
    source_id: str = "src-1",
    text: str = "The capital is Paris.",
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        source_type="document",
        text=text,
        created_at=_now(),
    )


def _fact(*, fact_id: str = "fact-1", evidence_ids: tuple[str, ...] = ("ev-1",)) -> FactCandidate:
    return FactCandidate(
        fact_id=fact_id,
        claim="Paris is the capital of France.",
        confidence=0.9,
        evidence_ids=evidence_ids,
        extractor="test",
        created_at=_now(),
    )


# ── Test 1: evidence span requires source grounding ────────────────────────────


def test_evidence_span_requires_source_grounding() -> None:
    span = _evidence()
    assert span.content_hash
    data = span.to_dict()
    json.dumps(data, sort_keys=True)
    with pytest.raises(ValueError, match="source_id"):
        EvidenceSpan(
            evidence_id="ev-bad",
            source_id="",
            source_type="document",
            text="x",
            created_at=_now(),
        )
    with pytest.raises(ValueError, match="offset"):
        EvidenceSpan(
            evidence_id="ev-bad2",
            source_id="s",
            source_type="document",
            text="x",
            start_offset=10,
            end_offset=5,
            created_at=_now(),
        )


# ── Test 2: secret metadata redaction ────────────────────────────────────────


def test_secret_metadata_redaction() -> None:
    span = EvidenceSpan(
        evidence_id="ev-sec",
        source_id="src",
        source_type="tool",
        text="visible",
        created_at=_now(),
        metadata={"api_key": "sk-live", "nested": {"token": "t", "note": "ok"}},
    )
    dumped = json.dumps(span.to_dict())
    assert REDACTED in dumped
    assert "sk-live" not in dumped
    assert "t" not in dumped or REDACTED in dumped
    assert span.to_dict()["metadata"]["nested"]["note"] == "ok"

    entity = EntityCard(
        entity_id="ent-1",
        canonical_name="Acme",
        aliases=("Acme",),
        attributes={"password": "hidden"},
        evidence_ids=("ev-1",),
        sensitivity=Sensitivity.INTERNAL,
        created_at=_now(),
        updated_at=_now(),
        metadata={"authorization": "Bearer x"},
    )
    ent_dump = json.dumps(entity.to_dict())
    assert "hidden" not in ent_dump
    assert REDACTED in ent_dump


# ── Test 3: fact candidate requires evidence ─────────────────────────────────


def test_fact_candidate_requires_evidence() -> None:
    _fact()
    with pytest.raises(ValueError, match="evidence_ids"):
        FactCandidate(
            fact_id="f",
            claim="orphan",
            confidence=0.5,
            evidence_ids=(),
            extractor="x",
            created_at=_now(),
        )


# ── Test 4: confidence bounds ────────────────────────────────────────────────


def test_confidence_bounds() -> None:
    with pytest.raises(ValueError, match="confidence"):
        FactCandidate(
            fact_id="f",
            claim="x",
            confidence=1.5,
            evidence_ids=("ev-1",),
            extractor="x",
            created_at=_now(),
        )
    with pytest.raises(ValueError, match="confidence"):
        ConsolidatedFact(
            consolidated_id="c",
            claim="x",
            supporting_fact_ids=(),
            contradicting_fact_ids=(),
            supporting_evidence_ids=("ev-1",),
            contradicting_evidence_ids=(),
            confidence=-0.1,
            status=ConsolidatedFactStatus.CANDIDATE,
            created_at=_now(),
        )
    with pytest.raises(ValueError, match="confidence"):
        RelationshipEdge(
            edge_id="e",
            subject_id="a",
            predicate="knows",
            object_id="b",
            evidence_ids=("ev-1",),
            confidence=2.0,
            created_at=_now(),
        )
    with pytest.raises(ValueError, match="confidence"):
        ReflectionQAPair(
            qa_id="q",
            question="Q?",
            answer="A.",
            evidence_ids=("ev-1",),
            source_fact_ids=(),
            confidence=1.01,
            self_contained=True,
            created_at=_now(),
        )


# ── Test 5: consolidated fact keeps contradictions explicit ────────────────────


def test_consolidated_fact_keeps_contradictions_explicit() -> None:
    fact = ConsolidatedFact(
        consolidated_id="cf-1",
        claim="Claim",
        supporting_fact_ids=("fact-1",),
        contradicting_fact_ids=("fact-2",),
        supporting_evidence_ids=("ev-1",),
        contradicting_evidence_ids=("ev-2",),
        confidence=0.8,
        status=ConsolidatedFactStatus.CANDIDATE,
        created_at=_now(),
    )
    assert "fact-2" in fact.contradicting_fact_ids
    assert "ev-2" in fact.contradicting_evidence_ids
    assert "fact-2" not in fact.supporting_fact_ids


def test_verified_consolidated_fact_requires_supporting_evidence() -> None:
    with pytest.raises(ValueError, match="supporting_evidence"):
        ConsolidatedFact(
            consolidated_id="cf-bad",
            claim="x",
            supporting_fact_ids=(),
            contradicting_fact_ids=(),
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(),
            confidence=0.9,
            status=ConsolidatedFactStatus.VERIFIED,
            created_at=_now(),
        )


# ── Test 6: entity cards deduplicate aliases ─────────────────────────────────


def test_entity_cards_deduplicate_aliases_and_require_canonical_name() -> None:
    card = EntityCard(
        entity_id="ent",
        canonical_name="Paris",
        aliases=("paris", "Paris", "paris"),
        attributes={},
        evidence_ids=("ev-1",),
        sensitivity=Sensitivity.PUBLIC,
        created_at=_now(),
        updated_at=_now(),
    )
    assert card.aliases == ("paris", "Paris")
    with pytest.raises(ValueError, match="canonical_name"):
        EntityCard(
            entity_id="ent-bad",
            canonical_name="",
            aliases=(),
            attributes={},
            evidence_ids=("ev-1",),
            sensitivity=Sensitivity.PUBLIC,
            created_at=_now(),
            updated_at=_now(),
        )


# ── Test 7: relationship edge requirements ───────────────────────────────────


def test_relationship_edge_requirements() -> None:
    RelationshipEdge(
        edge_id="edge-1",
        subject_id="ent-a",
        predicate="located_in",
        object_id="ent-b",
        evidence_ids=("ev-1",),
        confidence=0.7,
        created_at=_now(),
    )
    with pytest.raises(ValueError, match="predicate"):
        RelationshipEdge(
            edge_id="e",
            subject_id="a",
            predicate="",
            object_id="b",
            evidence_ids=("ev-1",),
            confidence=0.5,
            created_at=_now(),
        )
    with pytest.raises(ValueError, match="evidence_ids"):
        RelationshipEdge(
            edge_id="e",
            subject_id="a",
            predicate="rel",
            object_id="b",
            evidence_ids=(),
            confidence=0.5,
            created_at=_now(),
        )


# ── Test 8: reflection QA requirements ─────────────────────────────────────


def test_reflection_qa_requirements() -> None:
    ReflectionQAPair(
        qa_id="qa-1",
        question="What is the capital?",
        answer="Paris.",
        evidence_ids=("ev-1",),
        source_fact_ids=("fact-1",),
        confidence=0.85,
        self_contained=True,
        created_at=_now(),
    )
    with pytest.raises(ValueError, match="evidence_ids"):
        ReflectionQAPair(
            qa_id="qa-bad",
            question="Q?",
            answer="A.",
            evidence_ids=(),
            source_fact_ids=(),
            confidence=0.5,
            self_contained=False,
            created_at=_now(),
        )
    with pytest.raises(ValueError, match="self_contained"):
        ReflectionQAPair(
            qa_id="qa-bad2",
            question="Q?",
            answer="A.",
            evidence_ids=("ev-1",),
            source_fact_ids=(),
            confidence=0.5,
            self_contained="yes",  # type: ignore[arg-type]
            created_at=_now(),
        )


# ── Test 9: memory write plan admission metadata ─────────────────────────────


def test_memory_write_plan_admission_metadata() -> None:
    MemoryWritePlan(
        plan_id="plan-pending",
        artifact_ids=("fact-1",),
        target_tier="tier2",
        trust_score=0.8,
        decay_policy="default",
        safety_tier="standard",
        admission_status=AdmissionStatus.PENDING,
        created_at=_now(),
    )
    with pytest.raises(ValueError, match="admission"):
        MemoryWritePlan(
            plan_id="plan-bad",
            artifact_ids=("fact-1",),
            target_tier="tier2",
            trust_score=0.8,
            decay_policy="default",
            safety_tier="standard",
            admission_status=AdmissionStatus.ACCEPTED,
            created_at=_now(),
        )
    MemoryWritePlan(
        plan_id="plan-ok",
        artifact_ids=("fact-1",),
        target_tier="tier2",
        trust_score=0.8,
        decay_policy="default",
        safety_tier="standard",
        admission_status=AdmissionStatus.ACCEPTED,
        admission_proposal_id="prop-1",
        created_at=_now(),
    )


# ── Test 10: compiled artifact validates references ──────────────────────────


def _minimal_artifact(**kwargs: object) -> CompiledMemoryArtifact:
    evidence = [_evidence()]
    facts = [_fact()]
    entities = [
        EntityCard(
            entity_id="ent-1",
            canonical_name="France",
            aliases=(),
            attributes={},
            evidence_ids=("ev-1",),
            sensitivity=Sensitivity.PUBLIC,
            created_at=_now(),
            updated_at=_now(),
        )
    ]
    edges = [
        RelationshipEdge(
            edge_id="edge-1",
            subject_id="ent-1",
            predicate="has_capital",
            object_id="ent-2",
            evidence_ids=("ev-1",),
            confidence=0.8,
            created_at=_now(),
        )
    ]
    qa = [
        ReflectionQAPair(
            qa_id="qa-1",
            question="Capital?",
            answer="Paris",
            evidence_ids=("ev-1",),
            source_fact_ids=("fact-1",),
            confidence=0.9,
            self_contained=True,
            created_at=_now(),
        )
    ]
    plans = [
        MemoryWritePlan(
            plan_id="plan-1",
            artifact_ids=("fact-1",),
            target_tier="tier2",
            trust_score=0.7,
            decay_policy="default",
            safety_tier="standard",
            admission_status=AdmissionStatus.PENDING,
            created_at=_now(),
        )
    ]
    params: dict[str, object] = {
        "compile_id": "compile-1",
        "raw_evidence": evidence,
        "fact_candidates": facts,
        "entity_cards": entities,
        "relationship_edges": edges,
        "reflection_qa_pairs": qa,
        "memory_write_plans": plans,
        "source_ids": ("src-1",),
    }
    params.update(kwargs)
    return build_compiled_artifact(**params)  # type: ignore[arg-type]


def test_compiled_artifact_validates_references() -> None:
    artifact = _minimal_artifact()
    artifact.validate()

    bad_plan = MemoryWritePlan(
        plan_id="plan-bad",
        artifact_ids=("missing-id",),
        target_tier="tier2",
        trust_score=0.5,
        decay_policy="default",
        safety_tier="standard",
        admission_status=AdmissionStatus.PENDING,
        created_at=_now(),
    )
    with pytest.raises(CompiledArtifactError, match="unknown artifact_id"):
        build_compiled_artifact(
            compile_id="compile-bad",
            raw_evidence=[_evidence()],
            fact_candidates=[_fact()],
            memory_write_plans=[bad_plan],
            source_ids=("src-1",),
        )


# ── Test 11: compiled artifact hash stability ────────────────────────────────


def test_compiled_artifact_hash_is_stable() -> None:
    a1 = _minimal_artifact(compile_id="c1")
    a2 = _minimal_artifact(compile_id="c1")
    assert a1.artifact_hash == a2.artifact_hash

    facts_alt = [
        FactCandidate(
            fact_id="fact-1",
            claim="Different claim.",
            confidence=0.9,
            evidence_ids=("ev-1",),
            extractor="test",
            created_at=_now(),
        )
    ]
    a3 = build_compiled_artifact(
        compile_id="c1",
        raw_evidence=[_evidence()],
        fact_candidates=facts_alt,
        source_ids=("src-1",),
    )
    assert a3.artifact_hash != a1.artifact_hash


def test_stable_hash_helper() -> None:
    assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})


# ── Test 12: JSON serialization ──────────────────────────────────────────────


def test_json_serialization() -> None:
    artifact = _minimal_artifact()
    text = json.dumps(artifact.to_dict(), sort_keys=True)
    assert "artifact_hash" in text
    assert "2026-05-22" in text
    parsed = json.loads(text)
    assert isinstance(parsed["created_at"], str)


def test_expiry_after_created() -> None:
    created = _now()
    expiry = created - timedelta(hours=1)
    with pytest.raises(ValueError, match="expiry_at"):
        ConsolidatedFact(
            consolidated_id="cf",
            claim="x",
            supporting_fact_ids=(),
            contradicting_fact_ids=(),
            supporting_evidence_ids=("ev-1",),
            contradicting_evidence_ids=(),
            confidence=0.5,
            status=ConsolidatedFactStatus.CANDIDATE,
            created_at=created,
            expiry_at=expiry,
        )
