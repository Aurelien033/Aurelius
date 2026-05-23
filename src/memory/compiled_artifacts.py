"""RC-AMC compiled memory artifact schemas (P0.2 / OC-8).

Typed, evidence-grounded schemas for facts, entities, relationships, reflection QA,
and memory write plans. Schema/contract only — no LLM extraction or durable writes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src._compat import StrEnum
from src.memory.sdb_runtime import sanitize_memory_payload

SCHEMA_VERSION = "rc_amc/v1"


class CompiledArtifactError(ValueError):
    """Invalid compiled memory artifact or reference."""


class Sensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SECRET = "secret"
    UNSAFE = "unsafe"


class ConsolidatedFactStatus(StrEnum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    STALE = "stale"
    REJECTED = "rejected"


class AdmissionStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _validate_confidence(value: float, *, field_name: str = "confidence") -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value!r}")
    return value


def _validate_expiry(created_at: datetime, expiry_at: datetime | None) -> None:
    if expiry_at is not None and expiry_at <= created_at:
        raise ValueError("expiry_at must be after created_at")


def _validate_offsets(start_offset: int | None, end_offset: int | None) -> None:
    if start_offset is not None and start_offset < 0:
        raise ValueError("start_offset must be >= 0")
    if end_offset is not None and end_offset < 0:
        raise ValueError("end_offset must be >= 0")
    if start_offset is not None and end_offset is not None and end_offset < start_offset:
        raise ValueError("end_offset must be >= start_offset when both are set")


def sanitize_artifact_payload(value: Any) -> Any:
    """Redact secret-bearing keys in artifact metadata (delegates to SDB sanitizer)."""
    return sanitize_memory_payload(value)


def _ensure_json_compatible(value: Any, *, path: str = "root") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _ensure_json_compatible(v, path=f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_json_compatible(item, path=f"{path}[]") for item in value]
    raise TypeError(f"non-JSON-compatible value at {path}: {type(value).__name__}")


def stable_json_dumps(value: Any) -> str:
    canonical = _ensure_json_compatible(value)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def _evidence_content_hash(
    *,
    source_id: str,
    source_type: str,
    text: str,
    start_offset: int | None,
    end_offset: int | None,
) -> str:
    return stable_hash(
        {
            "source_id": source_id,
            "source_type": source_type,
            "text": text,
            "start_offset": start_offset,
            "end_offset": end_offset,
        }
    )


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    source_id: str
    source_type: str
    text: str
    created_at: datetime
    start_offset: int | None = None
    end_offset: int | None = None
    license: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.source_type.strip():
            raise ValueError("source_type must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        _validate_offsets(self.start_offset, self.end_offset)
        sanitized_text = str(sanitize_artifact_payload(self.text))
        sanitized_meta = sanitize_artifact_payload(self.metadata)
        if not isinstance(sanitized_meta, dict):
            sanitized_meta = {}
        object.__setattr__(self, "text", sanitized_text)
        object.__setattr__(self, "metadata", sanitized_meta)
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                _evidence_content_hash(
                    source_id=self.source_id,
                    source_type=self.source_type,
                    text=sanitized_text,
                    start_offset=self.start_offset,
                    end_offset=self.end_offset,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "timestamp": _iso(self.created_at),
            "license": self.license,
            "sensitivity": str(self.sensitivity),
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FactCandidate:
    fact_id: str
    claim: str
    confidence: float
    evidence_ids: tuple[str, ...]
    extractor: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("claim must be non-empty")
        if not self.evidence_ids:
            raise ValueError("evidence_ids must be non-empty")
        _validate_confidence(self.confidence)
        if not self.extractor.strip():
            raise ValueError("extractor must be non-empty")
        object.__setattr__(self, "metadata", sanitize_artifact_payload(self.metadata))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "claim": self.claim,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "extractor": self.extractor,
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
        }


def _default_fact_id(claim: str, evidence_ids: tuple[str, ...]) -> str:
    return stable_hash({"claim": claim, "evidence_ids": list(evidence_ids)})


@dataclass(frozen=True)
class ConsolidatedFact:
    consolidated_id: str
    claim: str
    supporting_fact_ids: tuple[str, ...]
    contradicting_fact_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: ConsolidatedFactStatus
    created_at: datetime
    expiry_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("claim must be non-empty")
        _validate_confidence(self.confidence)
        _validate_expiry(self.created_at, self.expiry_at)
        status = ConsolidatedFactStatus(self.status)
        object.__setattr__(self, "status", status)
        if status == ConsolidatedFactStatus.VERIFIED and not self.supporting_evidence_ids:
            raise ValueError("verified consolidated facts require supporting_evidence_ids")
        object.__setattr__(self, "metadata", sanitize_artifact_payload(self.metadata))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "consolidated_id": self.consolidated_id,
            "claim": self.claim,
            "supporting_fact_ids": list(self.supporting_fact_ids),
            "contradicting_fact_ids": list(self.contradicting_fact_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "confidence": self.confidence,
            "status": str(self.status),
            "expiry_at": _iso(self.expiry_at),
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EntityCard:
    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    attributes: dict[str, Any]
    evidence_ids: tuple[str, ...]
    sensitivity: Sensitivity
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical_name.strip():
            raise ValueError("canonical_name must be non-empty")
        if not self.evidence_ids:
            raise ValueError("evidence_ids must be non-empty")
        deduped = tuple(dict.fromkeys(alias.strip() for alias in self.aliases if alias.strip()))
        object.__setattr__(self, "aliases", deduped)
        object.__setattr__(self, "attributes", sanitize_artifact_payload(self.attributes))
        if not isinstance(self.attributes, dict):
            object.__setattr__(self, "attributes", {})
        object.__setattr__(self, "metadata", sanitize_artifact_payload(self.metadata))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "attributes": self.attributes,
            "evidence_ids": list(self.evidence_ids),
            "sensitivity": str(self.sensitivity),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RelationshipEdge:
    edge_id: str
    subject_id: str
    predicate: str
    object_id: str
    evidence_ids: tuple[str, ...]
    confidence: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject_id.strip() or not self.object_id.strip():
            raise ValueError("subject_id and object_id must be non-empty")
        if not self.predicate.strip():
            raise ValueError("predicate must be non-empty")
        if not self.evidence_ids:
            raise ValueError("evidence_ids must be non-empty")
        _validate_confidence(self.confidence)
        object.__setattr__(self, "metadata", sanitize_artifact_payload(self.metadata))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ReflectionQAPair:
    qa_id: str
    question: str
    answer: str
    evidence_ids: tuple[str, ...]
    source_fact_ids: tuple[str, ...]
    confidence: float
    self_contained: bool
    created_at: datetime
    expiry_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("question and answer must be non-empty")
        if not self.evidence_ids:
            raise ValueError("evidence_ids must be non-empty")
        if not isinstance(self.self_contained, bool):
            raise ValueError("self_contained must be a bool")
        _validate_confidence(self.confidence)
        _validate_expiry(self.created_at, self.expiry_at)
        object.__setattr__(self, "metadata", sanitize_artifact_payload(self.metadata))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "answer": self.answer,
            "evidence_ids": list(self.evidence_ids),
            "source_fact_ids": list(self.source_fact_ids),
            "confidence": self.confidence,
            "self_contained": self.self_contained,
            "created_at": _iso(self.created_at),
            "expiry_at": _iso(self.expiry_at),
            "metadata": self.metadata,
        }


def _has_admission_metadata(
    *,
    admission_proposal_id: str | None,
    admission_verification_id: str | None,
    admission_replay_key: str | None,
) -> bool:
    return bool(
        (admission_proposal_id and admission_proposal_id.strip())
        or (admission_verification_id and admission_verification_id.strip())
        or (admission_replay_key and admission_replay_key.strip())
    )


@dataclass(frozen=True)
class MemoryWritePlan:
    plan_id: str
    artifact_ids: tuple[str, ...]
    target_tier: str
    trust_score: float
    decay_policy: str
    safety_tier: str
    admission_status: AdmissionStatus
    created_at: datetime
    admission_proposal_id: str | None = None
    admission_verification_id: str | None = None
    admission_replay_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_ids:
            raise ValueError("artifact_ids must be non-empty")
        if not self.target_tier.strip():
            raise ValueError("target_tier must be non-empty")
        _validate_confidence(self.trust_score, field_name="trust_score")
        if not self.decay_policy.strip():
            raise ValueError("decay_policy must be non-empty")
        if not self.safety_tier.strip():
            raise ValueError("safety_tier must be non-empty")
        status = AdmissionStatus(self.admission_status)
        object.__setattr__(self, "admission_status", status)
        if status == AdmissionStatus.ACCEPTED and not _has_admission_metadata(
            admission_proposal_id=self.admission_proposal_id,
            admission_verification_id=self.admission_verification_id,
            admission_replay_key=self.admission_replay_key,
        ):
            raise ValueError(
                "accepted memory write plans require admission_proposal_id, "
                "admission_verification_id, or admission_replay_key"
            )
        object.__setattr__(self, "metadata", sanitize_artifact_payload(self.metadata))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "artifact_ids": list(self.artifact_ids),
            "target_tier": self.target_tier,
            "trust_score": self.trust_score,
            "decay_policy": self.decay_policy,
            "safety_tier": self.safety_tier,
            "admission_status": str(self.admission_status),
            "admission_proposal_id": self.admission_proposal_id,
            "admission_verification_id": self.admission_verification_id,
            "admission_replay_key": self.admission_replay_key,
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CompiledMemoryArtifact:
    compile_id: str
    schema_version: str
    raw_evidence: tuple[EvidenceSpan, ...]
    fact_candidates: tuple[FactCandidate, ...]
    consolidated_facts: tuple[ConsolidatedFact, ...]
    entity_cards: tuple[EntityCard, ...]
    relationship_edges: tuple[RelationshipEdge, ...]
    reflection_qa_pairs: tuple[ReflectionQAPair, ...]
    memory_write_plans: tuple[MemoryWritePlan, ...]
    source_ids: tuple[str, ...]
    compiler_version: str
    created_at: datetime
    artifact_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_id": self.compile_id,
            "schema_version": self.schema_version,
            "raw_evidence": [item.to_dict() for item in self.raw_evidence],
            "fact_candidates": [item.to_dict() for item in self.fact_candidates],
            "consolidated_facts": [item.to_dict() for item in self.consolidated_facts],
            "entity_cards": [item.to_dict() for item in self.entity_cards],
            "relationship_edges": [item.to_dict() for item in self.relationship_edges],
            "reflection_qa_pairs": [item.to_dict() for item in self.reflection_qa_pairs],
            "memory_write_plans": [item.to_dict() for item in self.memory_write_plans],
            "source_ids": list(self.source_ids),
            "compiler_version": self.compiler_version,
            "created_at": _iso(self.created_at),
            "artifact_hash": self.artifact_hash,
            "metadata": self.metadata,
        }

    def validate(self) -> None:
        evidence_ids = {span.evidence_id for span in self.raw_evidence}
        fact_ids = {fact.fact_id for fact in self.fact_candidates}
        consolidated_ids = {fact.consolidated_id for fact in self.consolidated_facts}
        entity_ids = {card.entity_id for card in self.entity_cards}
        edge_ids = {edge.edge_id for edge in self.relationship_edges}
        qa_ids = {qa.qa_id for qa in self.reflection_qa_pairs}
        plan_ids = {plan.plan_id for plan in self.memory_write_plans}

        known_artifact_ids = fact_ids | consolidated_ids | entity_ids | edge_ids | qa_ids | plan_ids

        def _check_refs(ids: tuple[str, ...], *, label: str, allow_external: bool = False) -> None:
            for ref in ids:
                if ref in evidence_ids or ref in known_artifact_ids:
                    continue
                if allow_external and ref.startswith("ext:"):
                    continue
                raise CompiledArtifactError(f"{label} references unknown id {ref!r}")

        for fact in self.fact_candidates:
            _check_refs(fact.evidence_ids, label="fact_candidates.evidence_ids")
            if not all(eid in evidence_ids for eid in fact.evidence_ids):
                missing = [eid for eid in fact.evidence_ids if eid not in evidence_ids]
                raise CompiledArtifactError(
                    f"fact {fact.fact_id!r} references missing evidence_ids: {missing}"
                )

        for consolidated in self.consolidated_facts:
            _check_refs(
                consolidated.supporting_evidence_ids,
                label="consolidated_facts.supporting_evidence_ids",
            )
            _check_refs(
                consolidated.contradicting_evidence_ids,
                label="consolidated_facts.contradicting_evidence_ids",
            )
            for fid in consolidated.supporting_fact_ids:
                if fid not in fact_ids:
                    raise CompiledArtifactError(
                        f"consolidated {consolidated.consolidated_id!r} "
                        f"missing supporting_fact_id {fid!r}"
                    )
            for fid in consolidated.contradicting_fact_ids:
                if fid not in fact_ids:
                    raise CompiledArtifactError(
                        f"consolidated {consolidated.consolidated_id!r} "
                        f"missing contradicting_fact_id {fid!r}"
                    )

        for card in self.entity_cards:
            if not all(eid in evidence_ids for eid in card.evidence_ids):
                missing = [eid for eid in card.evidence_ids if eid not in evidence_ids]
                raise CompiledArtifactError(
                    f"entity {card.entity_id!r} references missing evidence_ids: {missing}"
                )

        for edge in self.relationship_edges:
            if not all(eid in evidence_ids for eid in edge.evidence_ids):
                missing = [eid for eid in edge.evidence_ids if eid not in evidence_ids]
                raise CompiledArtifactError(
                    f"edge {edge.edge_id!r} references missing evidence_ids: {missing}"
                )

        for qa in self.reflection_qa_pairs:
            if not all(eid in evidence_ids for eid in qa.evidence_ids):
                missing = [eid for eid in qa.evidence_ids if eid not in evidence_ids]
                raise CompiledArtifactError(
                    f"qa {qa.qa_id!r} references missing evidence_ids: {missing}"
                )
            for fid in qa.source_fact_ids:
                if fid not in fact_ids:
                    raise CompiledArtifactError(
                        f"qa {qa.qa_id!r} references missing source_fact_id {fid!r}"
                    )

        for plan in self.memory_write_plans:
            for aid in plan.artifact_ids:
                if aid not in known_artifact_ids:
                    raise CompiledArtifactError(
                        f"write plan {plan.plan_id!r} references unknown artifact_id {aid!r}"
                    )


def _canonical_artifact_body(
    *,
    compile_id: str,
    schema_version: str,
    raw_evidence: tuple[EvidenceSpan, ...],
    fact_candidates: tuple[FactCandidate, ...],
    consolidated_facts: tuple[ConsolidatedFact, ...],
    entity_cards: tuple[EntityCard, ...],
    relationship_edges: tuple[RelationshipEdge, ...],
    reflection_qa_pairs: tuple[ReflectionQAPair, ...],
    memory_write_plans: tuple[MemoryWritePlan, ...],
    source_ids: tuple[str, ...],
    compiler_version: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "compile_id": compile_id,
        "schema_version": schema_version,
        "raw_evidence": [span.to_dict() for span in raw_evidence],
        "fact_candidates": [fact.to_dict() for fact in fact_candidates],
        "consolidated_facts": [fact.to_dict() for fact in consolidated_facts],
        "entity_cards": [card.to_dict() for card in entity_cards],
        "relationship_edges": [edge.to_dict() for edge in relationship_edges],
        "reflection_qa_pairs": [qa.to_dict() for qa in reflection_qa_pairs],
        "memory_write_plans": [plan.to_dict() for plan in memory_write_plans],
        "source_ids": list(source_ids),
        "compiler_version": compiler_version,
        "metadata": metadata,
    }


def build_compiled_artifact(
    *,
    compile_id: str,
    raw_evidence: list[EvidenceSpan] | tuple[EvidenceSpan, ...],
    fact_candidates: list[FactCandidate] | tuple[FactCandidate, ...] | None = None,
    consolidated_facts: list[ConsolidatedFact] | tuple[ConsolidatedFact, ...] | None = None,
    entity_cards: list[EntityCard] | tuple[EntityCard, ...] | None = None,
    relationship_edges: list[RelationshipEdge] | tuple[RelationshipEdge, ...] | None = None,
    reflection_qa_pairs: list[ReflectionQAPair] | tuple[ReflectionQAPair, ...] | None = None,
    memory_write_plans: list[MemoryWritePlan] | tuple[MemoryWritePlan, ...] | None = None,
    source_ids: tuple[str, ...] | list[str],
    compiler_version: str = "rc_amc/0",
    schema_version: str = SCHEMA_VERSION,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    validate: bool = True,
) -> CompiledMemoryArtifact:
    """Build a compiled artifact manifest with stable ``artifact_hash``."""
    evidence_tuple = tuple(raw_evidence)
    facts_tuple = tuple(fact_candidates or ())
    consolidated_tuple = tuple(consolidated_facts or ())
    entities_tuple = tuple(entity_cards or ())
    edges_tuple = tuple(relationship_edges or ())
    qa_tuple = tuple(reflection_qa_pairs or ())
    plans_tuple = tuple(memory_write_plans or ())
    sources_tuple = tuple(source_ids)
    created = created_at or _utc_now()
    meta = sanitize_artifact_payload(metadata or {})
    if not isinstance(meta, dict):
        meta = {}

    body = _canonical_artifact_body(
        compile_id=compile_id,
        schema_version=schema_version,
        raw_evidence=evidence_tuple,
        fact_candidates=facts_tuple,
        consolidated_facts=consolidated_tuple,
        entity_cards=entities_tuple,
        relationship_edges=edges_tuple,
        reflection_qa_pairs=qa_tuple,
        memory_write_plans=plans_tuple,
        source_ids=sources_tuple,
        compiler_version=compiler_version,
        metadata=meta,
    )
    artifact_hash = stable_hash(body)
    artifact = CompiledMemoryArtifact(
        compile_id=compile_id,
        schema_version=schema_version,
        raw_evidence=evidence_tuple,
        fact_candidates=facts_tuple,
        consolidated_facts=consolidated_tuple,
        entity_cards=entities_tuple,
        relationship_edges=edges_tuple,
        reflection_qa_pairs=qa_tuple,
        memory_write_plans=plans_tuple,
        source_ids=sources_tuple,
        compiler_version=compiler_version,
        created_at=created,
        artifact_hash=artifact_hash,
        metadata=meta,
    )
    if validate:
        artifact.validate()
    return artifact


def fact_candidate_from_claim(
    *,
    claim: str,
    evidence_ids: tuple[str, ...],
    extractor: str,
    confidence: float,
    fact_id: str | None = None,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> FactCandidate:
    fid = fact_id or _default_fact_id(claim, evidence_ids)
    return FactCandidate(
        fact_id=fid,
        claim=claim,
        confidence=confidence,
        evidence_ids=evidence_ids,
        extractor=extractor,
        created_at=created_at or _utc_now(),
        metadata=metadata or {},
    )
