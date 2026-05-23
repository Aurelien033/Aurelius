"""SDB-Memory Runtime Contract — Stochastic-Deterministic Boundary for memory writes.

Typed propose → verify → commit/reject flow with fail-closed admission, provenance,
redacted payloads, and stable replay hashes. Pure stdlib; no torch, network, or GPU.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src._compat import StrEnum

if TYPE_CHECKING:
    from pathlib import Path

    from src.memory.sdb_persistent_log import SDBPersistentLog
    from src.memory.state_reconstruction import ReconstructedState

REDACTED = "[REDACTED]"

_SECRET_KEY_FRAGMENTS = (
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "secret",
    "credential",
    "authorization",
    "cookie",
    "private_key",
)


class SDBMemoryError(Exception):
    """Base error for SDB memory runtime failures."""


class VerificationRequiredError(SDBMemoryError):
    """Commit attempted without a verification result."""


class VerificationMismatchError(SDBMemoryError):
    """Verification result does not match the proposal being committed."""


class ProposalRejectedError(SDBMemoryError):
    """Proposal was rejected or quarantined and cannot be committed."""


class VerificationDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class MemoryOperation(StrEnum):
    STORE = "store"
    UPDATE = "update"
    DELETE = "delete"
    DECAY = "decay"
    PROMOTE = "promote"
    QUARANTINE = "quarantine"


class MemorySourceType(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
    EVAL = "eval"
    RUNTIME = "runtime"


class MemoryTargetTier(StrEnum):
    RUNTIME_CACHE = "runtime_cache"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def sanitize_memory_payload(value: Any) -> Any:
    """Return a JSON-compatible copy with secret-bearing keys redacted."""
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_secret_key(str(key)) else sanitize_memory_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_memory_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_memory_payload(item) for item in value]
    return value


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def _canonical_replay_body(event_type: str, proposal_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "proposal_id": proposal_id,
        **body,
    }


@dataclass(frozen=True)
class MemoryProposal:
    proposal_id: str
    session_id: str
    step: int
    proposer: str
    source_type: MemorySourceType
    target_tier: MemoryTargetTier
    operation: MemoryOperation
    payload: dict[str, Any]
    evidence: list[dict[str, Any]]
    created_at: datetime
    idempotency_key: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = str(self.source_type)
        data["target_tier"] = str(self.target_tier)
        data["operation"] = str(self.operation)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass(frozen=True)
class VerificationResult:
    proposal_id: str
    verifier: str
    decision: VerificationDecision
    reason: str
    deterministic: bool
    checks: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = str(self.decision)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass(frozen=True)
class MemoryCommitRecord:
    commit_id: str
    proposal_id: str
    verification_decision: VerificationDecision
    verifier: str
    target_tier: MemoryTargetTier
    operation: MemoryOperation
    committed_at: datetime
    replay_hash: str
    memory_entry_id: str | None = None
    affected_entry_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verification_decision"] = str(self.verification_decision)
        data["target_tier"] = str(self.target_tier)
        data["operation"] = str(self.operation)
        data["committed_at"] = self.committed_at.isoformat()
        return data


@dataclass(frozen=True)
class ReplayEvent:
    event_id: str
    event_type: str
    proposal_id: str
    timestamp: datetime
    metadata: dict[str, Any]
    replay_hash: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


def _proposal_idempotency_key(
    *,
    proposal_id: str,
    session_id: str,
    step: int,
    proposer: str,
    source_type: str,
    target_tier: str,
    operation: str,
    payload: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "proposal_id": proposal_id,
            "session_id": session_id,
            "step": step,
            "proposer": proposer,
            "source_type": source_type,
            "target_tier": target_tier,
            "operation": operation,
            "payload": payload,
        }
    )


def _build_replay_event(
    *,
    event_type: str,
    proposal_id: str,
    metadata: dict[str, Any],
) -> ReplayEvent:
    canonical = _canonical_replay_body(event_type, proposal_id, metadata)
    replay_hash = stable_hash(canonical)
    return ReplayEvent(
        event_id=replay_hash,
        event_type=event_type,
        proposal_id=proposal_id,
        timestamp=_utc_now(),
        metadata=metadata,
        replay_hash=replay_hash,
    )


@dataclass
class SDBMemoryRuntime:
    """Minimal orchestration for SDB memory-affecting actions."""

    persistent_log: SDBPersistentLog | None = field(default=None, repr=False)
    _events: list[ReplayEvent] = field(default_factory=list)
    _commits: dict[str, MemoryCommitRecord] = field(default_factory=dict)

    def _record_event(self, event: ReplayEvent) -> None:
        self._events.append(event)
        if self.persistent_log is not None:
            self.persistent_log.append(event)

    def propose(
        self,
        *,
        proposal_id: str | None = None,
        session_id: str,
        step: int,
        proposer: str,
        source_type: MemorySourceType | str,
        target_tier: MemoryTargetTier | str,
        operation: MemoryOperation | str,
        payload: dict[str, Any] | Any,
        evidence: list[dict[str, Any]] | list[Any] | None = None,
    ) -> MemoryProposal:
        pid = proposal_id or str(uuid.uuid4())
        src = MemorySourceType(source_type)
        tier = MemoryTargetTier(target_tier)
        op = MemoryOperation(operation)
        sanitized_payload = sanitize_memory_payload(payload)
        if not isinstance(sanitized_payload, dict):
            sanitized_payload = {"value": sanitized_payload}
        sanitized_evidence = [
            item if isinstance(item, dict) else {"value": sanitize_memory_payload(item)}
            for item in (evidence or [])
        ]
        sanitized_evidence = [
            sanitize_memory_payload(item) if isinstance(item, dict) else {"value": item}
            for item in sanitized_evidence
        ]
        created_at = _utc_now()
        idempotency_key = _proposal_idempotency_key(
            proposal_id=pid,
            session_id=session_id,
            step=step,
            proposer=proposer,
            source_type=str(src),
            target_tier=str(tier),
            operation=str(op),
            payload=sanitized_payload,
        )
        proposal = MemoryProposal(
            proposal_id=pid,
            session_id=session_id,
            step=step,
            proposer=proposer,
            source_type=src,
            target_tier=tier,
            operation=op,
            payload=sanitized_payload,
            evidence=sanitized_evidence,
            created_at=created_at,
            idempotency_key=idempotency_key,
        )
        event = _build_replay_event(
            event_type="proposed",
            proposal_id=pid,
            metadata={
                "proposer": proposer,
                "source_type": str(src),
                "target_tier": str(tier),
                "operation": str(op),
                "idempotency_key": idempotency_key,
            },
        )
        self._record_event(event)
        return proposal

    def verify(
        self,
        proposal: MemoryProposal,
        *,
        verifier: str,
        decision: VerificationDecision | str,
        reason: str,
        checks: dict[str, Any] | None = None,
        deterministic: bool = True,
    ) -> VerificationResult:
        dec = VerificationDecision(decision)
        if not reason.strip():
            raise ValueError("reason must be a non-empty string")
        result = VerificationResult(
            proposal_id=proposal.proposal_id,
            verifier=verifier,
            decision=dec,
            reason=reason,
            deterministic=deterministic,
            checks=dict(checks or {}),
            created_at=_utc_now(),
        )
        event = _build_replay_event(
            event_type="verified",
            proposal_id=proposal.proposal_id,
            metadata={
                "verifier": verifier,
                "decision": str(dec),
                "deterministic": deterministic,
                "checks": dict(checks or {}),
                "reason": reason,
            },
        )
        self._record_event(event)
        return result

    def _ensure_commit_allowed(
        self,
        proposal: MemoryProposal,
        verification_result: VerificationResult | None,
        *,
        deterministic_required: bool = False,
    ) -> VerificationResult:
        if verification_result is None:
            raise VerificationRequiredError(
                f"proposal {proposal.proposal_id!r} requires verification before commit"
            )
        if verification_result.proposal_id != proposal.proposal_id:
            raise VerificationMismatchError(
                f"verification for {verification_result.proposal_id!r} "
                f"cannot commit proposal {proposal.proposal_id!r}"
            )
        if verification_result.decision != VerificationDecision.ACCEPT:
            raise ProposalRejectedError(
                f"proposal {proposal.proposal_id!r} has decision "
                f"{verification_result.decision!r}; commit blocked"
            )
        if deterministic_required and not verification_result.deterministic:
            raise ProposalRejectedError(
                f"proposal {proposal.proposal_id!r} requires deterministic verification"
            )
        return verification_result

    def commit(
        self,
        proposal: MemoryProposal,
        *,
        verification_result: VerificationResult | None,
        memory_entry_id: str | None = None,
        affected_entry_ids: tuple[str, ...] | list[str] | None = None,
        deterministic_required: bool = False,
    ) -> MemoryCommitRecord:
        verification = self._ensure_commit_allowed(
            proposal,
            verification_result,
            deterministic_required=deterministic_required,
        )
        if proposal.idempotency_key in self._commits:
            return self._commits[proposal.idempotency_key]

        commit_id = stable_hash(
            {
                "proposal_id": proposal.proposal_id,
                "verifier": verification.verifier,
                "decision": str(verification.decision),
            }
        )
        affected = tuple(affected_entry_ids or ())
        replay_body = {
            "commit_id": commit_id,
            "verifier": verification.verifier,
            "decision": str(verification.decision),
            "target_tier": str(proposal.target_tier),
            "operation": str(proposal.operation),
            "idempotency_key": proposal.idempotency_key,
            "memory_entry_id": memory_entry_id,
            "affected_entry_ids": list(affected),
            "payload": dict(proposal.payload),
            "session_id": proposal.session_id,
            "step": proposal.step,
        }
        replay_hash = stable_hash(
            _canonical_replay_body("committed", proposal.proposal_id, replay_body)
        )
        committed_at = _utc_now()
        record = MemoryCommitRecord(
            commit_id=commit_id,
            proposal_id=proposal.proposal_id,
            verification_decision=verification.decision,
            verifier=verification.verifier,
            target_tier=proposal.target_tier,
            operation=proposal.operation,
            committed_at=committed_at,
            replay_hash=replay_hash,
            memory_entry_id=memory_entry_id,
            affected_entry_ids=affected,
        )
        self._commits[proposal.idempotency_key] = record
        event = ReplayEvent(
            event_id=replay_hash,
            event_type="committed",
            proposal_id=proposal.proposal_id,
            timestamp=committed_at,
            metadata=replay_body,
            replay_hash=replay_hash,
        )
        self._record_event(event)
        return record

    def reject(
        self,
        proposal: MemoryProposal,
        *,
        verification_result: VerificationResult | None = None,
        reason: str | None = None,
    ) -> ReplayEvent:
        if verification_result is not None and verification_result.proposal_id != proposal.proposal_id:
            raise VerificationMismatchError(
                f"verification for {verification_result.proposal_id!r} "
                f"cannot reject proposal {proposal.proposal_id!r}"
            )
        reject_reason = reason
        if verification_result is not None:
            reject_reason = verification_result.reason
        if not reject_reason or not str(reject_reason).strip():
            raise ValueError("reject requires a non-empty reason or verification_result")
        metadata: dict[str, Any] = {"reason": reject_reason}
        if verification_result is not None:
            metadata["decision"] = str(verification_result.decision)
            metadata["verifier"] = verification_result.verifier
        event = _build_replay_event(
            event_type="rejected",
            proposal_id=proposal.proposal_id,
            metadata=metadata,
        )
        self._record_event(event)
        return event

    def replay_events(self) -> list[ReplayEvent]:
        if self.persistent_log is not None:
            return self.persistent_log.replay_from(0)
        return list(self._events)

    def record_slr_selection(
        self,
        *,
        session_id: str,
        step: int,
        trial: Any,
        recalled_entry_ids: Sequence[str],
    ) -> ReplayEvent:
        """Append an SLR selection audit event (proposal-only; no memory commit)."""
        selection = trial.selection
        replay = trial.replay
        event = _build_replay_event(
            event_type="slr_selection",
            proposal_id=selection.selection_id,
            metadata={
                "session_id": session_id,
                "step": step,
                "selected_candidate_id": selection.selected_candidate_id,
                "recalled_entry_ids": list(recalled_entry_ids),
                "selection": selection.to_dict(),
                "replay": replay.to_dict(),
                "proposal_only": True,
            },
        )
        self._record_event(event)
        return event

    def recover_from_crash(
        self,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> ReconstructedState:
        """Recover Tier-2/Tier-3 by loading a checkpoint and replaying WAL tail events.

        Requires ``persistent_log``. When ``checkpoint_path`` is omitted, the latest
        row in ``amc_checkpoints`` may supply ``amc_checkpoint_path`` in its blob.
        """
        from pathlib import Path as PathLib

        from plugins.memory.episodic_memory import EpisodicMemory
        from src.memory.amc_checkpoint import load_amc_checkpoint
        from src.memory.amc_tier3 import AMCTier3Config, AMCTier3Hook
        from src.memory.state_reconstruction import StateReconstructor

        if self.persistent_log is None:
            raise RuntimeError("recover_from_crash requires persistent_log")

        tier2 = EpisodicMemory()
        tier3 = AMCTier3Hook(AMCTier3Config())
        from_seq = 0
        ckpt_path = PathLib(checkpoint_path) if checkpoint_path is not None else None

        loaded = self.persistent_log.load_latest_checkpoint()
        if loaded is not None:
            from_seq = int(loaded[0])
            blob = loaded[1]
            if ckpt_path is None and isinstance(blob.get("amc_checkpoint_path"), str):
                ckpt_path = PathLib(blob["amc_checkpoint_path"])

        if ckpt_path is not None:
            tier2, tier3, _header = load_amc_checkpoint(ckpt_path)

        recon = StateReconstructor(self.persistent_log)
        return recon.reconstruct_with_base(tier2, tier3, from_seq=from_seq)
