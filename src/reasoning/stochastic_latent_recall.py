"""P0.6 / OC-11 — SLR replayable stochastic recall contract (scaffold only).

Stochasticity is confined to candidate generation; selection is deterministic and
logged. SLR is disabled by default and does not commit durable memory or execute
agent actions. Selected candidates are proposals only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src._compat import StrEnum

SCHEMA_VERSION = "slr.v1"
REDACTED = "[REDACTED]"
DEFAULT_PERTURBATION_WIDTH = 8
DETERMINISTIC_TIEBREAKER = "metric_desc_then_candidate_id_asc"

_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "private_key",
    "connection_string",
)


class SLRSelectionMetric(StrEnum):
    SCORE = "score"
    CONFIDENCE = "confidence"
    VERIFIER = "verifier"
    COST_ADJUSTED_SCORE = "cost_adjusted_score"


class SLRCommitPolicy(StrEnum):
    PROPOSAL_ONLY = "proposal_only"
    SDB_REQUIRED = "sdb_required"
    DISABLED = "disabled"


class SLRCandidateStatus(StrEnum):
    GENERATED = "generated"
    SELECTED = "selected"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class SLRRecallMode(StrEnum):
    QUERY_PERTURBATION = "query_perturbation"
    LATENT_PERTURBATION = "latent_perturbation"
    MEMORY_KEY_PERTURBATION = "memory_key_perturbation"
    REPLAY_ONLY = "replay_only"


class SLRDisabledError(RuntimeError):
    """Raised when SLR candidate generation is invoked while disabled."""


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def sanitize_slr_payload(value: Any) -> Any:
    """Recursively sanitize dict/list/tuple values; redact secret-bearing keys."""
    if isinstance(value, Mapping):
        return {
            str(k): REDACTED if _is_secret_key(str(k)) else sanitize_slr_payload(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_slr_payload(item) for item in value]
    return value


def stable_json_dumps(value: Any) -> str:
    """Stable JSON for audit hashes; rejects unsupported non-JSON values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_sha256(value: Any) -> str:
    """SHA-256 hex digest of stable_json_dumps(value)."""
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def _deterministic_created_at(*, replay_seed: int, query_id: str, label: str) -> str:
    """Replay-stable timestamp string for audit records (not wall-clock)."""
    token = stable_sha256({"replay_seed": replay_seed, "query_id": query_id, "label": label})
    seconds = int(token[:8], 16) % 86_400
    base = datetime(2000, 1, 1, tzinfo=UTC)
    instant = base.replace(hour=0, minute=0, second=0) + timedelta(seconds=seconds)
    return instant.isoformat()


def default_slr_config(
    *,
    enabled: bool = False,
    k: int = 1,
    noise_sigma: float = 0.0,
    replay_seed: int = 0,
    selection_metric: SLRSelectionMetric = SLRSelectionMetric.SCORE,
    commit_policy: SLRCommitPolicy = SLRCommitPolicy.PROPOSAL_ONLY,
    recall_mode: SLRRecallMode = SLRRecallMode.QUERY_PERTURBATION,
    max_candidates: int = 8,
    metadata: Mapping[str, Any] | None = None,
) -> SLRConfig:
    """Construct SLR config with safe defaults (disabled, proposal-only)."""
    return SLRConfig(
        enabled=enabled,
        k=k,
        noise_sigma=noise_sigma,
        replay_seed=replay_seed,
        selection_metric=selection_metric,
        commit_policy=commit_policy,
        recall_mode=recall_mode,
        max_candidates=max_candidates,
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class SLRConfig:
    enabled: bool
    k: int
    noise_sigma: float
    replay_seed: int
    selection_metric: SLRSelectionMetric
    commit_policy: SLRCommitPolicy
    recall_mode: SLRRecallMode
    max_candidates: int = 8
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_slr_payload(dict(self.metadata)))
        validate_slr_config(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "k": self.k,
            "noise_sigma": self.noise_sigma,
            "replay_seed": self.replay_seed,
            "selection_metric": str(self.selection_metric),
            "commit_policy": str(self.commit_policy),
            "recall_mode": str(self.recall_mode),
            "max_candidates": self.max_candidates,
            "metadata": dict(self.metadata),
        }


def validate_slr_config(config: SLRConfig) -> None:
    """Fail closed on invalid SLR configuration."""
    if config.k < 1:
        raise ValueError(f"SLRConfig.k must be >= 1, got {config.k}")
    if config.k > config.max_candidates:
        raise ValueError(
            f"SLRConfig.k ({config.k}) must be <= max_candidates ({config.max_candidates})"
        )
    if config.max_candidates > 32:
        raise ValueError(f"SLRConfig.max_candidates must be <= 32, got {config.max_candidates}")
    if not math.isfinite(config.noise_sigma) or config.noise_sigma < 0.0:
        raise ValueError(f"SLRConfig.noise_sigma must be finite and >= 0, got {config.noise_sigma}")
    if not isinstance(config.replay_seed, int):
        raise ValueError("SLRConfig.replay_seed must be an int")
    if config.commit_policy not in (
        SLRCommitPolicy.PROPOSAL_ONLY,
        SLRCommitPolicy.SDB_REQUIRED,
        SLRCommitPolicy.DISABLED,
    ):
        raise ValueError(f"unsupported commit_policy: {config.commit_policy}")
    sanitize_slr_payload(dict(config.metadata))


@dataclass(frozen=True)
class SLRQuery:
    query_id: str
    query_text: str
    memory_scope: tuple[str, ...]
    baseline_candidate_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("SLRQuery.query_id must be non-empty")
        if not self.query_text:
            raise ValueError("SLRQuery.query_text must be non-empty")
        object.__setattr__(self, "metadata", sanitize_slr_payload(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "memory_scope": list(self.memory_scope),
            "baseline_candidate_ids": list(self.baseline_candidate_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SLRCandidate:
    candidate_id: str
    trajectory_index: int
    replay_seed: int
    perturbation: tuple[float, ...]
    recall_keys: tuple[str, ...]
    recalled_entry_ids: tuple[str, ...]
    score: float
    confidence: float
    verifier_score: float | None
    cost: float | None
    status: SLRCandidateStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("SLRCandidate.candidate_id must be non-empty")
        if self.trajectory_index < 0:
            raise ValueError("SLRCandidate.trajectory_index must be >= 0")
        if not math.isfinite(self.score):
            raise ValueError("SLRCandidate.score must be finite")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("SLRCandidate.confidence must be in [0.0, 1.0]")
        if self.verifier_score is not None and not math.isfinite(self.verifier_score):
            raise ValueError("SLRCandidate.verifier_score must be finite when set")
        if self.cost is not None and self.cost < 0.0:
            raise ValueError("SLRCandidate.cost must be >= 0 when set")
        object.__setattr__(self, "metadata", sanitize_slr_payload(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "trajectory_index": self.trajectory_index,
            "replay_seed": self.replay_seed,
            "perturbation": list(self.perturbation),
            "recall_keys": list(self.recall_keys),
            "recalled_entry_ids": list(self.recalled_entry_ids),
            "score": self.score,
            "confidence": self.confidence,
            "verifier_score": self.verifier_score,
            "cost": self.cost,
            "status": str(self.status),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SLRSelectionRecord:
    selection_id: str
    query_id: str
    selected_candidate_id: str
    candidate_ids: tuple[str, ...]
    selection_metric: SLRSelectionMetric
    replay_seed: int
    deterministic_tiebreaker: str
    selected_score: float
    rejected_candidate_ids: tuple[str, ...]
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selection_id:
            raise ValueError("SLRSelectionRecord.selection_id must be non-empty")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected_candidate_id must be one of candidate_ids")
        object.__setattr__(self, "metadata", sanitize_slr_payload(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "query_id": self.query_id,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_ids": list(self.candidate_ids),
            "selection_metric": str(self.selection_metric),
            "replay_seed": self.replay_seed,
            "deterministic_tiebreaker": self.deterministic_tiebreaker,
            "selected_score": self.selected_score,
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SLRReplayRecord:
    replay_id: str
    schema_version: str
    config_hash: str
    query_hash: str
    candidate_hashes: tuple[str, ...]
    selection_hash: str
    replay_seed: int
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("SLRReplayRecord.replay_id must be non-empty")
        object.__setattr__(self, "metadata", sanitize_slr_payload(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "query_hash": self.query_hash,
            "candidate_hashes": list(self.candidate_hashes),
            "selection_hash": self.selection_hash,
            "replay_seed": self.replay_seed,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SLRRecallTrialResult:
    """Bundle returned by run_slr_recall_trial (proposal-only; no commits)."""

    config: SLRConfig
    query: SLRQuery
    candidates: tuple[SLRCandidate, ...]
    selection: SLRSelectionRecord
    replay: SLRReplayRecord


def generate_replayable_perturbation(
    *,
    replay_seed: int,
    trajectory_index: int,
    width: int,
    noise_sigma: float,
) -> tuple[float, ...]:
    """Deterministic perturbation via local RNG; all zeros when noise_sigma == 0."""
    if width < 1 or width > 64:
        raise ValueError(f"perturbation width must be in [1, 64], got {width}")
    if not math.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("noise_sigma must be finite and >= 0")
    if noise_sigma == 0.0:
        return tuple(0.0 for _ in range(width))
    rng = random.Random(replay_seed ^ (trajectory_index * 0x9E3779B1))
    return tuple(rng.gauss(0.0, noise_sigma) for _ in range(width))


def _default_recall_key_provider(
    query: SLRQuery,
    perturbation: tuple[float, ...],
) -> tuple[str, ...]:
    digest = stable_sha256(
        {
            "query_id": query.query_id,
            "perturbation": list(perturbation),
        }
    )[:16]
    return (f"slr:{query.query_id}:{digest}",)


def _default_score_fn(
    query: SLRQuery,
    recall_keys: tuple[str, ...],
    perturbation: tuple[float, ...],
) -> float:
    raw = stable_sha256(
        {
            "query_id": query.query_id,
            "recall_keys": list(recall_keys),
            "perturbation": list(perturbation),
        }
    )
    return int(raw[:8], 16) / float(0xFFFFFFFF)


def _selection_sort_key(
    candidate: SLRCandidate,
    metric: SLRSelectionMetric,
) -> tuple[float, str]:
    if metric == SLRSelectionMetric.SCORE:
        value = candidate.score
    elif metric == SLRSelectionMetric.CONFIDENCE:
        value = candidate.confidence
    elif metric == SLRSelectionMetric.VERIFIER:
        value = candidate.verifier_score if candidate.verifier_score is not None else float("-inf")
    elif metric == SLRSelectionMetric.COST_ADJUSTED_SCORE:
        cost = candidate.cost if candidate.cost is not None else 0.0
        value = candidate.score - cost
    else:
        raise ValueError(f"unsupported selection metric: {metric}")
    return (-value, candidate.candidate_id)


def generate_slr_candidates(
    *,
    config: SLRConfig,
    query: SLRQuery,
    recall_key_provider: Callable[[SLRQuery, tuple[float, ...]], Sequence[str]] | None = None,
    score_fn: Callable[[SLRQuery, tuple[str, ...], tuple[float, ...]], float] | None = None,
) -> tuple[SLRCandidate, ...]:
    """Generate K replayable recall candidates (proposals only; no memory writes)."""
    validate_slr_config(config)
    if not config.enabled:
        raise SLRDisabledError(
            "SLR is disabled (config.enabled=False); enable explicitly for stochastic candidates"
        )

    key_fn = recall_key_provider or (lambda q, p: _default_recall_key_provider(q, p))
    score_fn_impl = score_fn or _default_score_fn
    width = DEFAULT_PERTURBATION_WIDTH
    candidates: list[SLRCandidate] = []

    for trajectory_index in range(config.k):
        perturbation = generate_replayable_perturbation(
            replay_seed=config.replay_seed,
            trajectory_index=trajectory_index,
            width=width,
            noise_sigma=config.noise_sigma,
        )
        recall_keys = tuple(key_fn(query, perturbation))
        score = float(score_fn_impl(query, recall_keys, perturbation))
        confidence = min(1.0, max(0.0, score))
        candidate_id = f"{query.query_id}-t{trajectory_index}"
        candidates.append(
            SLRCandidate(
                candidate_id=candidate_id,
                trajectory_index=trajectory_index,
                replay_seed=config.replay_seed,
                perturbation=perturbation,
                recall_keys=recall_keys,
                recalled_entry_ids=tuple(query.baseline_candidate_ids),
                score=score,
                confidence=confidence,
                verifier_score=None,
                cost=None,
                status=SLRCandidateStatus.GENERATED,
                metadata={"recall_mode": str(config.recall_mode)},
            )
        )

    return tuple(candidates)


def select_slr_candidate(
    *,
    candidates: Sequence[SLRCandidate],
    metric: SLRSelectionMetric,
    query_id: str,
    replay_seed: int,
) -> SLRSelectionRecord:
    """Deterministically select one candidate; stable tie-break by candidate_id ascending."""
    if not candidates:
        raise ValueError("select_slr_candidate requires at least one candidate")

    ordered = sorted(
        candidates,
        key=lambda c: _selection_sort_key(c, metric),
    )
    selected = ordered[0]
    candidate_ids = tuple(c.candidate_id for c in candidates)
    rejected = tuple(c.candidate_id for c in ordered[1:])
    sort_value, _ = _selection_sort_key(selected, metric)
    selected_metric_value = -sort_value

    return SLRSelectionRecord(
        selection_id=f"sel:{query_id}:{replay_seed}",
        query_id=query_id,
        selected_candidate_id=selected.candidate_id,
        candidate_ids=candidate_ids,
        selection_metric=metric,
        replay_seed=replay_seed,
        deterministic_tiebreaker=DETERMINISTIC_TIEBREAKER,
        selected_score=selected_metric_value,
        rejected_candidate_ids=rejected,
        created_at=_deterministic_created_at(
            replay_seed=replay_seed,
            query_id=query_id,
            label="selection",
        ),
        metadata={"proposal_only": True},
    )


def build_slr_replay_record(
    *,
    config: SLRConfig,
    query: SLRQuery,
    candidates: Sequence[SLRCandidate],
    selection: SLRSelectionRecord,
) -> SLRReplayRecord:
    """Build deterministic replay/audit record (no raw secrets)."""
    config_hash = stable_sha256(config.to_dict())
    query_hash = stable_sha256(query.to_dict())
    candidate_hashes = tuple(stable_sha256(c.to_dict()) for c in candidates)
    selection_hash = stable_sha256(selection.to_dict())
    replay_id = stable_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": config_hash,
            "query_hash": query_hash,
            "selection_hash": selection_hash,
            "replay_seed": config.replay_seed,
        }
    )[:32]

    return SLRReplayRecord(
        replay_id=f"replay:{replay_id}",
        schema_version=SCHEMA_VERSION,
        config_hash=config_hash,
        query_hash=query_hash,
        candidate_hashes=candidate_hashes,
        selection_hash=selection_hash,
        replay_seed=config.replay_seed,
        created_at=_deterministic_created_at(
            replay_seed=config.replay_seed,
            query_id=query.query_id,
            label="replay",
        ),
        metadata={
            "k": config.k,
            "noise_sigma": config.noise_sigma,
            "recall_mode": str(config.recall_mode),
            "commit_policy": str(config.commit_policy),
            "proposal_only": True,
        },
    )


def run_slr_recall_trial(
    *,
    config: SLRConfig,
    query: SLRQuery,
    recall_key_provider: Callable[[SLRQuery, tuple[float, ...]], Sequence[str]] | None = None,
    score_fn: Callable[[SLRQuery, tuple[str, ...], tuple[float, ...]], float] | None = None,
) -> SLRRecallTrialResult:
    """Validate, generate candidates, select, and build replay record (no commits)."""
    validate_slr_config(config)
    if not query.query_id:
        raise ValueError("SLRQuery.query_id must be non-empty")

    candidates = generate_slr_candidates(
        config=config,
        query=query,
        recall_key_provider=recall_key_provider,
        score_fn=score_fn,
    )
    selection = select_slr_candidate(
        candidates=candidates,
        metric=config.selection_metric,
        query_id=query.query_id,
        replay_seed=config.replay_seed,
    )
    replay = build_slr_replay_record(
        config=config,
        query=query,
        candidates=candidates,
        selection=selection,
    )
    return SLRRecallTrialResult(
        config=config,
        query=query,
        candidates=candidates,
        selection=selection,
        replay=replay,
    )
