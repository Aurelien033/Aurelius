"""EWM Erase/Write Update Contract for AMC (P0.3 / OC-7).

Contract-level schema for independent ``decay_gate``, ``erase_gate``, and
``write_gate`` updates with collision/contradiction policies and telemetry.
This is **not** production GDN2/KDA math and does not perform durable writes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src._compat import StrEnum
from src.memory.sdb_runtime import sanitize_memory_payload

SCHEMA_VERSION = "ewm/v1"
name = SCHEMA_VERSION  # import-smoke alias

_DURABLE_UPDATE_REASONS = frozenset(
    {
        "new",
        "correction",
        "tool_result",
    }
)


class EWMUpdateContractError(ValueError):
    """Invalid EWM update contract input or admission evaluation."""


class AMCUpdateReason(StrEnum):
    NEW = "new"
    CORRECTION = "correction"
    DECAY = "decay"
    CONFLICT = "conflict"
    MANUAL = "manual"
    TOOL_RESULT = "tool_result"
    SAFETY_QUARANTINE = "safety_quarantine"


class AMCCollisionPolicy(StrEnum):
    REJECT = "reject"
    OVERWRITE = "overwrite"
    MERGE = "merge"
    DECAY_OLD = "decay_old"
    QUARANTINE = "quarantine"


class AMCContradictionPolicy(StrEnum):
    REJECT = "reject"
    KEEP_BOTH = "keep_both"
    PREFER_NEWER = "prefer_newer"
    PREFER_HIGHER_TRUST = "prefer_higher_trust"
    QUARANTINE = "quarantine"


class AMCSafetyTier(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    UNSAFE = "unsafe"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _tensor_to_cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _gate_values(gate: float | Any) -> list[float]:
    if isinstance(gate, (int, float)):
        return [float(gate)]
    if _is_tensor(gate):
        tensor = _tensor_to_cpu(gate).flatten()
        return [float(x) for x in tensor.tolist()]
    raise EWMUpdateContractError(f"unsupported gate type: {type(gate).__name__}")


def _gate_mean(gate: float | Any) -> float:
    values = _gate_values(gate)
    return sum(values) / len(values) if values else 0.0


def validate_gate(gate: float | Any, *, name: str) -> float | Any:
    """Fail-closed gate validation; scalar gates return normalized float."""
    if isinstance(gate, (int, float)):
        value = float(gate)
        if not math.isfinite(value):
            raise EWMUpdateContractError(f"{name} must be finite, got {gate!r}")
        if not 0.0 <= value <= 1.0:
            raise EWMUpdateContractError(f"{name} must be in [0.0, 1.0], got {value!r}")
        return value
    if _is_tensor(gate):
        import torch

        tensor = _tensor_to_cpu(gate).to(dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise EWMUpdateContractError(f"{name} tensor contains NaN or Inf")
        min_v = float(tensor.min().item())
        max_v = float(tensor.max().item())
        if min_v < 0.0 or max_v > 1.0:
            raise EWMUpdateContractError(
                f"{name} tensor values must be in [0.0, 1.0], got min={min_v}, max={max_v}"
            )
        return tensor
    raise EWMUpdateContractError(f"{name} must be a float or torch.Tensor")


def clamp_gate(gate: float | Any) -> float | Any:
    """Explicit clamp helper for tests or callers that opt in to soft bounds."""
    if isinstance(gate, (int, float)):
        return max(0.0, min(1.0, float(gate)))
    if _is_tensor(gate):
        import torch

        tensor = _tensor_to_cpu(gate).to(dtype=torch.float32)
        return torch.clamp(tensor, 0.0, 1.0)
    raise EWMUpdateContractError(f"unsupported gate type for clamp: {type(gate).__name__}")


def summarize_gate(gate: float | Any) -> dict[str, Any]:
    """JSON-compatible gate summary without dumping full tensor payloads."""
    if isinstance(gate, (int, float)):
        value = float(gate)
        return {
            "kind": "scalar",
            "value": value,
            "mean": value,
            "min": value,
            "max": value,
        }
    if _is_tensor(gate):
        import torch

        tensor = _tensor_to_cpu(gate).to(dtype=torch.float32)
        flat = tensor.flatten()
        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": float(flat.mean().item()) if flat.numel() else 0.0,
            "min": float(flat.min().item()) if flat.numel() else 0.0,
            "max": float(flat.max().item()) if flat.numel() else 0.0,
            "numel": int(flat.numel()),
            "content_hash": digest,
        }
    return {"kind": "opaque", "type": type(gate).__name__}


def _pearson_correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0.0 or var_b == 0.0:
        return None
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    return cov / math.sqrt(var_a * var_b)


def gate_correlation(erase_gate: float | Any, write_gate: float | Any) -> float | None:
    """Pearson correlation when erase/write gates are same-shaped vectors."""
    erase_values = _gate_values(erase_gate)
    write_values = _gate_values(write_gate)
    if len(erase_values) != len(write_values):
        return None
    return _pearson_correlation(erase_values, write_values)


def _gate_entropy(gate: float | Any) -> float | None:
    values = _gate_values(gate)
    if not values:
        return None
    total = sum(values)
    if total <= 0.0:
        return None
    probs = [v / total for v in values if v > 0.0]
    if not probs:
        return None
    return -sum(p * math.log(p) for p in probs)


def _summarize_memory_tensor(value: Any, *, label: str) -> dict[str, Any]:
    if _is_tensor(value):
        tensor = _tensor_to_cpu(value)
        flat = tensor.flatten()
        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
        return {
            "kind": "tensor",
            "label": label,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(flat.numel()),
            "content_hash": digest,
        }
    if isinstance(value, (int, float)):
        return {"kind": "scalar", "label": label, "value": float(value)}
    return {"kind": "opaque", "label": label, "type": type(value).__name__}


def _validate_trust_score(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise EWMUpdateContractError(f"trust_score must be in [0.0, 1.0], got {value!r}")
    return value


def _validate_provenance(
    provenance: dict[str, Any],
    *,
    update_reason: AMCUpdateReason,
) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise EWMUpdateContractError("provenance must be a dict")
    sanitized = sanitize_memory_payload(provenance)
    if not isinstance(sanitized, dict):
        raise EWMUpdateContractError("provenance must be a dict")
    if str(update_reason) in _DURABLE_UPDATE_REASONS and len(sanitized) == 0:
        raise EWMUpdateContractError(
            f"provenance is required for durable update reason {update_reason!r}"
        )
    return sanitized


def _validate_memory_pair(memory_key: Any, memory_value: Any) -> None:
    if _is_tensor(memory_key) and _is_tensor(memory_value):
        import torch

        key = _tensor_to_cpu(memory_key)
        value = _tensor_to_cpu(memory_value)
        if key.shape != value.shape:
            raise EWMUpdateContractError(
                f"memory_key shape {list(key.shape)} incompatible with "
                f"memory_value shape {list(value.shape)}"
            )
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise EWMUpdateContractError("memory_key and memory_value must be finite tensors")


@dataclass(frozen=True)
class AMCEraseWriteTelemetry:
    update_id: str
    decay_mean: float
    erase_mean: float
    write_mean: float
    erase_write_correlation: float | None
    decay_gate_entropy: float | None
    erase_gate_entropy: float | None
    write_gate_entropy: float | None
    trust_score: float
    admitted: bool
    reject_code: str | None
    collision_policy: AMCCollisionPolicy
    contradiction_policy: AMCContradictionPolicy
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "decay_mean": self.decay_mean,
            "erase_mean": self.erase_mean,
            "write_mean": self.write_mean,
            "erase_write_correlation": self.erase_write_correlation,
            "decay_gate_entropy": self.decay_gate_entropy,
            "erase_gate_entropy": self.erase_gate_entropy,
            "write_gate_entropy": self.write_gate_entropy,
            "trust_score": self.trust_score,
            "admitted": self.admitted,
            "reject_code": self.reject_code,
            "collision_policy": str(self.collision_policy),
            "contradiction_policy": str(self.contradiction_policy),
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AMCEraseWriteUpdate:
    """Proposed gated AMC memory edit (contract only; no durable write)."""

    update_id: str
    memory_key: Any
    memory_value: Any
    decay_gate: float | Any
    erase_gate: float | Any
    write_gate: float | Any
    trust_score: float
    provenance: dict[str, Any]
    safety_tier: AMCSafetyTier
    update_reason: AMCUpdateReason
    collision_policy: AMCCollisionPolicy
    contradiction_policy: AMCContradictionPolicy
    created_at: datetime
    admission_proposal_id: str | None = None
    admission_replay_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.update_id.strip():
            raise EWMUpdateContractError("update_id must be non-empty")
        if self.contradiction_policy is None:
            raise EWMUpdateContractError("contradiction_policy must be explicit")
        try:
            reason = AMCUpdateReason(self.update_reason)
            collision = AMCCollisionPolicy(self.collision_policy)
            contradiction = AMCContradictionPolicy(self.contradiction_policy)
            safety = AMCSafetyTier(self.safety_tier)
        except ValueError as exc:
            raise EWMUpdateContractError(str(exc)) from exc
        object.__setattr__(self, "update_reason", reason)
        if reason == AMCUpdateReason.CONFLICT and self.contradiction_policy is None:
            raise EWMUpdateContractError("conflict updates require explicit contradiction_policy")
        object.__setattr__(self, "collision_policy", collision)
        object.__setattr__(self, "contradiction_policy", contradiction)
        object.__setattr__(self, "safety_tier", safety)
        object.__setattr__(self, "decay_gate", validate_gate(self.decay_gate, name="decay_gate"))
        object.__setattr__(self, "erase_gate", validate_gate(self.erase_gate, name="erase_gate"))
        object.__setattr__(self, "write_gate", validate_gate(self.write_gate, name="write_gate"))
        object.__setattr__(self, "trust_score", _validate_trust_score(float(self.trust_score)))
        object.__setattr__(
            self,
            "provenance",
            _validate_provenance(self.provenance, update_reason=reason),
        )
        meta = sanitize_memory_payload(self.metadata)
        object.__setattr__(self, "metadata", meta if isinstance(meta, dict) else {})
        _validate_memory_pair(self.memory_key, self.memory_value)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "schema_version": SCHEMA_VERSION,
            "memory_key": _summarize_memory_tensor(self.memory_key, label="memory_key"),
            "memory_value": _summarize_memory_tensor(self.memory_value, label="memory_value"),
            "decay_gate": summarize_gate(self.decay_gate),
            "erase_gate": summarize_gate(self.erase_gate),
            "write_gate": summarize_gate(self.write_gate),
            "trust_score": self.trust_score,
            "provenance": self.provenance,
            "safety_tier": str(self.safety_tier),
            "update_reason": str(self.update_reason),
            "collision_policy": str(self.collision_policy),
            "contradiction_policy": str(self.contradiction_policy),
            "admission_proposal_id": self.admission_proposal_id,
            "admission_replay_key": self.admission_replay_key,
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
        }


def _coerce_enum(enum_cls: type[StrEnum], value: Any, *, field_name: str) -> StrEnum:
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise EWMUpdateContractError(f"invalid {field_name}: {value!r}") from exc


def build_ewm_update(
    *,
    update_id: str,
    memory_key: Any,
    memory_value: Any,
    decay_gate: float | Any,
    erase_gate: float | Any,
    write_gate: float | Any,
    trust_score: float,
    provenance: dict[str, Any],
    safety_tier: AMCSafetyTier | str,
    update_reason: AMCUpdateReason | str,
    created_at: datetime | None = None,
    collision_policy: AMCCollisionPolicy | str = AMCCollisionPolicy.REJECT,
    contradiction_policy: AMCContradictionPolicy | str = AMCContradictionPolicy.QUARANTINE,
    admission_proposal_id: str | None = None,
    admission_replay_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AMCEraseWriteUpdate:
    return AMCEraseWriteUpdate(
        update_id=update_id,
        memory_key=memory_key,
        memory_value=memory_value,
        decay_gate=decay_gate,
        erase_gate=erase_gate,
        write_gate=write_gate,
        trust_score=trust_score,
        provenance=provenance,
        safety_tier=_coerce_enum(AMCSafetyTier, safety_tier, field_name="safety_tier"),
        update_reason=_coerce_enum(AMCUpdateReason, update_reason, field_name="update_reason"),
        collision_policy=_coerce_enum(
            AMCCollisionPolicy, collision_policy, field_name="collision_policy"
        ),
        contradiction_policy=_coerce_enum(
            AMCContradictionPolicy, contradiction_policy, field_name="contradiction_policy"
        ),
        admission_proposal_id=admission_proposal_id,
        admission_replay_key=admission_replay_key,
        created_at=created_at or _utc_now(),
        metadata=metadata or {},
    )


def admit_ewm_update(
    update: AMCEraseWriteUpdate,
    *,
    require_admission_metadata: bool = False,
) -> AMCEraseWriteTelemetry:
    """Evaluate contract-level admission only; does not write memory."""
    admitted = True
    reject_code: str | None = None

    if update.safety_tier == AMCSafetyTier.UNSAFE:
        admitted = False
        reject_code = "unsafe_memory"
    elif update.trust_score < 0.0 or update.trust_score > 1.0:
        admitted = False
        reject_code = "invalid_trust_score"
    elif str(update.update_reason) in _DURABLE_UPDATE_REASONS and not update.provenance:
        admitted = False
        reject_code = "missing_provenance"
    elif require_admission_metadata and not (
        (update.admission_proposal_id and update.admission_proposal_id.strip())
        or (update.admission_replay_key and update.admission_replay_key.strip())
    ):
        admitted = False
        reject_code = "missing_admission_metadata"
    elif (
        update.collision_policy == AMCCollisionPolicy.REJECT
        and update.update_reason == AMCUpdateReason.CONFLICT
    ):
        admitted = False
        reject_code = "collision_reject"
    elif (
        update.contradiction_policy
        in (
            AMCContradictionPolicy.REJECT,
            AMCContradictionPolicy.QUARANTINE,
        )
        and update.update_reason == AMCUpdateReason.CONFLICT
    ):
        if update.contradiction_policy == AMCContradictionPolicy.REJECT:
            admitted = False
            reject_code = "contradiction_reject"
        elif update.contradiction_policy == AMCContradictionPolicy.QUARANTINE:
            admitted = False
            reject_code = "contradiction_quarantine"

    return AMCEraseWriteTelemetry(
        update_id=update.update_id,
        decay_mean=_gate_mean(update.decay_gate),
        erase_mean=_gate_mean(update.erase_gate),
        write_mean=_gate_mean(update.write_gate),
        erase_write_correlation=gate_correlation(update.erase_gate, update.write_gate),
        decay_gate_entropy=_gate_entropy(update.decay_gate),
        erase_gate_entropy=_gate_entropy(update.erase_gate),
        write_gate_entropy=_gate_entropy(update.write_gate),
        trust_score=update.trust_score,
        admitted=admitted,
        reject_code=reject_code,
        collision_policy=update.collision_policy,
        contradiction_policy=update.contradiction_policy,
        created_at=update.created_at,
        metadata={
            "update_reason": str(update.update_reason),
            "safety_tier": str(update.safety_tier),
            "admission_proposal_id": update.admission_proposal_id,
            "admission_replay_key": update.admission_replay_key,
        },
    )
