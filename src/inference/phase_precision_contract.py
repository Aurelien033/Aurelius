"""P0.7 / OC-13 — PPDQ phase-aware precision contract (scaffold only).

Defines serving-phase precision plans with BF16-safe defaults, optional low-precision
prefill only when capability is explicitly detected, and conservative decode/verifier/
commit precision. No NVFP4 kernels, no weight quantization, no default serving changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src._compat import StrEnum

SCHEMA_VERSION = "ppdq.v1"
REDACTED = "[REDACTED]"

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

_NVIDIA_SMI_TIMEOUT_SEC = 2.0

NvidiaSmiRunner = Callable[[list[str]], subprocess.CompletedProcess[str] | None]


class ServingPhase(StrEnum):
    PROMPT_PREFILL = "prompt_prefill"
    AMC_MEMORY_PREFILL = "amc_memory_prefill"
    DECODE = "decode"
    SAFETY_VERIFIER = "safety_verifier"
    COMMIT_AUDIT = "commit_audit"


class PrecisionMode(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    NVFP4 = "nvfp4"
    INT8 = "int8"
    UNKNOWN = "unknown"


class PrecisionSupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class FallbackPolicy(StrEnum):
    BF16 = "bf16"
    FP32 = "fp32"
    ERROR = "error"


class PhaseRiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrecisionDecision(StrEnum):
    USE_REQUESTED = "use_requested"
    FALLBACK = "fallback"
    REJECT = "reject"


ALL_SERVING_PHASES = tuple(ServingPhase)


class PrecisionRejectedError(RuntimeError):
    """Raised when fallback_policy ERROR blocks an unsupported precision request."""


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def sanitize_precision_payload(value: Any) -> Any:
    """Recursively sanitize dict/list/tuple values; redact secret-bearing keys."""
    if isinstance(value, Mapping):
        return {
            str(k): REDACTED if _is_secret_key(str(k)) else sanitize_precision_payload(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_precision_payload(item) for item in value]
    return value


def stable_json_dumps(value: Any) -> str:
    """Stable JSON for audit hashes; rejects unsupported non-JSON values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_sha256(value: Any) -> str:
    """SHA-256 hex digest of stable_json_dumps(value)."""
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fallback_precision(policy: FallbackPolicy) -> PrecisionMode:
    if policy == FallbackPolicy.FP32:
        return PrecisionMode.FP32
    return PrecisionMode.BF16


def _phase_forbids_precision(
    phase: ServingPhase,
    precision: PrecisionMode,
    *,
    allow_experimental: bool,
) -> str | None:
    if phase == ServingPhase.SAFETY_VERIFIER and precision == PrecisionMode.NVFP4:
        return "safety_verifier forbids NVFP4"
    if phase == ServingPhase.COMMIT_AUDIT and precision in (
        PrecisionMode.NVFP4,
        PrecisionMode.INT8,
    ):
        return f"commit_audit forbids {precision}"
    if phase == ServingPhase.DECODE and precision == PrecisionMode.NVFP4:
        if not allow_experimental:
            return "decode forbids NVFP4 unless allow_experimental"
    return None


def _default_nvidia_smi_runner(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(  # noqa: S603
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


@dataclass(frozen=True)
class PrecisionCapability:
    precision: PrecisionMode
    support: PrecisionSupport
    reason: str
    hardware: str | None
    backend: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("PrecisionCapability.reason must be non-empty")
        object.__setattr__(self, "metadata", sanitize_precision_payload(dict(self.metadata)))


@dataclass(frozen=True)
class PhasePrecisionRequest:
    phase: ServingPhase
    requested_precision: PrecisionMode
    fallback_policy: FallbackPolicy
    risk_tier: PhaseRiskTier
    allow_experimental: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_precision_payload(dict(self.metadata)))


@dataclass(frozen=True)
class PhasePrecisionDecision:
    phase: ServingPhase
    requested_precision: PrecisionMode
    effective_precision: PrecisionMode
    decision: PrecisionDecision
    fallback_reason: str | None
    capability: PrecisionCapability
    risk_tier: PhaseRiskTier
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effective_precision == PrecisionMode.UNKNOWN:
            raise ValueError("effective_precision must not be UNKNOWN")
        object.__setattr__(self, "metadata", sanitize_precision_payload(dict(self.metadata)))


@dataclass(frozen=True)
class PhasePrecisionPlan:
    schema_version: str
    plan_id: str
    decisions: tuple[PhasePrecisionDecision, ...]
    created_at: str
    default_precision: PrecisionMode
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_precision_payload(dict(self.metadata)))


@dataclass(frozen=True)
class PhaseTelemetryRecord:
    phase: ServingPhase
    effective_precision: PrecisionMode
    latency_ms: float | None
    tokens: int | None
    memory_bytes: int | None
    quality_delta: float | None
    safety_reject_count: int | None
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms is not None:
            if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
                raise ValueError("latency_ms must be finite and >= 0")
        if self.tokens is not None and self.tokens < 0:
            raise ValueError("tokens must be >= 0")
        if self.memory_bytes is not None and self.memory_bytes < 0:
            raise ValueError("memory_bytes must be >= 0")
        object.__setattr__(self, "metadata", sanitize_precision_payload(dict(self.metadata)))


def _probe_nvfp4_hardware(
    runner: NvidiaSmiRunner | None = None,
) -> tuple[str | None, str | None]:
    """Best-effort NVFP4 hardware hint; safe when nvidia-smi is absent."""
    run = runner or _default_nvidia_smi_runner
    result = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if result is None or result.returncode != 0:
        return None, "cpu"
    names = (result.stdout or "").strip().lower()
    if not names:
        return None, "cpu"
    first_line = names.splitlines()[0].strip()
    return first_line or None, "nvidia-smi"


def detect_precision_capability(
    precision: PrecisionMode,
    *,
    runner: NvidiaSmiRunner | None = None,
) -> PrecisionCapability:
    """Standard-library capability probe; no GPU imports at module import time."""
    system = platform.system()
    machine = platform.machine()
    backend = "contract"

    if precision in (PrecisionMode.FP32, PrecisionMode.BF16, PrecisionMode.FP16):
        return PrecisionCapability(
            precision=precision,
            support=PrecisionSupport.SUPPORTED,
            reason=f"{precision} supported by PPDQ contract baseline",
            hardware=f"{system}/{machine}",
            backend=backend,
            metadata={"probe": "contract_baseline"},
        )

    if precision == PrecisionMode.INT8:
        return PrecisionCapability(
            precision=precision,
            support=PrecisionSupport.UNSUPPORTED,
            reason="INT8 serving path not enabled in P0.7 contract",
            hardware=f"{system}/{machine}",
            backend=backend,
            metadata={"probe": "contract_disabled"},
        )

    if precision == PrecisionMode.UNKNOWN:
        return PrecisionCapability(
            precision=precision,
            support=PrecisionSupport.UNKNOWN,
            reason="unknown precision mode",
            hardware=f"{system}/{machine}",
            backend=backend,
            metadata={"probe": "unknown"},
        )

    if precision == PrecisionMode.NVFP4:
        gpu_name, probe_backend = _probe_nvfp4_hardware(runner)
        metadata: dict[str, Any] = {
            "probe": probe_backend,
            "ppdq_nvfp4_kernels": False,
        }
        if gpu_name:
            metadata["gpu_name"] = gpu_name
        # P0.7: no real NVFP4 kernel detection; require explicit future evidence.
        if gpu_name and ("blackwell" in gpu_name or "b200" in gpu_name or "gb200" in gpu_name):
            return PrecisionCapability(
                precision=precision,
                support=PrecisionSupport.UNKNOWN,
                reason="NVFP4 hardware hint present; kernel path not implemented in P0.7",
                hardware=gpu_name,
                backend=probe_backend,
                metadata=metadata,
            )
        return PrecisionCapability(
            precision=precision,
            support=PrecisionSupport.UNSUPPORTED,
            reason="NVFP4 not detected or not enabled in P0.7 contract",
            hardware=gpu_name or f"{system}/{machine}",
            backend=probe_backend,
            metadata=metadata,
        )

    raise ValueError(f"unsupported precision mode: {precision}")


def default_phase_precision_requests() -> tuple[PhasePrecisionRequest, ...]:
    """Conservative BF16/FP32 defaults; no NVFP4."""
    return (
        PhasePrecisionRequest(
            phase=ServingPhase.PROMPT_PREFILL,
            requested_precision=PrecisionMode.BF16,
            fallback_policy=FallbackPolicy.BF16,
            risk_tier=PhaseRiskTier.LOW,
        ),
        PhasePrecisionRequest(
            phase=ServingPhase.AMC_MEMORY_PREFILL,
            requested_precision=PrecisionMode.BF16,
            fallback_policy=FallbackPolicy.BF16,
            risk_tier=PhaseRiskTier.LOW,
        ),
        PhasePrecisionRequest(
            phase=ServingPhase.DECODE,
            requested_precision=PrecisionMode.BF16,
            fallback_policy=FallbackPolicy.BF16,
            risk_tier=PhaseRiskTier.HIGH,
        ),
        PhasePrecisionRequest(
            phase=ServingPhase.SAFETY_VERIFIER,
            requested_precision=PrecisionMode.BF16,
            fallback_policy=FallbackPolicy.FP32,
            risk_tier=PhaseRiskTier.CRITICAL,
        ),
        PhasePrecisionRequest(
            phase=ServingPhase.COMMIT_AUDIT,
            requested_precision=PrecisionMode.FP32,
            fallback_policy=FallbackPolicy.FP32,
            risk_tier=PhaseRiskTier.CRITICAL,
        ),
    )


def resolve_phase_precision(
    request: PhasePrecisionRequest,
    *,
    runner: NvidiaSmiRunner | None = None,
) -> PhasePrecisionDecision:
    """Resolve one phase request to an effective precision with explicit fallbacks."""
    capability = detect_precision_capability(request.requested_precision, runner=runner)
    warnings: list[str] = []
    fallback_target = _fallback_precision(request.fallback_policy)

    phase_block = _phase_forbids_precision(
        request.phase,
        request.requested_precision,
        allow_experimental=request.allow_experimental,
    )
    if phase_block:
        warnings.append(phase_block)
        return PhasePrecisionDecision(
            phase=request.phase,
            requested_precision=request.requested_precision,
            effective_precision=fallback_target,
            decision=PrecisionDecision.FALLBACK,
            fallback_reason=phase_block,
            capability=capability,
            risk_tier=request.risk_tier,
            warnings=tuple(warnings),
            metadata={"conservative_phase_policy": True},
        )

    if capability.support == PrecisionSupport.SUPPORTED:
        return PhasePrecisionDecision(
            phase=request.phase,
            requested_precision=request.requested_precision,
            effective_precision=request.requested_precision,
            decision=PrecisionDecision.USE_REQUESTED,
            fallback_reason=None,
            capability=capability,
            risk_tier=request.risk_tier,
            warnings=tuple(warnings),
        )

    if capability.support == PrecisionSupport.UNKNOWN:
        warnings.append(f"{request.requested_precision} capability unknown; conservative fallback")
        if request.fallback_policy == FallbackPolicy.ERROR:
            return PhasePrecisionDecision(
                phase=request.phase,
                requested_precision=request.requested_precision,
                effective_precision=fallback_target,
                decision=PrecisionDecision.REJECT,
                fallback_reason=capability.reason,
                capability=capability,
                risk_tier=request.risk_tier,
                warnings=tuple(warnings),
            )
        return PhasePrecisionDecision(
            phase=request.phase,
            requested_precision=request.requested_precision,
            effective_precision=fallback_target,
            decision=PrecisionDecision.FALLBACK,
            fallback_reason=capability.reason,
            capability=capability,
            risk_tier=request.risk_tier,
            warnings=tuple(warnings),
        )

    # UNSUPPORTED
    reason = capability.reason
    if request.fallback_policy == FallbackPolicy.ERROR:
        warnings.append(f"unsupported precision rejected: {reason}")
        return PhasePrecisionDecision(
            phase=request.phase,
            requested_precision=request.requested_precision,
            effective_precision=fallback_target,
            decision=PrecisionDecision.REJECT,
            fallback_reason=reason,
            capability=capability,
            risk_tier=request.risk_tier,
            warnings=tuple(warnings),
        )

    warnings.append(f"unsupported precision fallback: {reason}")
    return PhasePrecisionDecision(
        phase=request.phase,
        requested_precision=request.requested_precision,
        effective_precision=fallback_target,
        decision=PrecisionDecision.FALLBACK,
        fallback_reason=reason,
        capability=capability,
        risk_tier=request.risk_tier,
        warnings=tuple(warnings),
    )


def _plan_hash_payload(plan: PhasePrecisionPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "default_precision": str(plan.default_precision),
        "decisions": [
            {
                "phase": str(d.phase),
                "requested_precision": str(d.requested_precision),
                "effective_precision": str(d.effective_precision),
                "decision": str(d.decision),
                "fallback_reason": d.fallback_reason,
            }
            for d in sorted(plan.decisions, key=lambda item: str(item.phase))
        ],
    }


def plan_hash(plan: PhasePrecisionPlan) -> str:
    """Stable hash; changes when effective phase precision changes."""
    return stable_sha256(_plan_hash_payload(plan))


def build_phase_precision_plan(
    requests: Sequence[PhasePrecisionRequest] | None = None,
    default_precision: PrecisionMode = PrecisionMode.BF16,
    metadata: Mapping[str, Any] | None = None,
    *,
    runner: NvidiaSmiRunner | None = None,
) -> PhasePrecisionPlan:
    """Build a full phase plan from requests (defaults when None)."""
    req_list = tuple(requests) if requests is not None else default_phase_precision_requests()
    decisions = tuple(resolve_phase_precision(request, runner=runner) for request in req_list)
    provisional = PhasePrecisionPlan(
        schema_version=SCHEMA_VERSION,
        plan_id="pending",
        decisions=decisions,
        created_at=_utc_now_iso(),
        default_precision=default_precision,
        metadata=dict(metadata or {}),
    )
    digest = plan_hash(provisional)
    return PhasePrecisionPlan(
        schema_version=SCHEMA_VERSION,
        plan_id=f"ppdq:{digest[:24]}",
        decisions=decisions,
        created_at=provisional.created_at,
        default_precision=default_precision,
        metadata=provisional.metadata,
    )


def validate_phase_precision_plan(plan: PhasePrecisionPlan) -> None:
    """Fail closed on missing phases or unsafe effective precisions."""
    phases_seen = {decision.phase for decision in plan.decisions}
    missing = [phase for phase in ALL_SERVING_PHASES if phase not in phases_seen]
    if missing:
        raise ValueError(f"plan missing phase decisions: {missing}")

    for decision in plan.decisions:
        if decision.effective_precision == PrecisionMode.UNKNOWN:
            raise ValueError(f"phase {decision.phase} has UNKNOWN effective precision")
        if decision.phase == ServingPhase.SAFETY_VERIFIER and decision.effective_precision in (
            PrecisionMode.NVFP4,
            PrecisionMode.INT8,
        ):
            raise ValueError("safety_verifier effective precision is unsafe")
        if decision.phase == ServingPhase.COMMIT_AUDIT and decision.effective_precision in (
            PrecisionMode.NVFP4,
            PrecisionMode.INT8,
        ):
            raise ValueError("commit_audit effective precision is unsafe")

    sanitize_precision_payload(dict(plan.metadata))


def build_phase_telemetry_record(
    *,
    phase: ServingPhase,
    effective_precision: PrecisionMode,
    latency_ms: float | None = None,
    tokens: int | None = None,
    memory_bytes: int | None = None,
    quality_delta: float | None = None,
    safety_reject_count: int | None = None,
    warnings: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> PhaseTelemetryRecord:
    """Build per-phase telemetry with validated numeric placeholders."""
    return PhaseTelemetryRecord(
        phase=phase,
        effective_precision=effective_precision,
        latency_ms=latency_ms,
        tokens=tokens,
        memory_bytes=memory_bytes,
        quality_delta=quality_delta,
        safety_reject_count=safety_reject_count,
        warnings=tuple(warnings),
        metadata=dict(metadata or {}),
    )
