"""VEL result registry — verified evolution loop provenance for AMC/eval runs (P0.5 / OC-10).

Machine-readable JSONL registry tracing benchmark numbers to command, config hash,
seed, checkpoint SHA, environment, raw output path, sources, and Pivot/Refine outcome.
Contract-only; no autonomous agent workflow and no writes at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src._compat import StrEnum

name = "vel/v1"
REDACTED = "[REDACTED]"
DEFAULT_REGISTRY_PATH = Path("docs/reports/vel_results.jsonl")

_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "authorization",
    "bearer",
    "connection_string",
    "private_key",
)

_COMMAND_CLAIM_TYPES = frozenset(
    {
        "benchmark",
        "ablation",
        "regression",
        "performance",
    }
)

_FAIL_STATUSES = frozenset({"fail", "error"})
_PIVOT_DECISIONS = frozenset({"refine", "pivot", "reject"})


class VELRegistryError(ValueError):
    """Invalid VEL registry record or IO failure."""


class VELRunStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"


class VELDecision(StrEnum):
    ACCEPT = "accept"
    REFINE = "refine"
    PIVOT = "pivot"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class VELClaimType(StrEnum):
    BENCHMARK = "benchmark"
    ABLATION = "ablation"
    REGRESSION = "regression"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    ROADMAP = "roadmap"
    CITATION = "citation"


class VELSourceType(StrEnum):
    CONFIG = "config"
    RAW_OUTPUT = "raw_output"
    CHECKPOINT = "checkpoint"
    DATASET = "dataset"
    PAPER = "paper"
    CODE = "code"
    COMMAND = "command"
    REPORT = "report"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def sanitize_vel_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_secret_key(str(key)) else sanitize_vel_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_vel_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_vel_value(item) for item in value]
    return value


def _sanitize_command_arg(arg: str) -> str:
    stripped = arg.strip()
    if not stripped:
        return arg
    if stripped.lower().startswith("authorization:") and "bearer" in stripped.lower():
        return "Authorization: [REDACTED]"
    if "=" in stripped:
        key, _, val = stripped.partition("=")
        if _is_secret_key(key.lstrip("-")):
            return f"{key}={REDACTED}"
        return stripped
    if stripped.startswith("--") and " " not in stripped:
        return stripped
    parts = stripped.split()
    if len(parts) >= 2 and parts[0].startswith("--") and _is_secret_key(parts[0].lstrip("-")):
        return f"{parts[0]} {REDACTED}"
    if len(parts) == 2 and "=" not in parts[0] and _is_secret_key(parts[0]):
        return f"{parts[0]}={REDACTED}"
    return stripped


def sanitize_vel_command(command: Sequence[str]) -> tuple[str, ...]:
    return tuple(_sanitize_command_arg(str(part)) for part in command)


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise VELRegistryError("non-finite float cannot be serialized")
        return value
    raise VELRegistryError(f"non-JSON-compatible value: {type(value).__name__}")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class VELMetric:
    name: str
    value: int | float | str | bool
    unit: str | None = None
    split: str | None = None
    higher_is_better: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise VELRegistryError("metric name must be non-empty")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise VELRegistryError(f"metric {self.name!r} must be finite")
        meta = sanitize_vel_value(dict(self.metadata))
        object.__setattr__(self, "metadata", meta if isinstance(meta, dict) else {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "split": self.split,
            "higher_is_better": self.higher_is_better,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VELMetric:
        return cls(
            name=str(data["name"]),
            value=data["value"],  # type: ignore[arg-type]
            unit=data.get("unit"),
            split=data.get("split"),
            higher_is_better=data.get("higher_is_better"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class VELSource:
    source_id: str
    source_type: VELSourceType
    path_or_url: str
    sha256: str | None = None
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise VELRegistryError("source_id must be non-empty")
        if not self.path_or_url.strip():
            raise VELRegistryError("path_or_url must be non-empty")
        object.__setattr__(self, "source_type", VELSourceType(self.source_type))
        meta = sanitize_vel_value(dict(self.metadata))
        object.__setattr__(self, "metadata", meta if isinstance(meta, dict) else {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": str(self.source_type),
            "path_or_url": self.path_or_url,
            "sha256": self.sha256,
            "title": self.title,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VELSource:
        return cls(
            source_id=str(data["source_id"]),
            source_type=VELSourceType(str(data["source_type"])),
            path_or_url=str(data["path_or_url"]),
            sha256=data.get("sha256"),
            title=data.get("title"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class VELRunEnvironment:
    python_version: str
    platform: str
    machine: str | None = None
    processor: str | None = None
    cuda_available: bool | None = None
    gpu_name: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        meta = sanitize_vel_value(dict(self.metadata))
        object.__setattr__(self, "metadata", meta if isinstance(meta, dict) else {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VELRunEnvironment:
        return cls(
            python_version=str(data["python_version"]),
            platform=str(data["platform"]),
            machine=data.get("machine"),
            processor=data.get("processor"),
            cuda_available=data.get("cuda_available"),
            gpu_name=data.get("gpu_name"),
            git_commit=data.get("git_commit"),
            git_dirty=data.get("git_dirty"),
            metadata=dict(data.get("metadata") or {}),
        )


def collect_vel_environment(*, git: bool = True) -> VELRunEnvironment:
    cuda_available: bool | None = None
    gpu_name: str | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        cuda_available = None
        gpu_name = None

    git_commit: str | None = None
    git_dirty: bool | None = None
    if git:
        git_bin = shutil.which("git")
        if git_bin:
            try:
                commit = subprocess.run(  # noqa: S603
                    [git_bin, "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                git_commit = commit.stdout.strip() or None
                status = subprocess.run(  # noqa: S603
                    [git_bin, "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                git_dirty = bool(status.stdout.strip())
            except (OSError, subprocess.CalledProcessError):
                git_commit = None
                git_dirty = None

    return VELRunEnvironment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine() or None,
        processor=platform.processor() or None,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        git_commit=git_commit,
        git_dirty=git_dirty,
        metadata={},
    )


@dataclass(frozen=True)
class VELResultRecord:
    record_id: str
    created_at: str
    claim_type: VELClaimType
    run_status: VELRunStatus
    decision: VELDecision
    command: tuple[str, ...]
    config_path: str | None
    config_sha256: str | None
    seed: int | None
    checkpoint_sha: str | None
    raw_output_path: str | None
    metrics: tuple[VELMetric, ...]
    sources: tuple[VELSource, ...]
    environment: VELRunEnvironment
    pivot_or_refine_reason: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise VELRegistryError("record_id must be non-empty")
        claim = VELClaimType(self.claim_type)
        status = VELRunStatus(self.run_status)
        decision = VELDecision(self.decision)
        object.__setattr__(self, "claim_type", claim)
        object.__setattr__(self, "run_status", status)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "command", sanitize_vel_command(self.command))

        if str(claim) in _COMMAND_CLAIM_TYPES and not self.command:
            raise VELRegistryError(f"{claim!r} claims require a non-empty command")

        if self.config_path:
            cfg = Path(self.config_path)
            if cfg.is_file() and not self.config_sha256:
                object.__setattr__(self, "config_sha256", sha256_file(cfg))

        meta = sanitize_vel_value(dict(self.metadata))
        if not isinstance(meta, dict):
            meta = {}
        object.__setattr__(self, "metadata", meta)

        if str(claim) in _COMMAND_CLAIM_TYPES:
            if not self.raw_output_path and not meta.get("raw_output_missing_reason"):
                raise VELRegistryError(
                    "benchmark-like claims require raw_output_path or "
                    "metadata['raw_output_missing_reason']"
                )

        if str(status) in _FAIL_STATUSES and str(decision) == VELDecision.ACCEPT:
            if not meta.get("accept_override_reason"):
                raise VELRegistryError(
                    f"{status!r} runs cannot use ACCEPT without metadata accept_override_reason"
                )

        if str(decision) in _PIVOT_DECISIONS:
            if not self.pivot_or_refine_reason or not str(self.pivot_or_refine_reason).strip():
                raise VELRegistryError(
                    f"{decision!r} decisions require pivot_or_refine_reason"
                )

    def to_dict(self, *, include_record_hash: bool = True) -> dict[str, Any]:
        body = _record_body_dict(self)
        if include_record_hash:
            body["record_hash"] = compute_record_hash(self)
        return body

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VELResultRecord:
        return cls(
            record_id=str(data["record_id"]),
            created_at=str(data["created_at"]),
            claim_type=VELClaimType(str(data["claim_type"])),
            run_status=VELRunStatus(str(data["run_status"])),
            decision=VELDecision(str(data["decision"])),
            command=tuple(str(part) for part in data.get("command", ())),
            config_path=data.get("config_path"),
            config_sha256=data.get("config_sha256"),
            seed=data.get("seed"),
            checkpoint_sha=data.get("checkpoint_sha"),
            raw_output_path=data.get("raw_output_path"),
            metrics=tuple(VELMetric.from_dict(item) for item in data.get("metrics", ())),
            sources=tuple(VELSource.from_dict(item) for item in data.get("sources", ())),
            environment=VELRunEnvironment.from_dict(data["environment"]),
            pivot_or_refine_reason=data.get("pivot_or_refine_reason"),
            notes=data.get("notes"),
            metadata=dict(data.get("metadata") or {}),
        )


def _record_body_dict(record: VELResultRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "created_at": record.created_at,
        "claim_type": str(record.claim_type),
        "run_status": str(record.run_status),
        "decision": str(record.decision),
        "command": list(record.command),
        "config_path": record.config_path,
        "config_sha256": record.config_sha256,
        "seed": record.seed,
        "checkpoint_sha": record.checkpoint_sha,
        "raw_output_path": record.raw_output_path,
        "metrics": [metric.to_dict() for metric in record.metrics],
        "sources": [source.to_dict() for source in record.sources],
        "environment": record.environment.to_dict(),
        "pivot_or_refine_reason": record.pivot_or_refine_reason,
        "notes": record.notes,
        "metadata": dict(record.metadata),
    }


def compute_record_hash(record: VELResultRecord) -> str:
    return hashlib.sha256(stable_json_dumps(_record_body_dict(record)).encode("utf-8")).hexdigest()


def make_record_id(record: VELResultRecord) -> str:
    return compute_record_hash(record)[:16]


def build_vel_record(
    *,
    claim_type: VELClaimType | str,
    run_status: VELRunStatus | str,
    decision: VELDecision | str,
    command: Sequence[str],
    config_path: str | Path | None = None,
    config_sha256: str | None = None,
    seed: int | None = None,
    checkpoint_sha: str | None = None,
    raw_output_path: str | None = None,
    metrics: Sequence[VELMetric | Mapping[str, Any]] = (),
    sources: Sequence[VELSource] = (),
    environment: VELRunEnvironment | None = None,
    pivot_or_refine_reason: str | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    record_id: str | None = None,
    created_at: str | None = None,
) -> VELResultRecord:
    cfg_path = str(config_path) if config_path is not None else None
    cfg_hash = config_sha256
    if cfg_path and not cfg_hash and Path(cfg_path).is_file():
        cfg_hash = sha256_file(cfg_path)

    metric_objs = tuple(
        metric if isinstance(metric, VELMetric) else VELMetric.from_dict(metric)
        for metric in metrics
    )

    draft = VELResultRecord(
        record_id=record_id or "pending",
        created_at=created_at or datetime.now(UTC).isoformat(),
        claim_type=VELClaimType(claim_type),
        run_status=VELRunStatus(run_status),
        decision=VELDecision(decision),
        command=tuple(str(part) for part in command),
        config_path=cfg_path,
        config_sha256=cfg_hash,
        seed=seed,
        checkpoint_sha=checkpoint_sha,
        raw_output_path=raw_output_path,
        metrics=metric_objs,
        sources=tuple(sources),
        environment=environment or collect_vel_environment(),
        pivot_or_refine_reason=pivot_or_refine_reason,
        notes=notes,
        metadata=dict(metadata or {}),
    )
    rid = record_id or make_record_id(draft)
    return VELResultRecord(
        record_id=rid,
        created_at=draft.created_at,
        claim_type=draft.claim_type,
        run_status=draft.run_status,
        decision=draft.decision,
        command=draft.command,
        config_path=draft.config_path,
        config_sha256=draft.config_sha256,
        seed=draft.seed,
        checkpoint_sha=draft.checkpoint_sha,
        raw_output_path=draft.raw_output_path,
        metrics=draft.metrics,
        sources=draft.sources,
        environment=draft.environment,
        pivot_or_refine_reason=draft.pivot_or_refine_reason,
        notes=draft.notes,
        metadata=draft.metadata,
    )


class VELRegistry:
    """Append/read JSONL VEL result records."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path)

    def append(self, record: VELResultRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = stable_json_dumps(record.to_dict()) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def read_all(self, *, permissive: bool = False) -> list[VELResultRecord]:
        if not self.path.exists():
            return []
        records: list[VELResultRecord] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    if permissive:
                        continue
                    raise VELRegistryError(
                        f"malformed JSON at {self.path}:{line_no}: {exc}"
                    ) from exc
                records.append(VELResultRecord.from_dict(payload))
        return records


def record_from_amc_payload(
    *,
    payload: Mapping[str, Any],
    command: Sequence[str],
    config_path: str | None = None,
    raw_output_path: str | None = None,
    seed: int | None = None,
    run_status: VELRunStatus | str = VELRunStatus.PASS,
    decision: VELDecision | str = VELDecision.ACCEPT,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> VELResultRecord:
    """Optional adapter: build a VEL record from an AMC-Memory runner payload."""
    gate = payload.get("gate") or {}
    passed = gate.get("passed") if isinstance(gate, dict) else None
    metrics = [
        VELMetric(name="overall_score", value=float(payload.get("overall_score", 0.0))),
    ]
    if passed is not None:
        metrics.append(VELMetric(name="gate_passed", value=bool(passed)))
    sources = []
    if raw_output_path:
        sources.append(
            VELSource(
                source_id="raw-output",
                source_type=VELSourceType.RAW_OUTPUT,
                path_or_url=raw_output_path,
            )
        )
    if config_path:
        sources.append(
            VELSource(
                source_id="config",
                source_type=VELSourceType.CONFIG,
                path_or_url=config_path,
                sha256=sha256_file(config_path) if Path(config_path).is_file() else None,
            )
        )
    record = build_vel_record(
        claim_type=VELClaimType.BENCHMARK,
        run_status=run_status,
        decision=decision,
        command=command,
        config_path=config_path,
        seed=seed,
        raw_output_path=raw_output_path,
        metrics=metrics,
        sources=sources,
        metadata={"suite": payload.get("suite"), "generator": payload.get("generator")},
    )
    return record
