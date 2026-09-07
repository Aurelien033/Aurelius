"""TBP boundary / throughput accounting (P0.4 / OC-12).

Raw-byte counts, token fertility, boundary masks, memory-span masks, and
effective sample throughput for honest tokenizer/packing comparisons.
Contract/telemetry only — does not change tokenization or training behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from src._compat import StrEnum

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

name = "tbp/v1"

BOUNDARY_KINDS = frozenset(
    {
        "token",
        "word",
        "subword",
        "entity",
        "tool",
        "citation",
        "code",
        "memory_span",
        "turn",
    }
)


class BoundaryKind(StrEnum):
    TOKEN = "token"
    WORD = "word"
    SUBWORD = "subword"
    ENTITY = "entity"
    TOOL = "tool"
    CITATION = "citation"
    CODE = "code"
    MEMORY_SPAN = "memory_span"
    TURN = "turn"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def sanitize_metadata(value: Any) -> Any:
    """Recursively redact secret-bearing keys in metadata mappings."""
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_secret_key(str(key)) else sanitize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_metadata(item) for item in value]
    return value


def count_raw_bytes(text_or_bytes: str | bytes) -> int:
    if isinstance(text_or_bytes, bytes):
        return len(text_or_bytes)
    return len(text_or_bytes.encode("utf-8"))


def compute_token_fertility(token_count: int, raw_bytes: int) -> float:
    if raw_bytes == 0:
        if token_count == 0:
            return 0.0
        raise ValueError("token_count > 0 requires raw_bytes > 0 for fertility")
    return token_count / raw_bytes


def _normalize_kind(kind: str | BoundaryKind) -> str:
    normalized = str(kind).strip().lower()
    if normalized not in BOUNDARY_KINDS:
        raise ValueError(f"unknown boundary kind {kind!r}; allowed: {sorted(BOUNDARY_KINDS)}")
    return normalized


@dataclass(frozen=True)
class BoundarySpan:
    start: int
    end: int
    kind: str | BoundaryKind
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("start and end must be >= 0")
        if self.start > self.end:
            raise ValueError(f"start must be <= end, got {self.start} > {self.end}")
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        meta = sanitize_metadata(dict(self.metadata))
        object.__setattr__(self, "metadata", meta if isinstance(meta, dict) else {})


@dataclass(frozen=True)
class TBPAccounting:
    tokens_seen: int
    raw_bytes_seen: int
    real_tokens_seen: int
    pad_tokens_seen: int
    fertility_tokens_per_byte: float
    bytes_per_token: float
    boundary_count: int
    memory_boundary_count: int
    effective_sample_throughput: float | None
    boundary_prior_mode: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("tokens_seen", self.tokens_seen),
            ("raw_bytes_seen", self.raw_bytes_seen),
            ("real_tokens_seen", self.real_tokens_seen),
            ("pad_tokens_seen", self.pad_tokens_seen),
            ("boundary_count", self.boundary_count),
            ("memory_boundary_count", self.memory_boundary_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        meta = sanitize_metadata(dict(self.metadata))
        object.__setattr__(self, "metadata", meta if isinstance(meta, dict) else {})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def build_boundary_mask(
    length: int,
    spans: Sequence[BoundarySpan],
    *,
    kind: str | BoundaryKind | None = None,
) -> list[int]:
    if length < 0:
        raise ValueError("length must be >= 0")
    mask = [0] * length
    filter_kind = _normalize_kind(kind) if kind is not None else None
    for span in spans:
        if filter_kind is not None and span.kind != filter_kind:
            continue
        if span.end > length:
            raise ValueError(
                f"span end {span.end} exceeds mask length {length} for kind {span.kind!r}"
            )
        if span.start >= length and span.start != span.end:
            raise ValueError(f"span start {span.start} out of range for length {length}")
        if span.start == span.end:
            if span.start < length:
                mask[span.start] = 1
            continue
        for pos in range(span.start, min(span.end, length)):
            mask[pos] = 1
    return mask


def build_memory_boundary_mask(length: int, spans: Sequence[BoundarySpan]) -> list[int]:
    return build_boundary_mask(length, spans, kind=BoundaryKind.MEMORY_SPAN)


def _infer_real_tokens(
    token_ids: Sequence[int],
    *,
    real_length: int | None,
    pad_token_id: int | None,
) -> int:
    if real_length is not None:
        if real_length < 0 or real_length > len(token_ids):
            raise ValueError(f"real_length must be in [0, {len(token_ids)}], got {real_length}")
        return real_length
    if pad_token_id is None:
        return len(token_ids)
    real_len = len(token_ids)
    for idx in range(len(token_ids) - 1, -1, -1):
        if token_ids[idx] == pad_token_id:
            real_len = idx
        else:
            break
    return real_len


def compute_tbp_accounting(
    *,
    token_ids: Sequence[int],
    raw_text: str | bytes | None = None,
    real_length: int | None = None,
    pad_token_id: int | None = None,
    boundary_spans: Sequence[BoundarySpan] = (),
    elapsed_seconds: float | None = None,
    flops: float | None = None,
    boundary_prior_mode: str = "none",
    metadata: Mapping[str, Any] | None = None,
) -> TBPAccounting:
    tokens_seen = len(token_ids)
    real_tokens_seen = _infer_real_tokens(
        token_ids,
        real_length=real_length,
        pad_token_id=pad_token_id,
    )
    pad_tokens_seen = max(0, tokens_seen - real_tokens_seen)
    raw_bytes_seen = count_raw_bytes(raw_text) if raw_text is not None else 0

    fertility = 0.0
    bytes_per_token = 0.0
    if raw_bytes_seen > 0:
        fertility = compute_token_fertility(real_tokens_seen, raw_bytes_seen)
        bytes_per_token = raw_bytes_seen / real_tokens_seen if real_tokens_seen > 0 else 0.0

    boundary_mask = build_boundary_mask(tokens_seen, boundary_spans)
    memory_mask = build_memory_boundary_mask(tokens_seen, boundary_spans)
    boundary_count = sum(boundary_mask)
    memory_boundary_count = sum(memory_mask)

    throughput: float | None = None
    if elapsed_seconds is not None:
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be > 0 when provided")
        throughput = raw_bytes_seen / elapsed_seconds

    meta: dict[str, Any] = dict(metadata or {})
    if flops is not None:
        if flops <= 0:
            raise ValueError("flops must be > 0 when provided")
        meta["flops_normalized_bytes"] = raw_bytes_seen / flops
    meta["boundary_mask_coverage"] = boundary_count / tokens_seen if tokens_seen else 0.0
    meta["memory_boundary_mask_coverage"] = (
        memory_boundary_count / tokens_seen if tokens_seen else 0.0
    )

    return TBPAccounting(
        tokens_seen=tokens_seen,
        raw_bytes_seen=raw_bytes_seen,
        real_tokens_seen=real_tokens_seen,
        pad_tokens_seen=pad_tokens_seen,
        fertility_tokens_per_byte=fertility,
        bytes_per_token=bytes_per_token,
        boundary_count=boundary_count,
        memory_boundary_count=memory_boundary_count,
        effective_sample_throughput=throughput,
        boundary_prior_mode=boundary_prior_mode,
        metadata=meta,
    )
