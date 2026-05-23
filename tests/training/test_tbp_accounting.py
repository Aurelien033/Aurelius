"""Tests for TBP boundary / throughput accounting (P0.4 / OC-12)."""

from __future__ import annotations

import pytest

from src.training.tbp_accounting import (
    BOUNDARY_KINDS,
    REDACTED,
    BoundaryKind,
    BoundarySpan,
    TBPAccounting,
    build_boundary_mask,
    build_memory_boundary_mask,
    compute_tbp_accounting,
    compute_token_fertility,
    count_raw_bytes,
    sanitize_metadata,
)

# ── Test 1: count_raw_bytes ──────────────────────────────────────────────────


def test_count_raw_bytes_ascii_multibyte_and_bytes() -> None:
    assert count_raw_bytes("abc") == 3
    assert count_raw_bytes("café") == 5
    assert count_raw_bytes("café".encode()) == 5


# ── Test 2–3: compute_token_fertility ────────────────────────────────────────


def test_compute_token_fertility_normal() -> None:
    assert compute_token_fertility(10, 20) == pytest.approx(0.5)


def test_compute_token_fertility_zero_bytes() -> None:
    assert compute_token_fertility(0, 0) == 0.0
    with pytest.raises(ValueError):
        compute_token_fertility(3, 0)


# ── Test 4: BoundarySpan validation ──────────────────────────────────────────


def test_boundary_span_rejects_invalid_ranges() -> None:
    BoundarySpan(start=0, end=2, kind=BoundaryKind.TOKEN)
    with pytest.raises(ValueError, match="start"):
        BoundarySpan(start=3, end=1, kind="token")
    with pytest.raises(ValueError, match="unknown boundary kind"):
        BoundarySpan(start=0, end=1, kind="not_real")


def test_boundary_kinds_allowed() -> None:
    assert "memory_span" in BOUNDARY_KINDS
    assert BoundaryKind.MEMORY_SPAN.value == "memory_span"


# ── Test 5–7: build_boundary_mask ────────────────────────────────────────────


def test_build_boundary_mask_marks_positions() -> None:
    spans = [
        BoundarySpan(start=1, end=3, kind=BoundaryKind.TOKEN),
        BoundarySpan(start=5, end=5, kind=BoundaryKind.WORD),
    ]
    mask = build_boundary_mask(6, spans)
    assert mask == [0, 1, 1, 0, 0, 1]


def test_build_boundary_mask_filters_by_kind() -> None:
    spans = [
        BoundarySpan(start=0, end=2, kind=BoundaryKind.TOKEN),
        BoundarySpan(start=2, end=4, kind=BoundaryKind.MEMORY_SPAN),
    ]
    token_mask = build_boundary_mask(4, spans, kind=BoundaryKind.TOKEN)
    assert token_mask == [1, 1, 0, 0]


def test_build_boundary_mask_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        build_boundary_mask(3, [BoundarySpan(start=0, end=4, kind="token")])


# ── Test 8: memory boundary mask ─────────────────────────────────────────────


def test_build_memory_boundary_mask_only_memory_spans() -> None:
    spans = [
        BoundarySpan(start=0, end=2, kind=BoundaryKind.TOKEN),
        BoundarySpan(start=2, end=5, kind=BoundaryKind.MEMORY_SPAN),
    ]
    mask = build_memory_boundary_mask(5, spans)
    assert mask == [0, 0, 1, 1, 1]


# ── Test 9–10: compute_tbp_accounting real_length ────────────────────────────


def test_compute_tbp_accounting_uses_explicit_real_length() -> None:
    token_ids = [0, 0, 0, 99, 99]
    accounting = compute_tbp_accounting(
        token_ids=token_ids,
        raw_text="hello",
        real_length=3,
        pad_token_id=0,
    )
    assert accounting.real_tokens_seen == 3
    assert accounting.pad_tokens_seen == 2


def test_compute_tbp_accounting_real_tokens_equal_pad_token_id() -> None:
    token_ids = [7, 0, 0, 99, 99]
    accounting = compute_tbp_accounting(
        token_ids=token_ids,
        raw_text="abc",
        real_length=3,
        pad_token_id=0,
    )
    assert accounting.real_tokens_seen == 3


# ── Test 11–12: raw bytes, fertility, throughput ─────────────────────────────


def test_compute_tbp_accounting_raw_bytes_and_fertility() -> None:
    accounting = compute_tbp_accounting(
        token_ids=[1, 2, 3, 4],
        raw_text="abcd",
        real_length=4,
    )
    assert accounting.raw_bytes_seen == 4
    assert accounting.fertility_tokens_per_byte == pytest.approx(1.0)
    assert accounting.bytes_per_token == pytest.approx(1.0)


def test_effective_throughput_only_when_elapsed_valid() -> None:
    no_time = compute_tbp_accounting(token_ids=[1, 2], raw_text="ab", real_length=2)
    assert no_time.effective_sample_throughput is None

    with_time = compute_tbp_accounting(
        token_ids=[1, 2],
        raw_text="ab",
        real_length=2,
        elapsed_seconds=2.0,
    )
    assert with_time.effective_sample_throughput == pytest.approx(1.0)

    with pytest.raises(ValueError):
        compute_tbp_accounting(token_ids=[1], raw_text="a", elapsed_seconds=0.0)


def test_flops_normalized_bytes_in_metadata() -> None:
    accounting = compute_tbp_accounting(
        token_ids=[1, 2, 3],
        raw_text="abc",
        real_length=3,
        flops=3.0,
    )
    assert accounting.metadata["flops_normalized_bytes"] == pytest.approx(1.0)


# ── Test 13: metadata sanitization ───────────────────────────────────────────


def test_metadata_sanitized_on_serialization() -> None:
    accounting = compute_tbp_accounting(
        token_ids=[1],
        metadata={"api_key": "secret", "note": "ok"},
    )
    blob = accounting.to_json()
    assert REDACTED in blob
    assert "secret" not in blob
    assert sanitize_metadata({"password": "x"})["password"] == REDACTED


def test_tbp_accounting_negative_counts_rejected() -> None:
    with pytest.raises(ValueError, match="tokens_seen"):
        TBPAccounting(
            tokens_seen=-1,
            raw_bytes_seen=0,
            real_tokens_seen=0,
            pad_tokens_seen=0,
            fertility_tokens_per_byte=0.0,
            bytes_per_token=0.0,
            boundary_count=0,
            memory_boundary_count=0,
            effective_sample_throughput=None,
            boundary_prior_mode="none",
        )
