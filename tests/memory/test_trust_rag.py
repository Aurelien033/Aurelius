"""Tests for TrustRAG — quarantine-aware, trust-bound retrieval controller."""

from __future__ import annotations

import pytest

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState
from src.memory.trust_rag import TrustRAGConfig, TrustRAGController


def _block(
    block_id: str,
    tokens: tuple[int, ...] = (1, 2, 3),
    trust_state: TrustState = TrustState.UNVERIFIED,
    provenance: str = "",
    quarantine_state: str = "",
    revocation_epoch: int = 0,
) -> AMCMemoryBlock:
    return AMCMemoryBlock(
        block_id=block_id,
        tokens=tokens,
        trust_state=trust_state,
        provenance=provenance,
        quarantine_state=quarantine_state,
        revocation_epoch=revocation_epoch,
    )


# ── Config ─────────────────────────────────────────────────────────────────


def test_trust_rag_config_defaults() -> None:
    cfg = TrustRAGConfig()
    assert cfg.min_trust_level == TrustState.UNVERIFIED
    assert cfg.max_unverified == 5
    assert cfg.quarantine_excludes_retrieval is True
    assert cfg.detect_contradiction is True


def test_trust_rag_config_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="max_unverified"):
        TrustRAGConfig(max_unverified=-1)

    with pytest.raises(TypeError, match="TrustState"):
        TrustRAGConfig(min_trust_level="unverified")  # type: ignore[arg-type]


# ── Retrieval behavior ─────────────────────────────────────────────────────


def test_retrieve_excludes_quarantined_blocks() -> None:
    store = [
        _block("a", quarantine_state="poisoned"),
        _block("b"),
        _block("c", quarantine_state="contradicted"),
    ]
    result = TrustRAGController().retrieve(store)
    assert result.retrieved_ids == ("b",)
    assert set(result.quarantine_ids) == {"a", "c"}


def test_retrieve_enforces_min_trust_verified() -> None:
    store = [
        _block("a", trust_state=TrustState.UNVERIFIED),
        _block("b", trust_state=TrustState.VERIFIED),
        _block("c", trust_state=TrustState.UNVERIFIED),
    ]
    cfg = TrustRAGConfig(min_trust_level=TrustState.VERIFIED, max_unverified=99)
    result = TrustRAGController(cfg).retrieve(store)
    assert result.retrieved_ids == ("b",)
    assert result.trust_distribution == (("verified", 1),)


def test_retrieve_caps_unverified_at_max_unverified() -> None:
    store = [_block(f"u{i}") for i in range(10)]  # all UNVERIFIED by default
    cfg = TrustRAGConfig(max_unverified=3)
    result = TrustRAGController(cfg).retrieve(store, max_retrieve=20)
    # Cap on UNVERIFIED, even though max_retrieve is high
    assert len(result.retrieved_ids) == 3
    assert result.retrieved_ids == ("u0", "u1", "u2")


def test_retrieve_detects_same_tokens_diff_provenance_contradiction() -> None:
    store = [
        _block("p1", tokens=(10, 11, 12), provenance="source_A"),
        _block("p2", tokens=(10, 11, 12), provenance="source_B"),  # same tokens
        _block("ok", tokens=(99,), provenance="source_C"),  # different tokens
    ]
    result = TrustRAGController().retrieve(store)
    assert len(result.contradiction_pairs) == 1
    (id_a, id_b) = result.contradiction_pairs[0]
    assert {id_a, id_b} == {"p1", "p2"}


def test_retrieve_detect_contradiction_false_yields_no_pairs() -> None:
    store = [
        _block("p1", tokens=(1, 2), provenance="A"),
        _block("p2", tokens=(1, 2), provenance="B"),
    ]
    cfg = TrustRAGConfig(detect_contradiction=False)
    result = TrustRAGController(cfg).retrieve(store)
    assert result.contradiction_pairs == ()


def test_retrieve_flags_revocation_epoch_nonzero() -> None:
    store = [
        _block("a", revocation_epoch=0),
        _block("b", revocation_epoch=3),
        _block("c", revocation_epoch=1),
    ]
    result = TrustRAGController().retrieve(store)
    assert set(result.revocation_flags) == {"b", "c"}


def test_retrieve_emits_trust_distribution() -> None:
    store = [
        _block("u1", trust_state=TrustState.UNVERIFIED),
        _block("u2", trust_state=TrustState.UNVERIFIED),
        _block("v1", trust_state=TrustState.VERIFIED),
    ]
    cfg = TrustRAGConfig(max_unverified=10)
    result = TrustRAGController(cfg).retrieve(store)
    dist = dict(result.trust_distribution)
    assert dist == {"unverified": 2, "verified": 1}


def test_retrieve_returns_empty_on_empty_store() -> None:
    result = TrustRAGController().retrieve([])
    assert result.retrieved_ids == ()
    assert result.quarantine_ids == ()
    assert result.revocation_flags == ()
    assert result.contradiction_pairs == ()
    assert result.trust_distribution == ()
    assert result.retrieved_tokens == ()


def test_retrieve_preserves_iteration_order() -> None:
    store = [_block(f"x{i}") for i in range(5)]
    cfg = TrustRAGConfig(max_unverified=10)
    result = TrustRAGController(cfg).retrieve(store, max_retrieve=10)
    assert result.retrieved_ids == ("x0", "x1", "x2", "x3", "x4")


def test_result_fields_are_immutable_tuples() -> None:
    store = [_block("only")]
    result = TrustRAGController().retrieve(store)
    # Frozen dataclass
    with pytest.raises(AttributeError):
        result.retrieved_ids = ("other",)  # type: ignore[misc]
    # Fields are tuples, not lists
    assert isinstance(result.retrieved_ids, tuple)
    assert isinstance(result.quarantine_ids, tuple)
    assert isinstance(result.contradiction_pairs, tuple)
    assert isinstance(result.revocation_flags, tuple)
    assert isinstance(result.trust_distribution, tuple)
    assert isinstance(result.retrieved_tokens, tuple)


def test_retrieve_uses_custom_contradiction_fn() -> None:
    """Provide a custom contradiction_fn to detect semantic contradictions
    based on provenance text rather than token equality."""
    store = [
        _block("p1", tokens=(1, 2, 3), provenance="Paris is the capital of France"),
        _block("p2", tokens=(4, 5, 6), provenance="The capital of France is Lyon"),
    ]

    # Simple mock semantic contradiction: returns True if 'Lyon' is in one
    # and 'Paris' in the other (mocking NLI-style judgment)
    def mock_contradiction(a: str, b: str) -> bool:
        return ("Paris" in a and "Lyon" in b) or ("Paris" in b and "Lyon" in a)

    cfg = TrustRAGConfig(contradiction_fn=mock_contradiction)
    result = TrustRAGController(cfg).retrieve(store)

    assert len(result.contradiction_pairs) == 1
    assert {result.contradiction_pairs[0][0], result.contradiction_pairs[0][1]} == {"p1", "p2"}


def test_retrieve_custom_contradiction_fn_does_not_fire_on_no_contradiction() -> None:
    store = [
        _block("a", tokens=(1, 2), provenance="Paris is in France"),
        _block("b", tokens=(3, 4), provenance="Berlin is in Germany"),
    ]

    def mock_contradiction(a: str, b: str) -> bool:
        return ("Paris" in a and "Lyon" in b) or ("Paris" in b and "Lyon" in a)

    cfg = TrustRAGConfig(contradiction_fn=mock_contradiction)
    result = TrustRAGController(cfg).retrieve(store)

    assert result.contradiction_pairs == ()
