"""Tests for trust-aware AMC KV cache (T13)."""

from __future__ import annotations

import time

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState
from src.serving.amc_kv_cache import AMCKVCache, KVPageAllocation


def _block(
    *,
    trust: TrustState = TrustState.UNVERIFIED,
    tid: int = 0,
    last_used: float | None = None,
) -> AMCMemoryBlock:
    return AMCMemoryBlock(
        block_id=f"b{tid}",
        tokens=(1, 2, 3, 4),
        tier=2,
        trust_state=trust,
        provenance=f"test-{tid}",
        salience=0.5,
        surprise_score=0.1,
        last_used=time.monotonic() if last_used is None else last_used,
    )


def _install_page(cache: AMCKVCache, block: AMCMemoryBlock) -> str:
    key = block.to_cache_key(cache.policy_version)
    cache.pages[key.fingerprint] = KVPageAllocation(
        cache_key=key,
        tier=block.tier,
        tokens=block.tokens,
        kv_ref=block.kv_ref,
        trust_state=block.trust_state,
        last_used=block.last_used,
    )
    cache.lru[key.fingerprint] = block.last_used
    return key.fingerprint


def test_ingest_creates_pages() -> None:
    cache = AMCKVCache()
    result = cache.ingest_blocks([_block(trust=TrustState.VERIFIED, tid=1)])
    assert result.total_segments() >= 1
    assert cache.stats()["total_pages"] == 1


def test_quarantined_evicted_before_verified() -> None:
    cache = AMCKVCache(max_pages=2)
    verified_fp = _install_page(cache, _block(trust=TrustState.VERIFIED, tid=1, last_used=1.0))
    quarantined_fp = _install_page(
        cache,
        _block(trust=TrustState.QUARANTINED, tid=2, last_used=2.0),
    )
    victim = cache._evict_one()
    assert victim == quarantined_fp
    assert verified_fp in cache.pages


def test_trust_change_invalidates_old_key() -> None:
    cache = AMCKVCache()
    block = _block(trust=TrustState.VERIFIED, tid=1)
    cache.ingest_blocks([block])
    assert cache.stats()["total_pages"] == 1
    removed = cache.invalidate_trust_change(block, TrustState.UNVERIFIED)
    assert removed is True
    assert cache.stats()["total_pages"] == 0


def test_lookup_returns_none_for_unknown() -> None:
    cache = AMCKVCache()
    block = _block(trust=TrustState.VERIFIED, tid=1)
    key = block.to_cache_key(cache.policy_version)
    assert cache.lookup(key) is None
    cache.ingest_blocks([block])
    assert cache.lookup(key) is not None


def test_evict_to_reduces_count() -> None:
    cache = AMCKVCache(max_pages=10)
    for idx in range(8):
        trust = TrustState.VERIFIED if idx < 2 else TrustState.UNVERIFIED
        cache.ingest_blocks([_block(trust=trust, tid=idx)])
    assert cache.stats()["total_pages"] == 8
    evicted = cache.evict_to(3)
    assert evicted >= 5
    assert cache.stats()["total_pages"] <= 3


def test_stats_by_trust_state() -> None:
    cache = AMCKVCache()
    cache.ingest_blocks(
        [
            _block(trust=TrustState.VERIFIED, tid=1),
            _block(trust=TrustState.VERIFIED, tid=2),
            _block(trust=TrustState.UNVERIFIED, tid=3),
        ]
    )
    stats = cache.stats()
    assert stats["by_trust_state"]["verified"] == 2
    assert stats["by_trust_state"]["unverified"] == 1


def test_quarantined_not_cached() -> None:
    cache = AMCKVCache()
    cache.ingest_blocks([_block(trust=TrustState.QUARANTINED, tid=9)])
    assert cache.stats()["total_pages"] == 0


def test_clear_empties_cache() -> None:
    cache = AMCKVCache()
    cache.ingest_blocks([_block(trust=TrustState.VERIFIED, tid=1)])
    cache.clear()
    assert cache.stats()["total_pages"] == 0


def test_revoked_evicted_before_unverified() -> None:
    cache = AMCKVCache(max_pages=2)
    unverified_fp = _install_page(
        cache,
        _block(trust=TrustState.UNVERIFIED, tid=1, last_used=1.0),
    )
    revoked_fp = _install_page(
        cache,
        _block(trust=TrustState.REVOKED, tid=2, last_used=2.0),
    )
    victim = cache._evict_one()
    assert victim == revoked_fp
    assert unverified_fp in cache.pages
