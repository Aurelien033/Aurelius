"""Trust-aware KV cache for AMC inference serving.

The :class:`AMCPrefixCompiler` produces trust-state-bound cache keys. Identical
content under different trust labels maps to different fingerprints, so
quarantined memory cannot masquerade as verified and trust degradation
invalidates stale pages.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from src.memory.amc_runtime_cache import (
    AMCMemoryBlock,
    AMCMemoryCacheKey,
    AMCPrefixCompiler,
    AMCPrefixCompileResult,
    AMCPrefixSegment,
    TrustState,
)


@dataclass
class KVPageAllocation:
    """One cached KV page keyed by trust-aware fingerprint."""

    cache_key: AMCMemoryCacheKey
    tier: int
    tokens: tuple[int, ...]
    kv_ref: Any
    trust_state: TrustState
    last_used: float = 0.0
    dirty: bool = False


# Alias used in serving integration docs.
KVPage = KVPageAllocation

_TRUST_EVICTION_PRIORITY: dict[TrustState, int] = {
    TrustState.REVOKED: 0,
    TrustState.QUARANTINED: 1,
    TrustState.UNVERIFIED: 2,
    TrustState.VERIFIED: 3,
}


class AMCKVCache:
    """Paged KV cache with trust-prioritized eviction."""

    def __init__(
        self,
        *,
        max_pages: int = 1024,
        page_size: int = 16,
        policy_version: str = "v1",
    ) -> None:
        self.max_pages = max_pages
        self.page_size = page_size
        self.policy_version = policy_version
        self.compiler = AMCPrefixCompiler(policy_version=policy_version)
        self.pages: dict[str, KVPageAllocation] = {}
        self.lru: OrderedDict[str, float] = OrderedDict()

    def compile_and_ingest(self, blocks: list[AMCMemoryBlock]) -> AMCPrefixCompileResult:
        """Compile blocks and cache trusted/allowed segments only."""
        result = self.compiler.compile(blocks)
        for seg in result.trusted + result.allowed:
            self._ingest_segment(seg)
        return result

    def ingest_blocks(self, blocks: list[AMCMemoryBlock]) -> AMCPrefixCompileResult:
        """Alias for :meth:`compile_and_ingest`."""
        return self.compile_and_ingest(blocks)

    def _ingest_segment(self, seg: AMCPrefixSegment) -> None:
        fp = seg.cache_key.fingerprint
        if fp in self.pages:
            self.lru.move_to_end(fp)
            self.lru[fp] = seg.last_used
            self.pages[fp].last_used = seg.last_used
            return

        if len(self.pages) >= self.max_pages:
            self._evict_one()

        self.pages[fp] = KVPageAllocation(
            cache_key=seg.cache_key,
            tier=seg.tier,
            tokens=seg.tokens,
            kv_ref=seg.kv_ref,
            trust_state=seg.trust_state,
            last_used=seg.last_used,
        )
        self.lru[fp] = seg.last_used

    def lookup(self, cache_key: AMCMemoryCacheKey) -> KVPageAllocation | None:
        fp = cache_key.fingerprint
        page = self.pages.get(fp)
        if page is not None:
            self.lru.move_to_end(fp)
        return page

    def invalidate_trust_change(
        self,
        block: AMCMemoryBlock,
        new_trust: TrustState,
    ) -> bool:
        """Evict the page keyed by the block's current trust state."""
        _ = new_trust  # new key would differ; caller re-ingests after mutation
        old_fp = block.to_cache_key(self.policy_version).fingerprint
        if old_fp not in self.pages:
            return False
        del self.pages[old_fp]
        self.lru.pop(old_fp, None)
        return True

    def _evict_one(self) -> str | None:
        """Evict lowest-trust, then oldest LRU page."""
        if not self.lru:
            return None

        candidates: list[tuple[int, float, str]] = []
        for fp, ts in self.lru.items():
            page = self.pages[fp]
            prio = _TRUST_EVICTION_PRIORITY.get(page.trust_state, 2)
            candidates.append((prio, ts, fp))

        candidates.sort(key=lambda item: (item[0], item[1]))
        victim_fp = candidates[0][2]
        del self.pages[victim_fp]
        self.lru.pop(victim_fp)
        return victim_fp

    def evict_to(self, target_count: int) -> int:
        """Evict pages until at most ``target_count`` remain."""
        evicted = 0
        while len(self.pages) > target_count:
            if self._evict_one() is None:
                break
            evicted += 1
        return evicted

    def stats(self) -> dict[str, Any]:
        by_trust = {ts.value: 0 for ts in TrustState}
        for page in self.pages.values():
            by_trust[page.trust_state.value] += 1
        return {
            "total_pages": len(self.pages),
            "max_pages": self.max_pages,
            "utilization": len(self.pages) / self.max_pages if self.max_pages else 0.0,
            "by_trust_state": by_trust,
        }

    def clear(self) -> None:
        self.pages.clear()
        self.lru.clear()


__all__ = [
    "AMCKVCache",
    "KVPage",
    "KVPageAllocation",
]
