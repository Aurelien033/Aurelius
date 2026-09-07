"""TrustRAG — quarantine-aware, trust-bound retrieval controller.

Pure-Python filter over an iterable of AMCMemoryBlocks. Enforces:
- Quarantined blocks excluded from privileged retrieval.
- min_trust_level honored (VERIFIED-only when set).
- Bounded unverified inclusion (configurable max_unverified).
- Contradiction detection on identical-token different-provenance pairs.
- Revocation-epoch surface flagged.

Does NOT implement embedding-based similarity or reranking; the controller
operates on an already-ranked / already-iterable block stream and applies
trust-side constraints.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState


@dataclass(frozen=True)
class TrustRAGConfig:
    min_trust_level: TrustState = TrustState.UNVERIFIED
    max_unverified: int = 5
    quarantine_excludes_retrieval: bool = True
    detect_contradiction: bool = True
    # Optional semantic contradiction function: takes (text_a, text_b) -> bool
    # If None, falls back to token-equality + different provenance.
    contradiction_fn: Callable[[str, str], bool] | None = None

    def __post_init__(self) -> None:
        if self.max_unverified < 0:
            raise ValueError(f"max_unverified must be >= 0, got {self.max_unverified}")
        if not isinstance(self.min_trust_level, TrustState):
            raise TypeError(
                f"min_trust_level must be TrustState, got {type(self.min_trust_level).__name__}"
            )


@dataclass(frozen=True)
class TrustRAGResult:
    retrieved_ids: tuple[str, ...]
    trust_distribution: tuple[tuple[str, int], ...]
    contradiction_pairs: tuple[tuple[str, str], ...]
    quarantine_ids: tuple[str, ...]
    revocation_flags: tuple[str, ...]
    retrieved_tokens: tuple[tuple[int, ...], ...]


class TrustRAGController:
    def __init__(self, config: TrustRAGConfig | None = None) -> None:
        self.cfg = config or TrustRAGConfig()

    def retrieve(
        self,
        store: Iterable[AMCMemoryBlock],
        *,
        max_retrieve: int = 10,
    ) -> TrustRAGResult:
        if max_retrieve < 0:
            raise ValueError(f"max_retrieve must be >= 0, got {max_retrieve}")

        cfg = self.cfg

        retrieved_ids: list[str] = []
        retrieved_tokens: list[tuple[int, ...]] = []
        retrieved_blocks: list[AMCMemoryBlock] = []
        quarantine_ids: list[str] = []
        revocation_flags: list[str] = []
        trust_counts: dict[str, int] = defaultdict(int)
        unverified_count = 0

        for block in store:
            # Quarantine exclusion
            if cfg.quarantine_excludes_retrieval and block.quarantine_state:
                quarantine_ids.append(block.block_id)
                continue

            # min_trust_level enforcement
            if (
                cfg.min_trust_level == TrustState.VERIFIED
                and block.trust_state != TrustState.VERIFIED
            ):
                continue

            # UNVERIFIED cap
            if block.trust_state == TrustState.UNVERIFIED:
                if unverified_count >= cfg.max_unverified:
                    continue
                unverified_count += 1

            # max_retrieve cap
            if len(retrieved_ids) >= max_retrieve:
                break

            # Accept
            retrieved_ids.append(block.block_id)
            retrieved_tokens.append(tuple(block.tokens))
            retrieved_blocks.append(block)
            trust_counts[block.trust_state.value] += 1

            # Revocation flagging
            if block.revocation_epoch and block.revocation_epoch > 0:
                revocation_flags.append(block.block_id)

        # Contradiction detection
        contradiction_pairs: list[tuple[str, str]] = []
        if cfg.detect_contradiction and len(retrieved_blocks) >= 2:
            if cfg.contradiction_fn is not None:
                # Use the provided semantic contradiction function
                for i in range(len(retrieved_blocks)):
                    for j in range(i + 1, len(retrieved_blocks)):
                        bi = retrieved_blocks[i]
                        bj = retrieved_blocks[j]
                        if bi.provenance and bj.provenance and bi.provenance != bj.provenance:
                            if cfg.contradiction_fn(bi.provenance, bj.provenance):
                                contradiction_pairs.append((bi.block_id, bj.block_id))
            else:
                # Fallback: token-equality + different non-empty provenance
                by_tokens: dict[tuple[int, ...], list[AMCMemoryBlock]] = defaultdict(list)
                for blk in retrieved_blocks:
                    by_tokens[blk.tokens].append(blk)
                for blk_list in by_tokens.values():
                    if len(blk_list) < 2:
                        continue
                    # Pair each combination with differing provenance
                    for i in range(len(blk_list)):
                        for j in range(i + 1, len(blk_list)):
                            pi = blk_list[i].provenance
                            pj = blk_list[j].provenance
                            if pi and pj and pi != pj:
                                contradiction_pairs.append(
                                    (blk_list[i].block_id, blk_list[j].block_id)
                                )

        trust_distribution = tuple(sorted(trust_counts.items()))

        return TrustRAGResult(
            retrieved_ids=tuple(retrieved_ids),
            trust_distribution=trust_distribution,
            contradiction_pairs=tuple(contradiction_pairs),
            quarantine_ids=tuple(quarantine_ids),
            revocation_flags=tuple(revocation_flags),
            retrieved_tokens=tuple(retrieved_tokens),
        )
