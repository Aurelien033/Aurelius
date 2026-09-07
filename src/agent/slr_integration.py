"""SLR integration for the ReAct agent loop (T26)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.memory.amc_tier2 import AMCTier2Hook
from src.memory.amc_tier3 import AMCTier3Hook
from src.memory.sdb_runtime import SDBMemoryRuntime
from src.reasoning.stochastic_latent_recall import (
    SLRConfig,
    SLRQuery,
    SLRRecallTrialResult,
    run_slr_recall_trial,
)


@dataclass(frozen=True)
class SLRRecallContext:
    """SLR trial output prepared for prompt injection."""

    preamble: str
    trial: SLRRecallTrialResult
    recalled_entry_ids: tuple[str, ...]


def resolve_recall_context(
    *,
    recall_keys: Sequence[str],
    query_text: str,
    tier2_hook: AMCTier2Hook | None,
    tier3_hook: AMCTier3Hook | None,
) -> tuple[list[str], list[str]]:
    """Resolve SLR recall keys into human-readable context lines."""
    lines: list[str] = []
    entry_ids: list[str] = []
    seen: set[str] = set()

    for key in recall_keys:
        if tier3_hook is None:
            continue
        entry = tier3_hook._store.get(key)
        if entry is not None and key not in seen:
            lines.append(f"- [tier3:{key}] {entry.value}")
            entry_ids.append(key)
            seen.add(key)

    if tier2_hook is not None:
        recalled = tier2_hook.retrieve(query_text, limit=tier2_hook.config.max_retrieved)
        for entry in recalled:
            if entry.id in seen:
                continue
            lines.append(f"- [tier2:{entry.role}] {entry.content}")
            entry_ids.append(entry.id)
            seen.add(entry.id)

    return lines, entry_ids


def prepare_slr_recall_context(
    *,
    config: SLRConfig,
    query_text: str,
    query_id: str,
    tier2_hook: AMCTier2Hook | None = None,
    tier3_hook: AMCTier3Hook | None = None,
    sdb_runtime: SDBMemoryRuntime | None = None,
    session_id: str = "default",
    step_idx: int = 0,
) -> SLRRecallContext | None:
    """Run SLR recall trial and build injectable context (proposal-only)."""
    if not config.enabled:
        return None

    query = SLRQuery(
        query_id=query_id,
        query_text=query_text,
        memory_scope=("tier2", "tier3"),
        baseline_candidate_ids=(),
        metadata={"session_id": session_id, "step": step_idx},
    )
    trial = run_slr_recall_trial(config=config, query=query)
    selected = next(
        candidate
        for candidate in trial.candidates
        if candidate.candidate_id == trial.selection.selected_candidate_id
    )
    lines, entry_ids = resolve_recall_context(
        recall_keys=selected.recall_keys,
        query_text=query_text,
        tier2_hook=tier2_hook,
        tier3_hook=tier3_hook,
    )
    if sdb_runtime is not None:
        sdb_runtime.record_slr_selection(
            session_id=session_id,
            step=step_idx,
            trial=trial,
            recalled_entry_ids=entry_ids,
        )

    if not lines:
        preamble = (
            "[SLR stochastic recall]\n"
            f"Selected candidate: {trial.selection.selected_candidate_id}\n"
            f"Recall keys: {', '.join(selected.recall_keys)}"
        )
    else:
        preamble = "[SLR stochastic recall — selected memory context]:\n" + "\n".join(lines)

    return SLRRecallContext(
        preamble=preamble,
        trial=trial,
        recalled_entry_ids=tuple(entry_ids),
    )


__all__ = [
    "SLRRecallContext",
    "prepare_slr_recall_context",
    "resolve_recall_context",
]
