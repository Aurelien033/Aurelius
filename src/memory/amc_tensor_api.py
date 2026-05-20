"""AMC Tensor API — typed contracts for model-layer memory, tier operations, and benchmarks.

This module defines the stable interface that future model blocks (SSM, Mamba, MLA)
and benchmark evaluation loops can implement and depend on.  It is deliberately
free of heavyweight framework dependencies — ``torch`` is imported lazily inside
the concrete helpers so the protocol types can be imported in library code even
when torch is not installed.

Tier hierarchy
--------------
Tier-1  Per-layer working memory — :class:`AMCReadResult` / :class:`AMCWriteDecision`
Tier-2  Episodic / segment recall  — :class:`AMCMemoryRead`
Tier-3  Durable store + quarantine   — :class:`AMCMemoryConsolidate`

Benchmark scaffold
------------------
:class:`AMCMemoryModes` enumerates the five canonical ablation modes that the
benchmark runner can evaluate.  :class:`AMCBenchmarkConfig` supplies the
hyper-parameters for a single benchmark run, and :func:`score_ablation` +
:func:`build_benchmark_result` turn raw ``(mode, results)`` pairs into a
JSON-serialisable payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
import time


# ── Enumerations ─────────────────────────────────────────────────────────────

class MemoryTier(str, Enum):
    """AMC memory tier identifier."""

    TIER_1 = "tier_1"  # per-layer working memory (LM hidden states, KV latent)
    TIER_2 = "tier_2"  # episodic / segment recall
    TIER_3 = "tier_3"  # durable trusted/quarantined store


class AdmissionAction(str, Enum):
    """Outcome of an admission check (mirrors src.safety.AdmissionAction)."""

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"
    QUARANTINE = "quarantine"


# ── Frozen dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class AMCTensorState:
    """Immutable per-layer working-memory tensor snapshot for one transformer step.

    Shape conventions
    -----------------
    * ``batch``      B  — batch dimension (1 for single-sequence)
    * ``seq_len``    T  — new tokens in this step (1 for decode, >1 for prefill)
    * ``kv_lrank``   K  — compressed KV latent rank (``kv_lrank`` in config)
    * ``kvs`` is ``(compressed_k, compressed_v)``, each shaped
      ``(B, token_count, K)`` where ``token_count = past_len + T``.
    * ``rms_norm_stats`` is an optional per-head RMS sequence; when present it
      is shaped ``(B, token_count, n_rms_stats)`` or equivalent.

    Attributes
    ----------
    layer_index:
        Zero-based position of this layer inside the transformer stack.
    token_count:
        Absolute number of tokens the layer has seen so far
        (``past_len + seq_len``).
    kvs:
        Tuple ``(k_compressed, v_compressed)`` from this layer's attention.
    rms_norm_stats:
        Optional accumulated RMS / norm statistics for residual tracking.
    dtype:
        Floating-point type of the tensors, for compatibility checks.
    device_index:
        CUDA / MPS device ordinal (``None`` for CPU).
    metadata:
        Free-form key-value bag (e.g. ``{"step": 42, "role": "decoder"}``).
    """

    layer_index: int
    token_count: int
    kvs: tuple[Any, ...]  # (compressed_k, compressed_v) — Any so torch-free consumers can wire stubs
    rms_norm_stats: tuple[Any, ...] | None = None
    dtype: Any | None = None
    device_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError(f"layer_index must be >= 0, got {self.layer_index}")
        if self.token_count < 0:
            raise ValueError(f"token_count must be >= 0, got {self.token_count}")
        if not isinstance(self.kvs, tuple):
            raise TypeError("kvs must be a tuple (compressed_k, compressed_v)")


# ── Tier-1 results ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AMCWriteDecision:
    """Result of attempting to write an observation at the current layer.

    Attributes
    ----------
    admitted:
        ``True`` when the observation was stored.
    action:
        The :class:`AdmissionAction` that was taken.
    reason:
        Human-readable explanation (for logging and debugging).
    surprise_score:
        Raw surprise/novelty score (0.0–1.0) supplied by the caller.
    surprise_adjusted:
        Surprise after admission gating (may differ from ``surprise_score``
        if the content was sanitised or rejected).
    tier2_write_promoted:
        ``True`` if this write triggered a Tier-2 incidental promotion.
    """

    admitted: bool
    action: AdmissionAction = AdmissionAction.ALLOW
    reason: str = ""
    surprise_score: float = 0.0
    surprise_adjusted: float = 0.0
    tier2_write_promoted: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.surprise_score <= 1.0:
            raise ValueError(f"surprise_score must be in [0,1]: {self.surprise_score}")
        if not 0.0 <= self.surprise_adjusted <= 1.0:
            raise ValueError(f"surprise_adjusted must be in [0,1]: {self.surprise_adjusted}")


@dataclass(frozen=True)
class AMCReadResult:
    """Contents returned from a per-layer read at the current token position.

    Attributes
    ----------
    token_count:
        Total absolute tokens when this read was made (for position tracking).
    state:
        The most recent :class:`AMCTensorState` for this layer.
    scores:
        Optional per-step scores accumulated during this read window.
    metadata:
        Free-form context (e.g. ``{"source": "tier1_cache"}``).
    """

    token_count: int
    state: AMCTensorState
    scores: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Tier-2 / Tier-3 operation shapes ─────────────────────────────────────────

@dataclass(frozen=True)
class AMCMemoryRead:
    """Requested memory read at a specific step.

    Attributes
    ----------
    query:
        Natural-language or embedding query string.
    tier:
        Which tier to read from (TIER_2 = episodic, TIER_3 = durable).
    limit:
        Maximum entries to return.
    layer_index:
        Optional originating layer index (for per-layer routing in multi-layer
        retrievals).  ``None`` means "any layer".
    step:
        Monotonic step counter; used by some backends for time-windowed recall.
        ``0`` or ``None`` means no step constraint.
    """

    query: str
    tier: MemoryTier = MemoryTier.TIER_2
    limit: int = 5
    layer_index: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


@dataclass(frozen=True)
class AMCMemoryWrite:
    """Requested memory write at a specific step.

    Attributes
    ----------
    content:
        The text or embedding to store.
    tier:
        Target tier (TIER_2 = episodic, TIER_3 = durable via promotion).
    layer_index:
        Originating layer index.
    step:
        Monotonic step counter.
    surprise:
        Pre-computed surprise score (0.0–1.0).  Unset → 0.0.
    importance:
        Optional importance weight (0.0–1.0).  Unset → 1.0.
    """

    content: str
    tier: MemoryTier = MemoryTier.TIER_2
    layer_index: int | None = None
    step: int | None = None
    surprise: float = 0.0
    importance: float | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("content must be non-empty")
        if not 0.0 <= self.surprise <= 1.0:
            raise ValueError(f"surprise must be in [0,1]: {self.surprise}")


@dataclass(frozen=True)
class AMCMemoryConsolidate:
    """Request a Tier-3 consolidation pass.

    Attributes
    ----------
    tiers:
        Which tiers to touch.  Default (``(TIER_3,)``) consolidates only
        long-term store; add ``TIER_2`` for a combined rehearsal pass.
    max_entries:
        Hard cap on the number of long-term entries retained after
        consolidation.  ``None`` uses the hook's configured ``max_entries``.
    """

    tiers: tuple[MemoryTier, ...] = (MemoryTier.TIER_3,)
    max_entries: int | None = None


# ── Protocols ────────────────────────────────────────────────────────────────

@runtime_checkable
class AMCLayerMemory(Protocol):
    """Protocol for per-layer (Tier-1) working memory.

    Each transformer layer in the model stack holds one instance of this
    protocol so that read/write/reset are scoped to a single layer.

    Shape conventions
    -----------------
    All ``AMCTensorState`` objects flowing through this protocol carry shapes
    documented in that class: ``kvs = (B, token_count, kv_lrank)``.

    Implementations may store state as a Python object, a tensor buffer, or
    any other internal representation — callers only see ``AMCTensorState``.
    """

    # ── Tier-1 operations ───────────────────────────────────────────────────

    def read(
        self,
        layer_index: int,
        step: int,
        *,
        seq_len: int,
    ) -> AMCReadResult:
        """Return the accumulated working-memory state for *layer_index*.

        Parameters
        ----------
        layer_index:
            Zero-based layer position in the transformer stack.
        step:
            Monotonic generation step counter.  ``0`` at start of session.
        seq_len:
            New-token count in this forward pass (1 = decode, >1 = prefill).

        Returns
        -------
        AMCReadResult
            Containing at minimum the current :class:`AMCTensorState`.
        """
        ...  # pragma: no cover

    def write(
        self,
        state: AMCTensorState,
        *,
        step: int,
        surprise: float = 0.0,
    ) -> AMCWriteDecision:
        """Store *state* at *layer_index* and return the admission decision.

        Parameters
        ----------
        state:
            Per-layer tensor state returned from the layer's forward pass.
        step:
            Monotonic generation step counter.
        surprise:
            Signal for the tier-2 memory gate. 写入 surprise > threshold
            triggers an ``observe()`` call to the episodic store.

        Returns
        -------
        AMCWriteDecision
            Whether the state was admitted and why.
        """
        ...  # pragma: no cover

    def write_observation(
        self,
        layer_index: int,
        step: int,
        content: str,
        *,
        surprise: float,
        importance: float | None = None,
    ) -> AMCWriteDecision:
        """Write a text observation at *layer_index* to Tier-2 episodic memory.

        This is the higher-level write path used by agent/ReAct loops; it
        routes through the tier-2 surprise gate without requiring the caller
        to construct an ``AMCTensorState``.

        Parameters
        ----------
        content:
            Text or text-embedding observation to persist.
        surprise:
            Surprise/novelty score.  Below ``surprise_threshold`` the write
            is silently skipped.
        importance:
            Override importance weight (defaults to the hook's configured ``default_importance``).

        Returns
        -------
        AMCWriteDecision
            Whether the observation was stored.
        """
        ...  # pragma: no cover

    def read_memory(
        self,
        query: str,
        *,
        tier: MemoryTier = MemoryTier.TIER_2,
        limit: int = 5,
        layer_index: int | None = None,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from *tier* for *layer_index* at *step*.

        Returns a list of entry dicts, each with at minimum ``content``,
        ``source``, and ``trust_level`` keys.

        Parameters
        ----------
        query:
            Natural-language or embedding query for semantic search.
        tier:
            ``TIER_2`` (episodic) or ``TIER_3`` (durable).
        limit:
            Maximum entries to return.
        layer_index:
            Restrict reads to this layer's entries.  ``None`` = all layers.
        step:
            Restrict to entries written at or before this step.  ``None`` = no constraint.
        """
        ...  # pragma: no cover

    def consolidate(
        self,
        *,
        tiers: tuple[MemoryTier, ...] = (MemoryTier.TIER_3,),
        max_entries: int | None = None,
    ) -> dict[str, Any]:
        """Run a consolidation / eviction sweep on the requested tiers.

        Returns a serialisable dict with at minimum ``promoted``, ``quarantined``,
        ``expired_pruned``, and ``errors`` keys.

        Parameters
        ----------
        tiers:
            Which tiers to touch in this pass.
        max_entries:
            Hard cap for Tier-3 store after consolidation.  ``None`` keeps
            the hook's configured limit.
        """
        ...  # pragma: no cover

    def reset(self, layer_index: int | None = None) -> None:
        """Reset working-memory state for *layer_index*, or all layers if ``None``.

        After reset the layer's tensor state is cleared but Tier-2 / Tier-3
        entries are preserved.
        """
        ...  # pragma: no cover

    def stats(self) -> dict[str, Any]:
        """Return a small operational-statistics snapshot for this layer.

        Must include at minimum: ``layer_index``, ``episodic_entries``,
        ``surprise_threshold``, ``observed_events``, ``stored_events``.
        """
        ...  # pragma: no cover


@runtime_checkable
class AMCMemoryController(Protocol):
    """Protocol for the cross-tier memory controller (Tier-2 + Tier-3).

    Implementations compose one :class:`AMCLayerMemory` per transformer layer
    and expose a unified interface for read/write/promote/consolidate without
    leaking per-layer indexing details to the caller.

    A minimal concrete implementation is :class:`src.memory.amc_tier2.AMCTier2Hook`
    for Tier-2-only experiments, or a composite that wires :class:`AMCLayerMemory`
    instances into :class:`src.memory.amc_tier3.AMCTier3Hook`.
    """

    # ── Tier-2 operations ───────────────────────────────────────────────────

    def observe(
        self,
        role: str,
        content: str,
        *,
        surprise: float,
        importance: float | None = None,
    ) -> Any | None:
        """Store one observation to episodic memory if it crosses the surprise gate.

        Parameters
        ----------
        role:
            Message role (e.g. ``"user"``, ``"assistant"``, ``"tool"``).
        content:
            Text content to store.
        surprise:
            Surprise/novelty score (0.0–1.0).
        importance:
            Optional importance weight.

        Returns
        -------
        MemoryEntry | None
            The stored entry, or ``None`` if the surprise gate was not crossed.
        """
        ...  # pragma: no cover

    def retrieve(self, query: str, *, limit: int | None = None) -> list[Any]:
        """Retrieve matching Tier-2 entries; falls back to recent entries.

        Parameters
        ----------
        query:
            Semantic search query string.
        limit:
            Max entries to return.  ``None`` uses the hook's ``max_retrieved``.

        Returns
        -------
        list[MemoryEntry]
            Sorted most-relevant first.
        """
        ...  # pragma: no cover

    # ── Tier-3 promotion / quarantine ────────────────────────────────────────

    def promote(
        self,
        *,
        key: str,
        value: Any,
        source_tier2_id: str | None = None,
        confidence: float = 1.0,
        trust_level: Any | None = None,
        tags: frozenset[str] | None = None,
    ) -> Any | None:
        """Promote an observation to the Tier-3 durable store.

        Low-confidence observations are routed to :meth:`quarantine` automatically.

        Parameters
        ----------
        key, value, confidence, trust_level, tags, source_tier2_id:
            Forwarded to the underlying Tier-3 hook.
        """
        ...  # pragma: no cover

    def quarantine(
        self,
        *,
        key: str,
        value: Any,
        source_tier2_id: str | None = None,
        confidence: float = 0.0,
        tags: frozenset[str] | None = None,
    ) -> Any:
        """Place an observation in the Tier-3 quarantine store."""
        ...  # pragma: no cover

    def consolidate(self) -> Any:
        """Run a Tier-3 consolidation / expiry sweep.

        Returns a :class:`Tier3ConsolidationResult`-compatible object or dict.
        """
        ...  # pragma: no cover

    def prioritize(self, limit: int = 5) -> list[Any]:
        """Return active (trusted + unverified) Tier-3 entries sorted by decayed importance.

        Quarantined and revoked entries are always excluded.
        """
        ...  # pragma: no cover

    def verify_and_promote(self, key: str, confidence: float = 1.0) -> Any | None:
        """Manually verify a quarantined Tier-3 entry and promote it."""
        ...  # pragma: no cover

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return serialisable operational statistics for this memory controller."""
        ...  # pragma: no cover


# ── Benchmark scaffold ────────────────────────────────────────────────────────

class AMCMemoryModes(str, Enum):
    """The five canonical ablation modes that the benchmark runner evaluates.

    The benchmark evaluator uses these as keys in its results dict and
    score table.
    """

    NO_MEMORY = "no_memory"           # zero-shot, no retrieved context
    TIER2_CONTEXT = "tier2_context"   # Tier-2 episodic context injected into prompt
    TIER1_ONLY = "tier1_only"         # Tier-1 per-layer working memory; no prompt context
    TIER1_TIER2 = "tier1_tier2"       # Tier-1 + Tier-2 combined
    FULL = "full"                     # Tier-1 + Tier-2 + Tier-3 (full AMC pipeline)


@dataclass(frozen=True)
class AMCBenchmarkConfig:
    """Hyper-parameters for one AMC-Memory benchmark run.

    Attributes
    ----------
    mode:
        Which ablation configuration to evaluate.
    context_tokens:
        Approximate context length in model tokens.
    samples_per:
        Number of deterministic samples for each task.
    tasks:
        Subset of benchmark tasks to run.  ``None`` = all 6 canonical tasks.
    tier1_layers:
        Number of transformer layers when Tier-1 is in play.  Used by
        backends that allocate per-layer state.
    tier2_max_retrieved:
        ``max_retrieved`` passed to :class:`src.memory.amc_tier2.AMCTier2Config`
        for prompt-context injection benchmarks.
    tier2_surprise_threshold:
        ``surprise_threshold`` passed to ``AMCTier2Config``.
    tier3_min_confidence:
        ``min_confidence`` passed to :class:`src.memory.amc_tier3.AMCTier3Config`
        for consolidation-benchmark runs.
    """

    mode: AMCMemoryModes = AMCMemoryModes.NO_MEMORY
    context_tokens: int = 512
    samples_per: int = 3
    tasks: tuple[str, ...] | None = None
    tier1_layers: int = 6
    tier2_max_retrieved: int = 5
    tier2_surprise_threshold: float = 0.5
    tier3_min_confidence: float = 0.6

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError(f"context_tokens must be > 0: {self.context_tokens}")
        if self.samples_per <= 0:
            raise ValueError(f"samples_per must be > 0: {self.samples_per}")
        if not 0.0 <= self.tier2_surprise_threshold <= 1.0:
            raise ValueError(f"tier2_surprise_threshold must be in [0,1]")
        if not 0.0 <= self.tier3_min_confidence <= 1.0:
            raise ValueError(f"tier3_min_confidence must be in [0,1]")


@dataclass(frozen=True)
class AMCBenchmarkResult:
    """Aggregated score for one benchmark configuration.

    Attributes
    ----------
    mode:
        Which ablation was evaluated.
    context_tokens:
        Context length used.
    samples_per:
        Samples per task.
    overall_score:
        Mean per-task pass rate (0.0–1.0).
    per_task_scores:
        ``{task_name: pass_rate}`` mapping.
    results:
        Raw per-cell results from the evaluator.
    elapsed_seconds:
        Wall-clock time for this run.
    metadata:
        Free-form run information (e.g. model hash, git SHA).
    """

    mode: AMCMemoryModes
    context_tokens: int
    samples_per: int
    overall_score: float
    per_task_scores: dict[str, float]
    results: dict[str, Any]
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "mode": self.mode.value,
            "context_tokens": self.context_tokens,
            "samples_per": self.samples_per,
            "overall_score": self.overall_score,
            "per_task_scores": dict(self.per_task_scores),
            "results": self.results,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": dict(self.metadata),
        }


def score_ablation(
    results: dict[str, dict[str, Any]],
    *,
    mode: AMCMemoryModes,
    context_tokens: int,
    samples_per: int,
    metadata: dict[str, Any] | None = None,
) -> AMCBenchmarkResult:
    """Turn raw evaluator *results* into a ranked :class:`AMCBenchmarkResult`.

    Parameters
    ----------
    results:
        Raw per-task result dict as returned by ``AMCMemoryBenchmark.evaluate()``.
    mode, context_tokens, samples_per:
        Run configuration (carried in the result for downstream reporting).
    metadata:
        Optional free-form annotations (e.g. ``{"git_sha": "abc123"}``).

    Returns
    -------
    AMCBenchmarkResult
        Scored and ready for JSON serialisation.
    """
    from src.eval.amc_memory_benchmark import AMCMemoryBenchmark  # lazy import

    per_task = AMCMemoryBenchmark.score_per_task(results)
    overall = AMCMemoryBenchmark.overall_score(results)
    return AMCBenchmarkResult(
        mode=mode,
        context_tokens=context_tokens,
        samples_per=samples_per,
        overall_score=overall,
        per_task_scores=per_task,
        results=results,
        elapsed_seconds=0.0,  # caller fills in elapsed separately
        metadata=metadata or {},
    )


def build_benchmark_result(
    *,
    mode: AMCMemoryModes,
    context_tokens: int,
    samples_per: int,
    scores: dict[str, float],
    overall_score: float,
    elapsed_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> AMCBenchmarkResult:
    """Lightweight alternative to :func:`score_ablation` when raw *scores* are already known.

    Useful when scoring is computed outside the evaluator (e.g. engine benchmark).
    """
    return AMCBenchmarkResult(
        mode=mode,
        context_tokens=context_tokens,
        samples_per=samples_per,
        overall_score=overall_score,
        per_task_scores=scores,
        results={},
        elapsed_seconds=elapsed_seconds,
        metadata=metadata or {},
    )


# ── Public API ───────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "MemoryTier",
    "AdmissionAction",
    # Frozen dataclasses
    "AMCTensorState",
    "AMCWriteDecision",
    "AMCReadResult",
    "AMCMemoryRead",
    "AMCMemoryWrite",
    "AMCMemoryConsolidate",
    # Protocols
    "AMCLayerMemory",
    "AMCMemoryController",
    # Benchmark
    "AMCMemoryModes",
    "AMCBenchmarkConfig",
    "AMCBenchmarkResult",
    "score_ablation",
    "build_benchmark_result",
]
