"""Sink-token cache utilities for sliding-window decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def kv_cache_nbytes(key_cache: torch.Tensor, value_cache: torch.Tensor) -> int:
    """Return total bytes held by key and value cache tensors."""
    return (
        key_cache.numel() * key_cache.element_size()
        + value_cache.numel() * value_cache.element_size()
    )


def estimate_kv_cache_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    kv_tensors: int = 2,
) -> int:
    """Estimate KV-cache bytes for a dense [B, T, H, D] layout."""
    values = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "kv_tensors": kv_tensors,
    }
    for name, value in values.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return batch_size * sequence_length * num_heads * head_dim * dtype_bytes * kv_tensors


def cache_token_capacity(max_cache_bytes: int, per_token_bytes: int) -> int:
    """Return how many whole sequence tokens fit inside a byte budget."""
    if max_cache_bytes < 0:
        raise ValueError(f"max_cache_bytes must be non-negative, got {max_cache_bytes!r}")
    if per_token_bytes <= 0:
        raise ValueError(f"per_token_bytes must be positive, got {per_token_bytes!r}")
    return max_cache_bytes // per_token_bytes


def _budgeted_window_indices(
    total_tokens: int,
    sink_tokens: int,
    window_size: int,
    memory_token_indices: list[int] | tuple[int, ...] | None,
    max_token_count: int,
) -> list[int]:
    """Shrink the recent window to fit a token budget while keeping sink/memory."""
    if max_token_count <= 0:
        return []
    sink = set(range(min(sink_tokens, total_tokens)))
    memory = {
        index
        for index in (memory_token_indices or [])
        if isinstance(index, int) and 0 <= index < total_tokens
    }
    mandatory = sink.union(memory)
    room_for_recent = max(0, max_token_count - len(mandatory))
    trailing_start = max(sink_tokens, total_tokens - window_size)
    trailing_candidates = [
        index for index in range(trailing_start, total_tokens) if index not in mandatory
    ]
    recent = trailing_candidates[-room_for_recent:] if room_for_recent else []
    return sorted(mandatory.union(recent))


def sink_window_indices(
    total_tokens: int,
    sink_tokens: int,
    window_size: int,
    memory_token_indices: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Keep fixed sink tokens, AMC-critical memory tokens, and recent tokens."""
    if total_tokens < 0 or sink_tokens < 0 or window_size < 0:
        raise ValueError("total_tokens, sink_tokens, and window_size must be non-negative")
    sink = list(range(min(sink_tokens, total_tokens)))
    trailing_start = max(sink_tokens, total_tokens - window_size)
    trailing = list(range(trailing_start, total_tokens))
    memory = [
        index
        for index in (memory_token_indices or [])
        if isinstance(index, int) and 0 <= index < total_tokens
    ]
    return sorted(set(sink).union(memory).union(trailing))


def compress_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    sink_tokens: int,
    window_size: int,
    memory_token_indices: list[int] | tuple[int, ...] | None = None,
    max_cache_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Select the sink + AMC-critical + trailing window from KV caches.

    When ``max_cache_bytes`` is provided, the recent window is reduced to fit
    the approximate byte budget while preserving sink and AMC-critical memory
    indices whenever the mandatory set itself fits.
    """
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must match")
    if key_cache.dim() != 4:
        raise ValueError("key_cache and value_cache must be 4D")
    if max_cache_bytes is not None and max_cache_bytes <= 0:
        raise ValueError("max_cache_bytes must be positive when provided")
    total_tokens = key_cache.size(1)
    indices = sink_window_indices(
        total_tokens, sink_tokens, window_size, memory_token_indices=memory_token_indices
    )

    if max_cache_bytes is not None and indices:
        per_token_bytes = kv_cache_nbytes(key_cache[:, :1], value_cache[:, :1])
        if per_token_bytes > 0:
            max_token_count = cache_token_capacity(max_cache_bytes, per_token_bytes)
            if len(indices) > max_token_count:
                indices = _budgeted_window_indices(
                    total_tokens,
                    sink_tokens,
                    window_size,
                    memory_token_indices,
                    max_token_count,
                )
    index_tensor = torch.tensor(indices, device=key_cache.device, dtype=torch.long)
    return (
        key_cache.index_select(1, index_tensor),
        value_cache.index_select(1, index_tensor),
        indices,
    )


@dataclass
class SinkCache:
    sink_tokens: int
    window_size: int
    memory_token_indices: list[int] | None = None
    max_cache_bytes: int | None = None
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None
    token_indices: list[int] = field(default_factory=list)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one timestep and compress to the sink-window view.

        ``memory_token_indices`` are absolute token positions in the original
        stream. The compressed tensors only contain a subset of that stream, so
        this class keeps ``token_indices`` in lockstep with the cached tensors
        and maps absolute memory positions back to current tensor offsets before
        each compression pass.
        """
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError("key and value must be 3D")
        next_token_index = self.token_indices[-1] + 1 if self.token_indices else 0
        if self.key_cache is None:
            self.key_cache = key.unsqueeze(1)
            self.value_cache = value.unsqueeze(1)
            self.token_indices = [next_token_index]
        else:
            self.key_cache = torch.cat([self.key_cache, key.unsqueeze(1)], dim=1)
            self.value_cache = torch.cat([self.value_cache, value.unsqueeze(1)], dim=1)
            self.token_indices.append(next_token_index)

        memory_positions = self._current_memory_positions()
        self.key_cache, self.value_cache, selected_positions = compress_kv_cache(
            self.key_cache,
            self.value_cache,
            sink_tokens=self.sink_tokens,
            window_size=self.window_size,
            memory_token_indices=memory_positions,
            max_cache_bytes=self.max_cache_bytes,
        )
        self.token_indices = [self.token_indices[index] for index in selected_positions]

    def _current_memory_positions(self) -> list[int]:
        if not self.memory_token_indices:
            return []
        memory_set = set(self.memory_token_indices)
        return [
            position
            for position, token_index in enumerate(self.token_indices)
            if token_index in memory_set
        ]

    def current_length(self) -> int:
        return 0 if self.key_cache is None else self.key_cache.size(1)
