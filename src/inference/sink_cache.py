"""Sink-token cache utilities for sliding-window decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def kv_cache_nbytes(key_cache: torch.Tensor, value_cache: torch.Tensor) -> int:
    """Return total bytes held by key and value cache tensors."""
    return (
        key_cache.numel() * key_cache.element_size()
        + value_cache.numel() * value_cache.element_size()
    )


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
            max_token_count = max_cache_bytes // per_token_bytes
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

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one timestep and compress to the sink-window view."""
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError("key and value must be 3D")
        if self.key_cache is None:
            self.key_cache = key.unsqueeze(1)
            self.value_cache = value.unsqueeze(1)
        else:
            self.key_cache = torch.cat([self.key_cache, key.unsqueeze(1)], dim=1)
            self.value_cache = torch.cat([self.value_cache, value.unsqueeze(1)], dim=1)
        self.key_cache, self.value_cache, _ = compress_kv_cache(
            self.key_cache,
            self.value_cache,
            sink_tokens=self.sink_tokens,
            window_size=self.window_size,
            memory_token_indices=self.memory_token_indices,
            max_cache_bytes=self.max_cache_bytes,
        )

    def current_length(self) -> int:
        return 0 if self.key_cache is None else self.key_cache.size(1)
