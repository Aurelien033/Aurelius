"""Tests for sink-token cache utilities."""

import pytest
import torch

from src.inference.sink_cache import (
    SinkCache,
    cache_token_capacity,
    compress_kv_cache,
    estimate_kv_cache_bytes,
    kv_cache_nbytes,
    sink_window_indices,
)


def test_sink_window_indices_keep_prefix_and_tail():
    indices = sink_window_indices(total_tokens=10, sink_tokens=2, window_size=3)
    assert indices == [0, 1, 7, 8, 9]


def test_sink_window_indices_keep_memory_critical_tokens():
    indices = sink_window_indices(
        total_tokens=12,
        sink_tokens=2,
        window_size=3,
        memory_token_indices=[4, 8, 100, -1],
    )
    assert indices == [0, 1, 4, 8, 9, 10, 11]


def test_sink_window_indices_handles_short_sequence():
    indices = sink_window_indices(total_tokens=3, sink_tokens=4, window_size=2)
    assert indices == [0, 1, 2]


def test_compress_kv_cache_reduces_sequence_length():
    keys = torch.randn(1, 10, 2, 4)
    values = torch.randn(1, 10, 2, 4)
    compressed_k, compressed_v, indices = compress_kv_cache(
        keys, values, sink_tokens=2, window_size=3
    )
    assert compressed_k.shape[1] == len(indices)
    assert compressed_v.shape[1] == len(indices)


def test_kv_cache_nbytes_counts_key_and_value_tensors():
    keys = torch.zeros(1, 4, 2, 4, dtype=torch.float32)
    values = torch.zeros(1, 4, 2, 4, dtype=torch.float32)
    assert kv_cache_nbytes(keys, values) == keys.numel() * 4 + values.numel() * 4


def test_estimate_kv_cache_bytes_matches_tensor_layout():
    assert (
        estimate_kv_cache_bytes(
            batch_size=1,
            sequence_length=4,
            num_heads=2,
            head_dim=4,
            dtype_bytes=4,
        )
        == 256
    )


def test_cache_token_capacity_counts_whole_tokens():
    per_token = estimate_kv_cache_bytes(
        batch_size=1,
        sequence_length=1,
        num_heads=2,
        head_dim=4,
        dtype_bytes=4,
    )
    assert cache_token_capacity(per_token * 7 + per_token // 2, per_token) == 7


def test_compress_kv_cache_enforces_byte_budget_while_preserving_memory_tokens():
    keys = torch.randn(1, 10, 2, 4)
    values = torch.randn(1, 10, 2, 4)
    per_token_bytes = kv_cache_nbytes(keys[:, :1], values[:, :1])

    compressed_k, compressed_v, indices = compress_kv_cache(
        keys,
        values,
        sink_tokens=1,
        window_size=8,
        memory_token_indices=[5],
        max_cache_bytes=per_token_bytes * 4,
    )

    assert 0 in indices
    assert 5 in indices
    assert len(indices) <= 4
    assert kv_cache_nbytes(compressed_k, compressed_v) <= per_token_bytes * 4


def test_sink_cache_append_tracks_current_length():
    cache = SinkCache(sink_tokens=2, window_size=3)
    for _ in range(6):
        cache.append(torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    assert cache.current_length() == 5


def test_sink_cache_append_respects_byte_budget():
    first_key = torch.randn(1, 2, 4)
    first_value = torch.randn(1, 2, 4)
    per_token_bytes = kv_cache_nbytes(first_key.unsqueeze(1), first_value.unsqueeze(1))
    cache = SinkCache(sink_tokens=1, window_size=8, max_cache_bytes=per_token_bytes * 3)
    cache.append(first_key, first_value)
    for _ in range(5):
        cache.append(torch.randn(1, 2, 4), torch.randn(1, 2, 4))

    assert cache.key_cache is not None
    assert cache.value_cache is not None
    assert cache.current_length() <= 3
    assert kv_cache_nbytes(cache.key_cache, cache.value_cache) <= per_token_bytes * 3


def test_sink_cache_preserves_absolute_memory_tokens_after_compression():
    first_key = torch.zeros(1, 1, 1)
    first_value = torch.zeros(1, 1, 1)
    per_token_bytes = kv_cache_nbytes(first_key.unsqueeze(1), first_value.unsqueeze(1))
    cache = SinkCache(
        sink_tokens=1,
        window_size=8,
        memory_token_indices=[5],
        max_cache_bytes=per_token_bytes * 4,
    )

    for token_index in range(10):
        token = torch.ones(1, 1, 1) * float(token_index)
        cache.append(token, token)

    assert cache.key_cache is not None
    retained_values = [int(value) for value in cache.key_cache[0, :, 0, 0].tolist()]
    assert 0 in retained_values
    assert 5 in retained_values
    assert cache.token_indices == retained_values
    assert len(retained_values) <= 4


def test_sink_cache_preserves_sink_positions():
    cache = SinkCache(sink_tokens=2, window_size=2)
    keys = [torch.full((1, 1, 1), float(index)) for index in range(5)]
    for key in keys:
        cache.append(key, key)
    assert torch.allclose(cache.key_cache[:, 0], torch.tensor([[[0.0]]]))
    assert torch.allclose(cache.key_cache[:, 1], torch.tensor([[[1.0]]]))


def test_compress_kv_cache_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compress_kv_cache(torch.randn(1, 2, 3), torch.randn(1, 2, 3), sink_tokens=1, window_size=1)


def test_sink_window_indices_reject_negative_args():
    with pytest.raises(ValueError):
        sink_window_indices(-1, 1, 1)
