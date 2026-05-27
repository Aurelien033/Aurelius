"""C-19 regression: multiprocessing tokenize path must produce distinct tokenizations per text.

The bug at training_data/tokenize_pipeline.py:81 was:
    pool.map(self._tokenize_text, [tokenizer] * len(texts), texts)

pool.map signature is (func, iterable, chunksize). This passed the wrong arity.
Fix: use pool.starmap with [(tokenizer, t) for t in texts].
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from training_data.tokenize_pipeline import TokenizePipeline


@pytest.fixture
def mp_pipeline():
    """Pipeline with multiprocessing enabled (num_workers=2)."""
    return TokenizePipeline(
        {
            "tokenize": {
                "shard_size": 4,
                "max_length": 32,
                "num_workers": 2,
                "verify_integrity": False,
            },
        }
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestTokenizePipelineMultiprocessing:
    def test_multiprocessing_produces_distinct_tokenizations(self, mp_pipeline, temp_dir):
        """Each text must tokenize to distinct ids. If all shards are identical, the bug is present."""
        # 8 distinct texts, shard_size=4 → triggers multiprocessing path (len(texts) > shard_size)
        texts = [f"document_{i}_with_unique_content_for_testing" for i in range(8)]
        shards = mp_pipeline.tokenize_texts(texts, temp_dir, shard_size=4)
        
        assert len(shards) == 2
        
        # Load both shards
        arr0 = np.load(shards[0])
        arr1 = np.load(shards[1])
        
        # Each shard has 4 rows
        assert arr0.shape[0] == 4
        assert arr1.shape[0] == 4
        
        # Concatenate all 8 tokenized sequences
        all_rows = np.concatenate([arr0, arr1], axis=0)
        
        # Each of the 8 rows must be distinct (not all identical)
        # If the bug is present, all rows will be identical (all tokenized the same text)
        unique_rows = len(set(tuple(row) for row in all_rows))
        assert unique_rows == 8, f"Expected 8 distinct tokenizations, got {unique_rows}. Bug C-19 is present."

    def test_multiprocessing_matches_serial_output(self, temp_dir):
        """Multiprocessing path must produce identical output to serial path on same input."""
        texts = [f"test_text_{i}" for i in range(6)]
        
        # Serial path
        serial_pipeline = TokenizePipeline(
            {
                "tokenize": {
                    "shard_size": 8,
                    "max_length": 16,
                    "num_workers": 1,  # serial
                    "verify_integrity": False,
                },
            }
        )
        serial_dir = os.path.join(temp_dir, "serial")
        serial_shards = serial_pipeline.tokenize_texts(texts, serial_dir, shard_size=8)
        serial_arr = np.load(serial_shards[0])
        
        # Multiprocessing path (force it by making shard_size small)
        mp_pipeline = TokenizePipeline(
            {
                "tokenize": {
                    "shard_size": 3,
                    "max_length": 16,
                    "num_workers": 2,
                    "verify_integrity": False,
                },
            }
        )
        mp_dir = os.path.join(temp_dir, "mp")
        mp_shards = mp_pipeline.tokenize_texts(texts, mp_dir, shard_size=3)
        
        # Concatenate mp shards
        mp_arrs = [np.load(s) for s in mp_shards]
        mp_arr = np.concatenate(mp_arrs, axis=0)
        
        # Both must have same shape and content
        assert serial_arr.shape == mp_arr.shape
        np.testing.assert_array_equal(serial_arr, mp_arr)
