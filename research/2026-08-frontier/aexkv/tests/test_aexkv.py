#!/usr/bin/env python3
"""Exhaustive-style correctness tests for the AEX-KV research prototype."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import aexkv


class AEXKVTests(unittest.TestCase):
    def assert_words_equal(self, a: np.ndarray, b: np.ndarray) -> None:
        av = a.view(np.uint16) if a.dtype == np.float16 else a.astype(np.uint16, copy=False)
        bv = b.view(np.uint16) if b.dtype == np.float16 else b.astype(np.uint16, copy=False)
        self.assertTrue(np.array_equal(av, bv))

    def test_float_flip_bijection_all_patterns(self) -> None:
        x = np.arange(65536, dtype=np.uint16)
        y = aexkv._float_flip(x)
        z = aexkv._float_unflip(y)
        self.assertTrue(np.array_equal(x, z))
        self.assertEqual(len(np.unique(y)), 65536)

    def test_special_fp16_patterns(self) -> None:
        # +0, -0, min/max subnormal, min normal, +/-1, max finite,
        # +/-inf, qNaN/sNaN-like payloads, and arbitrary payload patterns.
        patterns = np.array(
            [
                0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00,
                0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7D01, 0xFE55,
                0x3555, 0xAAAA,
            ],
            dtype=np.uint16,
        ).reshape(1, -1, 1)
        blob, stats = aexkv.encode(patterns, block_tokens=7, block_channels=1)
        out, dtype_code = aexkv.decode_bits(blob)
        self.assertEqual(dtype_code, aexkv.DTYPE_FP16)
        self.assertTrue(np.array_equal(patterns, out))
        self.assertTrue(aexkv.verify_exact(patterns, blob))
        self.assertEqual(stats.raw_bytes, patterns.size * 2)

    def test_every_transform_roundtrip_random_shapes(self) -> None:
        rng = np.random.default_rng(20260801)
        for shape in [(1, 1), (7, 1), (9, 3), (64, 16), (17, 5)]:
            block = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            for mode in aexkv.Mode:
                transformed = aexkv._transform(block, mode)
                if transformed is None:
                    continue
                restored = aexkv._inverse(transformed, mode, shape)
                self.assertTrue(
                    np.array_equal(block, restored),
                    msg=f"mode={mode.name} shape={shape}",
                )

    def test_all_uint16_patterns_chunked(self) -> None:
        # Covers every possible FP16/BF16 bit pattern in one tensor.
        x = np.arange(65536, dtype=np.uint16).reshape(1, 1024, 64)
        blob, _ = aexkv.encode(x, block_tokens=128, block_channels=8, zstd_level=1)
        out, _ = aexkv.decode_bits(blob)
        self.assertTrue(np.array_equal(x, out))

    def test_random_adaptive_roundtrips(self) -> None:
        rng = np.random.default_rng(7)
        for shape, bt, bc in [
            ((1, 3, 2), 2, 1),
            ((2, 17, 5), 8, 3),
            ((3, 64, 16), 16, 4),
            ((2, 257, 7), 64, 1),
        ]:
            x = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            for bh in sorted(set((1, max(1, shape[0] // 2), shape[0]))):
                blob, stats = aexkv.encode(
                    x,
                    block_heads=bh,
                    block_tokens=bt,
                    block_channels=bc,
                    zstd_level=3,
                )
                out, _ = aexkv.decode_bits(blob)
                self.assertTrue(np.array_equal(x, out), msg=f"shape={shape}, bh={bh}")
                # Raw fallback guarantees payload <= raw; framing is the only expansion.
                max_size = x.nbytes + aexkv.HEADER.size + stats.nblocks * aexkv.ENTRY.size
                self.assertLessEqual(len(blob), max_size)

    def test_highly_structured_modes_are_selected(self) -> None:
        # Smooth numeric channels, exact repeats, and exponent-local values.
        base = np.arange(512, dtype=np.uint16)
        ordered = np.stack([(base + 17 * c) & 0xFFFF for c in range(8)], axis=1)
        words = aexkv._float_unflip(ordered).reshape(1, 512, 8)
        blob, stats = aexkv.encode(words, block_tokens=256, block_channels=8, zstd_level=3)
        self.assertTrue(aexkv.verify_exact(words, blob))
        self.assertGreater(stats.ratio, 1.0)
        self.assertTrue(any(name != "RAW" for name in stats.mode_counts))

    def test_random_access(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.integers(0, 65536, size=(2, 33, 7), dtype=np.uint16)
        blob, stats = aexkv.encode(x, block_heads=2, block_tokens=8, block_channels=3)
        header, _ = aexkv.parse_header(blob)
        coords = aexkv._block_coords(header)
        self.assertEqual(len(coords), stats.nblocks)
        for i, coord in enumerate(coords):
            decoded_coord, block = aexkv.decode_block(blob, i)
            self.assertEqual(decoded_coord, coord)
            h0, nh, t0, nt, c0, nc = coord
            self.assertTrue(
                np.array_equal(
                    block,
                    x[h0 : h0 + nh, t0 : t0 + nt, c0 : c0 + nc],
                )
            )

    def test_corruption_is_detected(self) -> None:
        x = np.zeros((1, 64, 4), dtype=np.uint16)
        blob, _ = aexkv.encode(x, block_tokens=32, block_channels=2)
        header, entries = aexkv.parse_header(blob)
        bad = bytearray(blob)
        bad[entries[0].offset] ^= 0x01
        with self.assertRaises(Exception):
            aexkv.decode_bits(bytes(bad))

    def test_real_capture_subset(self) -> None:
        cap = os.path.join(
            os.path.dirname(ROOT),
            "neuro_frontier_2026-08-01",
            "kv_capture_L2_L12_L22.npy",
        )
        if not os.path.exists(cap):
            self.skipTest("real KV capture unavailable")
        data = np.load(cap, allow_pickle=True).item()
        for kind in ("keys", "vals"):
            x = data[kind]["L22"][:2, :257, :17]
            blob, _ = aexkv.encode(x, block_tokens=64, block_channels=8)
            out = aexkv.decode(blob)
            self.assert_words_equal(x, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
