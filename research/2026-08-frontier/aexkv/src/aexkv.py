"""AEX-KV: adaptive exact compression for 16-bit KV-cache tensors.

Research prototype.  All modes preserve the original 16-bit patterns exactly,
including signed zero, subnormals, infinities, and NaN payloads.  The codec is
agnostic to whether the words encode IEEE FP16 or BF16; dtype_code records the
intended interpretation.

The on-disk format contains an O(1) block directory.  Each block chooses the
smallest *actual* payload among reversible transforms and a raw fallback.
Directory/header/padding bytes are included in every reported size.
"""
from __future__ import annotations

import binascii
import io
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np
import zstandard as zstd

MAGIC = b"AEXKV002"
VERSION = 2
DTYPE_FP16 = 1
DTYPE_BF16 = 2

# magic, version, dtype, ndim, flags, H, S, D, block_heads,
# block_tokens, block_channels, nblocks, payload_start
HEADER = struct.Struct("<8sBBBBIIIHHHIQ")
# absolute payload offset, payload bytes, mode, flags, raw bytes, CRC32(raw)
ENTRY = struct.Struct("<QIBBII")


class Mode(IntEnum):
    RAW = 0
    ZSTD_RAW = 1
    ZSTD_CHANNEL = 2
    ZSTD_BYTE_SHUFFLE = 3
    ZSTD_CHANNEL_BYTE_SHUFFLE = 4
    ZSTD_XOR_TOKEN = 5
    ZSTD_DELTA_TOKEN = 6
    ZSTD_ORDERED_DELTA_TOKEN = 7
    ZSTD_ORDERED_DELTA2_TOKEN = 8
    ZSTD_DELTA_2D = 9
    ZSTD_XOR_2D = 10
    ZSTD_BITPLANE = 11
    ZSTD_FIELDS = 12
    DICTIONARY = 13
    ZSTD_DICTIONARY = 14
    RLE_DICTIONARY = 15
    ZSTD_RLE_DICTIONARY = 16
    ZSTD_XOR_TOKEN_BITPLANE = 17
    ZSTD_DELTA_TOKEN_BITPLANE = 18
    ZSTD_ORDERED_DELTA_BITPLANE = 19
    ZSTD_TRACE_EXPDELTA_BITPLANE = 20


DEFAULT_MODES: tuple[Mode, ...] = tuple(Mode)


@dataclass(frozen=True)
class HeaderInfo:
    dtype_code: int
    shape: tuple[int, int, int]
    block_heads: int
    block_tokens: int
    block_channels: int
    nblocks: int
    payload_start: int


@dataclass(frozen=True)
class BlockEntry:
    offset: int
    length: int
    mode: Mode
    flags: int
    raw_length: int
    crc32: int


@dataclass(frozen=True)
class EncodeStats:
    raw_bytes: int
    encoded_bytes: int
    ratio: float
    header_bytes: int
    directory_bytes: int
    payload_bytes: int
    nblocks: int
    mode_counts: dict[str, int]


def _as_u16_words(array: np.ndarray) -> np.ndarray:
    """Return a contiguous native uint16 view without numeric conversion."""
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"expected [heads, tokens, channels], got {arr.shape}")
    if arr.dtype == np.float16:
        words = arr.view(np.uint16)
    elif arr.dtype == np.uint16:
        words = arr
    elif arr.dtype == np.int16:
        words = arr.view(np.uint16)
    else:
        raise TypeError("array must have dtype float16, uint16, or int16")
    return np.ascontiguousarray(words, dtype=np.uint16)


def _raw_bytes(block: np.ndarray) -> bytes:
    return np.asarray(block, dtype="<u2").tobytes(order="C")


def _from_raw_bytes(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    if len(data) != 2 * n:
        raise ValueError(f"raw length mismatch: {len(data)} != {2*n}")
    return np.frombuffer(data, dtype="<u2").reshape(shape).astype(np.uint16, copy=False)


def _byte_shuffle(words: np.ndarray) -> bytes:
    x = np.asarray(words, dtype=np.uint16).reshape(-1)
    lo = (x & 0xFF).astype(np.uint8)
    hi = (x >> 8).astype(np.uint8)
    return lo.tobytes() + hi.tobytes()


def _byte_unshuffle(data: bytes, n: int) -> np.ndarray:
    if len(data) != 2 * n:
        raise ValueError("byte-shuffle payload length mismatch")
    b = np.frombuffer(data, dtype=np.uint8)
    return (b[:n].astype(np.uint16) | (b[n:].astype(np.uint16) << 8))


def _float_flip(words: np.ndarray) -> np.ndarray:
    """Bijective sign-magnitude -> monotone unsigned map for IEEE-like words.

    Positive patterns flip the sign bit; negative patterns complement all bits.
    This is bijective over every uint16 pattern, including all NaN payloads.
    """
    x = np.asarray(words, dtype=np.uint16)
    sign = (x & np.uint16(0x8000)) != 0
    return np.where(sign, np.bitwise_not(x), x ^ np.uint16(0x8000)).astype(np.uint16)


def _float_unflip(keys: np.ndarray) -> np.ndarray:
    y = np.asarray(keys, dtype=np.uint16)
    positive = (y & np.uint16(0x8000)) != 0
    return np.where(positive, y ^ np.uint16(0x8000), np.bitwise_not(y)).astype(np.uint16)


def _delta_token(words: np.ndarray) -> np.ndarray:
    x = np.asarray(words, dtype=np.uint16)
    r = np.empty_like(x)
    r[0] = x[0]
    if x.shape[0] > 1:
        r[1:] = ((x[1:].astype(np.uint32) - x[:-1].astype(np.uint32)) & 0xFFFF).astype(np.uint16)
    return r


def _undelta_token(residual: np.ndarray) -> np.ndarray:
    r = np.asarray(residual, dtype=np.uint16)
    # uint64 cumsum avoids intermediate overflow; reduce modulo 2^16.
    return (np.cumsum(r.astype(np.uint64), axis=0) & 0xFFFF).astype(np.uint16)


def _delta2_token(words: np.ndarray) -> np.ndarray:
    x = np.asarray(words, dtype=np.uint16)
    r = np.empty_like(x)
    if x.shape[0] == 0:
        return r
    r[0] = x[0]
    if x.shape[0] >= 2:
        d1 = ((x[1:].astype(np.uint32) - x[:-1].astype(np.uint32)) & 0xFFFF).astype(np.uint16)
        r[1] = d1[0]
        if x.shape[0] > 2:
            r[2:] = ((d1[1:].astype(np.uint32) - d1[:-1].astype(np.uint32)) & 0xFFFF).astype(np.uint16)
    return r


def _undelta2_token(residual: np.ndarray) -> np.ndarray:
    r = np.asarray(residual, dtype=np.uint16)
    if r.shape[0] == 0:
        return r.copy()
    out = np.empty_like(r)
    out[0] = r[0]
    if r.shape[0] >= 2:
        d = np.empty_like(r)
        d[0] = 0
        d[1:] = (np.cumsum(r[1:].astype(np.uint64), axis=0) & 0xFFFF).astype(np.uint16)
        for t in range(1, r.shape[0]):
            out[t] = ((out[t - 1].astype(np.uint32) + d[t].astype(np.uint32)) & 0xFFFF).astype(np.uint16)
    return out


def _predict2d(words: np.ndarray, xor: bool = False) -> np.ndarray:
    x = np.asarray(words, dtype=np.uint16)
    up = np.zeros_like(x)
    left = np.zeros_like(x)
    diag = np.zeros_like(x)
    up[1:, :] = x[:-1, :]
    left[:, 1:] = x[:, :-1]
    diag[1:, 1:] = x[:-1, :-1]
    if xor:
        return (x ^ up ^ left ^ diag).astype(np.uint16)
    pred = (up.astype(np.uint32) + left.astype(np.uint32) - diag.astype(np.uint32)) & 0xFFFF
    return ((x.astype(np.uint32) - pred) & 0xFFFF).astype(np.uint16)


def _unpredict2d(residual: np.ndarray, xor: bool = False) -> np.ndarray:
    r = np.asarray(residual, dtype=np.uint16)
    nt, nc = r.shape
    x = np.empty_like(r)
    for t in range(nt):
        for c in range(nc):
            up = int(x[t - 1, c]) if t else 0
            left = int(x[t, c - 1]) if c else 0
            diag = int(x[t - 1, c - 1]) if (t and c) else 0
            if xor:
                pred = up ^ left ^ diag
                x[t, c] = int(r[t, c]) ^ pred
            else:
                pred = (up + left - diag) & 0xFFFF
                x[t, c] = (int(r[t, c]) + pred) & 0xFFFF
    return x


def _pack_fixed(values: np.ndarray, width: int) -> bytes:
    vals = np.asarray(values, dtype=np.uint32).reshape(-1)
    if not 0 <= width <= 16:
        raise ValueError("width must be in [0,16]")
    if width == 0:
        return b""
    if vals.size and int(vals.max()) >= (1 << width):
        raise ValueError("value does not fit bit width")
    bits = ((vals[:, None] >> np.arange(width, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def _unpack_fixed(data: bytes, count: int, width: int) -> np.ndarray:
    if width == 0:
        return np.zeros(count, dtype=np.uint16)
    need = (count * width + 7) // 8
    if len(data) < need:
        raise ValueError("truncated fixed-width stream")
    bits = np.unpackbits(np.frombuffer(data[:need], dtype=np.uint8), bitorder="little")[: count * width]
    matrix = bits.reshape(count, width).astype(np.uint32)
    weights = (np.uint32(1) << np.arange(width, dtype=np.uint32))
    return (matrix @ weights).astype(np.uint16)


def _bitplane_pack(words: np.ndarray) -> bytes:
    # Channel-major flattening gives each plane long same-channel contexts.
    x = np.asarray(words, dtype=np.uint16).T.reshape(-1)
    planes = ((x[None, :] >> np.arange(16, dtype=np.uint16)[:, None]) & 1).astype(np.uint8)
    return np.packbits(planes, axis=1, bitorder="little").tobytes()


def _bitplane_unpack(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    nt, nc = shape
    n = nt * nc
    plane_bytes = (n + 7) // 8
    if len(data) != 16 * plane_bytes:
        raise ValueError("bitplane payload length mismatch")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(16, plane_bytes)
    planes = np.unpackbits(packed, axis=1, bitorder="little")[:, :n].astype(np.uint16)
    x = np.zeros(n, dtype=np.uint16)
    for bit in range(16):
        x |= planes[bit] << bit
    return x.reshape(nc, nt).T


def _bitplanes_width_pack(values: np.ndarray, width: int) -> bytes:
    x = np.asarray(values, dtype=np.uint16).reshape(-1)
    planes = ((x[None, :] >> np.arange(width, dtype=np.uint16)[:, None]) & 1).astype(np.uint8)
    return np.packbits(planes, axis=1, bitorder="little").tobytes()


def _bitplanes_width_unpack(data: bytes, count: int, width: int) -> np.ndarray:
    plane_bytes = (count + 7) // 8
    if len(data) != width * plane_bytes:
        raise ValueError("width-bitplane payload length mismatch")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(width, plane_bytes)
    planes = np.unpackbits(packed, axis=1, bitorder="little")[:, :count].astype(np.uint16)
    out = np.zeros(count, dtype=np.uint16)
    for bit in range(width):
        out |= planes[bit] << bit
    return out


def _trace_fields(dtype_code: int) -> tuple[int, int]:
    if dtype_code == DTYPE_FP16:
        return 5, 10
    if dtype_code == DTYPE_BF16:
        return 8, 7
    raise ValueError(f"unsupported dtype code {dtype_code}")


def _trace_expdelta_pack(words: np.ndarray, dtype_code: int) -> bytes:
    """TRACE-style proxy: channel grouping, base-exponent deltas, bit planes."""
    x = np.asarray(words, dtype=np.uint16)
    nt, nc = x.shape
    exp_bits, mant_bits = _trace_fields(dtype_code)
    mant_mask = (1 << mant_bits) - 1
    sign = (x >> 15).T.reshape(-1)
    exp = ((x >> mant_bits) & ((1 << exp_bits) - 1)).astype(np.uint16)
    mant = (x & mant_mask).T.reshape(-1)
    # TRACE does not disclose its beta-selection rule.  The proxy uses the
    # per-channel median and modular deltas, which is exact for every exponent
    # pattern and is more robust to a single extreme exponent than min(exp).
    base = np.median(exp, axis=0).astype(np.uint16)
    exp_mask = (1 << exp_bits) - 1
    delta = (
        (exp.astype(np.int32) - base[None, :].astype(np.int32)) & exp_mask
    ).astype(np.uint16).T.reshape(-1)
    return (
        _pack_fixed(base, exp_bits)
        + _bitplanes_width_pack(sign, 1)
        + _bitplanes_width_pack(delta, exp_bits)
        + _bitplanes_width_pack(mant, mant_bits)
    )


def _trace_expdelta_unpack(data: bytes, shape: tuple[int, int], dtype_code: int) -> np.ndarray:
    nt, nc = shape
    n = nt * nc
    exp_bits, mant_bits = _trace_fields(dtype_code)
    nb = (nc * exp_bits + 7) // 8
    ns = (n + 7) // 8
    ne = exp_bits * ((n + 7) // 8)
    nm = mant_bits * ((n + 7) // 8)
    if len(data) != nb + ns + ne + nm:
        raise ValueError("TRACE-proxy payload length mismatch")
    pos = 0
    base = _unpack_fixed(data[pos : pos + nb], nc, exp_bits)
    pos += nb
    sign = _bitplanes_width_unpack(data[pos : pos + ns], n, 1)
    pos += ns
    delta = _bitplanes_width_unpack(data[pos : pos + ne], n, exp_bits)
    pos += ne
    mant = _bitplanes_width_unpack(data[pos : pos + nm], n, mant_bits)
    exp = ((delta.reshape(nc, nt).T + base[None, :]) & ((1 << exp_bits) - 1)).astype(np.uint16)
    sign = sign.reshape(nc, nt).T
    mant = mant.reshape(nc, nt).T
    return ((sign << 15) | (exp << mant_bits) | mant).astype(np.uint16)


def _fields_pack(words: np.ndarray) -> bytes:
    x = np.asarray(words, dtype=np.uint16).T.reshape(-1)
    sign = x >> 15
    exp = (x >> 10) & 0x1F
    mant = x & 0x03FF
    return _pack_fixed(sign, 1) + _pack_fixed(exp, 5) + _pack_fixed(mant, 10)


def _fields_unpack(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    nt, nc = shape
    n = nt * nc
    ns = (n + 7) // 8
    ne = (5 * n + 7) // 8
    nm = (10 * n + 7) // 8
    if len(data) != ns + ne + nm:
        raise ValueError("field payload length mismatch")
    sign = _unpack_fixed(data[:ns], n, 1)
    exp = _unpack_fixed(data[ns : ns + ne], n, 5)
    mant = _unpack_fixed(data[ns + ne :], n, 10)
    x = ((sign << 15) | (exp << 10) | mant).astype(np.uint16)
    return x.reshape(nc, nt).T


def _encode_dictionary(words: np.ndarray, rle: bool) -> bytes | None:
    x = np.asarray(words, dtype=np.uint16)
    if x.shape[1] != 1:
        return None
    stream = x[:, 0]
    dictionary, indices = np.unique(stream, return_inverse=True)
    d = int(dictionary.size)
    if d == 0 or d > 65535:
        return None
    width = max(1, (d - 1).bit_length())
    out = bytearray(struct.pack("<HB", d, width))
    out.extend(np.asarray(dictionary, dtype="<u2").tobytes())
    if not rle:
        packed = _pack_fixed(indices.astype(np.uint16), width)
        out.extend(struct.pack("<I", len(stream)))
        out.extend(packed)
        return bytes(out)

    run_values: list[int] = []
    run_lengths: list[int] = []
    i = 0
    while i < len(indices):
        j = i + 1
        while j < len(indices) and indices[j] == indices[i]:
            j += 1
        run_values.append(int(indices[i]))
        run_lengths.append(j - i)
        i = j
    out.extend(struct.pack("<II", len(stream), len(run_values)))
    packed = _pack_fixed(np.asarray(run_values, dtype=np.uint16), width)
    out.extend(struct.pack("<I", len(packed)))
    out.extend(packed)
    for length in run_lengths:
        out.extend(_encode_varint(length))
    return bytes(out)


def _decode_dictionary(data: bytes, shape: tuple[int, int], rle: bool) -> np.ndarray:
    nt, nc = shape
    if nc != 1:
        raise ValueError("dictionary mode requires one channel")
    view = memoryview(data)
    if len(view) < 3:
        raise ValueError("truncated dictionary header")
    d, width = struct.unpack_from("<HB", view, 0)
    pos = 3
    dict_bytes = 2 * d
    if len(view) < pos + dict_bytes:
        raise ValueError("truncated dictionary")
    dictionary = np.frombuffer(view[pos : pos + dict_bytes], dtype="<u2")
    pos += dict_bytes
    if not rle:
        if len(view) < pos + 4:
            raise ValueError("truncated dictionary count")
        count = struct.unpack_from("<I", view, pos)[0]
        pos += 4
        indices = _unpack_fixed(view[pos:].tobytes(), count, width).astype(np.int64)
        if count != nt or (indices.size and int(indices.max()) >= d):
            raise ValueError("invalid dictionary indices")
        return dictionary[indices].reshape(nt, 1).astype(np.uint16)

    if len(view) < pos + 8:
        raise ValueError("truncated RLE dictionary header")
    count, nruns = struct.unpack_from("<II", view, pos)
    pos += 8
    if len(view) < pos + 4:
        raise ValueError("truncated run-index length")
    packed_len = struct.unpack_from("<I", view, pos)[0]
    pos += 4
    if len(view) < pos + packed_len:
        raise ValueError("truncated run indices")
    run_values = _unpack_fixed(view[pos : pos + packed_len].tobytes(), nruns, width).astype(np.int64)
    pos += packed_len
    if run_values.size and int(run_values.max()) >= d:
        raise ValueError("invalid run dictionary index")
    run_lengths = []
    for _ in range(nruns):
        length, pos = _decode_varint(view, pos)
        if length <= 0:
            raise ValueError("zero-length run")
        run_lengths.append(length)
    if sum(run_lengths) != count or count != nt:
        raise ValueError("RLE count mismatch")
    out = np.repeat(dictionary[run_values], np.asarray(run_lengths, dtype=np.int64))
    return out.reshape(nt, 1).astype(np.uint16)


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be nonnegative")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        out.append(b | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _decode_varint(data: memoryview, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data) or shift > 63:
            raise ValueError("invalid/truncated varint")
        b = int(data[pos])
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7


def _zcompress(data: bytes, level: int) -> bytes:
    return zstd.ZstdCompressor(level=level, write_checksum=False, write_content_size=True).compress(data)


def _zdecompress(data: bytes, max_output: int) -> bytes:
    return zstd.ZstdDecompressor().decompress(data, max_output_size=max_output)


def _transform(block: np.ndarray, mode: Mode, dtype_code: int = DTYPE_FP16) -> bytes | None:
    nt, nc = block.shape
    if mode == Mode.RAW:
        return _raw_bytes(block)
    if mode == Mode.ZSTD_RAW:
        return _raw_bytes(block)
    if mode == Mode.ZSTD_CHANNEL:
        return _raw_bytes(block.T)
    if mode == Mode.ZSTD_BYTE_SHUFFLE:
        return _byte_shuffle(block.reshape(-1))
    if mode == Mode.ZSTD_CHANNEL_BYTE_SHUFFLE:
        return _byte_shuffle(block.T.reshape(-1))
    if mode == Mode.ZSTD_XOR_TOKEN:
        r = np.empty_like(block)
        r[0] = block[0]
        if nt > 1:
            r[1:] = block[1:] ^ block[:-1]
        return _byte_shuffle(r.T.reshape(-1))
    if mode == Mode.ZSTD_DELTA_TOKEN:
        return _byte_shuffle(_delta_token(block).T.reshape(-1))
    if mode == Mode.ZSTD_ORDERED_DELTA_TOKEN:
        return _byte_shuffle(_delta_token(_float_flip(block)).T.reshape(-1))
    if mode == Mode.ZSTD_ORDERED_DELTA2_TOKEN:
        return _byte_shuffle(_delta2_token(_float_flip(block)).T.reshape(-1))
    if mode == Mode.ZSTD_DELTA_2D:
        return _byte_shuffle(_predict2d(block, xor=False).T.reshape(-1))
    if mode == Mode.ZSTD_XOR_2D:
        return _byte_shuffle(_predict2d(block, xor=True).T.reshape(-1))
    if mode == Mode.ZSTD_BITPLANE:
        return _bitplane_pack(block)
    if mode == Mode.ZSTD_FIELDS:
        return _fields_pack(block)
    if mode == Mode.ZSTD_XOR_TOKEN_BITPLANE:
        r = np.empty_like(block)
        r[0] = block[0]
        if nt > 1:
            r[1:] = block[1:] ^ block[:-1]
        return _bitplane_pack(r)
    if mode == Mode.ZSTD_DELTA_TOKEN_BITPLANE:
        return _bitplane_pack(_delta_token(block))
    if mode == Mode.ZSTD_ORDERED_DELTA_BITPLANE:
        return _bitplane_pack(_delta_token(_float_flip(block)))
    if mode == Mode.ZSTD_TRACE_EXPDELTA_BITPLANE:
        return _trace_expdelta_pack(block, dtype_code)
    if mode in (Mode.DICTIONARY, Mode.ZSTD_DICTIONARY):
        return _encode_dictionary(block, rle=False)
    if mode in (Mode.RLE_DICTIONARY, Mode.ZSTD_RLE_DICTIONARY):
        return _encode_dictionary(block, rle=True)
    raise ValueError(f"unsupported mode {mode}")


def _inverse(
    transformed: bytes,
    mode: Mode,
    shape: tuple[int, int],
    dtype_code: int = DTYPE_FP16,
) -> np.ndarray:
    nt, nc = shape
    n = nt * nc
    if mode in (Mode.RAW, Mode.ZSTD_RAW):
        return _from_raw_bytes(transformed, shape)
    if mode == Mode.ZSTD_CHANNEL:
        return _from_raw_bytes(transformed, (nc, nt)).T.copy()
    if mode == Mode.ZSTD_BYTE_SHUFFLE:
        return _byte_unshuffle(transformed, n).reshape(nt, nc)
    if mode == Mode.ZSTD_CHANNEL_BYTE_SHUFFLE:
        return _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
    if mode == Mode.ZSTD_XOR_TOKEN:
        r = _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
        return np.bitwise_xor.accumulate(r, axis=0).astype(np.uint16)
    if mode == Mode.ZSTD_DELTA_TOKEN:
        r = _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
        return _undelta_token(r)
    if mode == Mode.ZSTD_ORDERED_DELTA_TOKEN:
        r = _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
        return _float_unflip(_undelta_token(r))
    if mode == Mode.ZSTD_ORDERED_DELTA2_TOKEN:
        r = _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
        return _float_unflip(_undelta2_token(r))
    if mode == Mode.ZSTD_DELTA_2D:
        r = _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
        return _unpredict2d(r, xor=False)
    if mode == Mode.ZSTD_XOR_2D:
        r = _byte_unshuffle(transformed, n).reshape(nc, nt).T.copy()
        return _unpredict2d(r, xor=True)
    if mode == Mode.ZSTD_BITPLANE:
        return _bitplane_unpack(transformed, shape)
    if mode == Mode.ZSTD_FIELDS:
        return _fields_unpack(transformed, shape)
    if mode == Mode.ZSTD_XOR_TOKEN_BITPLANE:
        r = _bitplane_unpack(transformed, shape)
        return np.bitwise_xor.accumulate(r, axis=0).astype(np.uint16)
    if mode == Mode.ZSTD_DELTA_TOKEN_BITPLANE:
        return _undelta_token(_bitplane_unpack(transformed, shape))
    if mode == Mode.ZSTD_ORDERED_DELTA_BITPLANE:
        r = _bitplane_unpack(transformed, shape)
        return _float_unflip(_undelta_token(r))
    if mode == Mode.ZSTD_TRACE_EXPDELTA_BITPLANE:
        return _trace_expdelta_unpack(transformed, shape, dtype_code)
    if mode in (Mode.DICTIONARY, Mode.ZSTD_DICTIONARY):
        return _decode_dictionary(transformed, shape, rle=False)
    if mode in (Mode.RLE_DICTIONARY, Mode.ZSTD_RLE_DICTIONARY):
        return _decode_dictionary(transformed, shape, rle=True)
    raise ValueError(f"unsupported mode {mode}")


def _mode_uses_zstd(mode: Mode) -> bool:
    return mode.name.startswith("ZSTD_")


def encode(
    array: np.ndarray,
    *,
    dtype_code: int = DTYPE_FP16,
    block_heads: int = 1,
    block_tokens: int = 256,
    block_channels: int = 16,
    zstd_level: int = 3,
    modes: Sequence[Mode] = DEFAULT_MODES,
    raw_fallback: bool = True,
) -> tuple[bytes, EncodeStats]:
    """Encode a [heads,tokens,channels] 16-bit tensor into a real byte stream.

    raw_fallback=True is the production/default contract.  Setting it false
    is intended only for forced-mode ablations and may expand the stream.
    """
    words = _as_u16_words(array)
    if dtype_code not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError("invalid dtype_code")
    if block_heads <= 0 or block_tokens <= 0 or block_channels <= 0:
        raise ValueError("block dimensions must be positive")
    hdim, sdim, ddim = map(int, words.shape)
    coords = [
        (
            h0,
            min(block_heads, hdim - h0),
            t0,
            min(block_tokens, sdim - t0),
            c0,
            min(block_channels, ddim - c0),
        )
        for h0 in range(0, hdim, block_heads)
        for t0 in range(0, sdim, block_tokens)
        for c0 in range(0, ddim, block_channels)
    ]
    nblocks = len(coords)
    payload_start = HEADER.size + nblocks * ENTRY.size
    payloads: list[bytes] = []
    entries: list[BlockEntry] = []
    mode_counts: dict[str, int] = {}
    offset = payload_start

    mode_list = [Mode(m) for m in modes]
    if raw_fallback and Mode.RAW not in mode_list:
        mode_list.insert(0, Mode.RAW)
    if not mode_list:
        raise ValueError("at least one mode is required")

    for h0, nh, t0, nt, c0, nc in coords:
        # Canonical 2-D block: token rows, with (head, channel) columns.
        # This exposes longer per-channel token streams while allowing a tunable
        # head-group/random-access trade-off.  The reshape is exactly invertible.
        cube = words[h0 : h0 + nh, t0 : t0 + nt, c0 : c0 + nc]
        block = np.ascontiguousarray(cube.transpose(1, 0, 2).reshape(nt, nh * nc))
        raw = _raw_bytes(block)
        best_mode: Mode | None = Mode.RAW if raw_fallback else None
        best_payload: bytes | None = raw if raw_fallback else None
        for mode in mode_list:
            if mode == Mode.RAW:
                payload = raw
            else:
                transformed = _transform(block, mode, dtype_code)
                if transformed is None:
                    continue
                payload = _zcompress(transformed, zstd_level) if _mode_uses_zstd(mode) else transformed
            if best_payload is None or len(payload) < len(best_payload):
                best_mode, best_payload = mode, payload
        if best_mode is None or best_payload is None:
            raise ValueError("no requested mode supports this block geometry")
        crc = binascii.crc32(raw) & 0xFFFFFFFF
        entries.append(BlockEntry(offset, len(best_payload), best_mode, 0, len(raw), crc))
        payloads.append(best_payload)
        offset += len(best_payload)
        mode_counts[best_mode.name] = mode_counts.get(best_mode.name, 0) + 1

    header = HEADER.pack(
        MAGIC,
        VERSION,
        dtype_code,
        3,
        0,
        hdim,
        sdim,
        ddim,
        block_heads,
        block_tokens,
        block_channels,
        nblocks,
        payload_start,
    )
    directory = b"".join(
        ENTRY.pack(e.offset, e.length, int(e.mode), e.flags, e.raw_length, e.crc32)
        for e in entries
    )
    blob = header + directory + b"".join(payloads)
    raw_bytes = words.size * 2
    stats = EncodeStats(
        raw_bytes=raw_bytes,
        encoded_bytes=len(blob),
        ratio=raw_bytes / len(blob),
        header_bytes=HEADER.size,
        directory_bytes=len(directory),
        payload_bytes=sum(map(len, payloads)),
        nblocks=nblocks,
        mode_counts=mode_counts,
    )
    return blob, stats


def parse_header(blob: bytes | bytearray | memoryview) -> tuple[HeaderInfo, list[BlockEntry]]:
    view = memoryview(blob)
    if len(view) < HEADER.size:
        raise ValueError("truncated AEX-KV header")
    magic, version, dtype_code, ndim, flags, h, s, d, bh, bt, bc, nb, payload_start = HEADER.unpack_from(view, 0)
    if magic != MAGIC or version != VERSION or ndim != 3:
        raise ValueError("invalid AEX-KV stream")
    expected_payload = HEADER.size + nb * ENTRY.size
    if payload_start != expected_payload or payload_start > len(view):
        raise ValueError("invalid directory/payload offset")
    entries: list[BlockEntry] = []
    pos = HEADER.size
    for _ in range(nb):
        off, length, mode, eflags, raw_len, crc = ENTRY.unpack_from(view, pos)
        pos += ENTRY.size
        if off < payload_start or off + length > len(view):
            raise ValueError("block payload outside stream")
        entries.append(BlockEntry(off, length, Mode(mode), eflags, raw_len, crc))
    if bh <= 0 or bt <= 0 or bc <= 0:
        raise ValueError("invalid block geometry")
    return HeaderInfo(dtype_code, (h, s, d), bh, bt, bc, nb, payload_start), entries


def _block_coords(header: HeaderInfo) -> list[tuple[int, int, int, int, int, int]]:
    hdim, sdim, ddim = header.shape
    return [
        (
            h0,
            min(header.block_heads, hdim - h0),
            t0,
            min(header.block_tokens, sdim - t0),
            c0,
            min(header.block_channels, ddim - c0),
        )
        for h0 in range(0, hdim, header.block_heads)
        for t0 in range(0, sdim, header.block_tokens)
        for c0 in range(0, ddim, header.block_channels)
    ]


def decode_block(blob: bytes, block_index: int) -> tuple[tuple[int, int, int, int, int, int], np.ndarray]:
    header, entries = parse_header(blob)
    coords = _block_coords(header)
    if not 0 <= block_index < len(entries):
        raise IndexError(block_index)
    h0, nh, t0, nt, c0, nc = coords[block_index]
    entry = entries[block_index]
    payload = blob[entry.offset : entry.offset + entry.length]
    mode = entry.mode
    if _mode_uses_zstd(mode):
        # Transforms are at most field/bitplane padded above raw by < 32 bytes.
        transformed = _zdecompress(payload, max(entry.raw_length + 128, entry.raw_length * 2 + 128))
    else:
        transformed = payload
    block = _inverse(transformed, mode, (nt, nh * nc), header.dtype_code)
    raw = _raw_bytes(block)
    if len(raw) != entry.raw_length or (binascii.crc32(raw) & 0xFFFFFFFF) != entry.crc32:
        raise ValueError(f"block {block_index} checksum/length mismatch")
    cube = block.reshape(nt, nh, nc).transpose(1, 0, 2).copy()
    return (h0, nh, t0, nt, c0, nc), cube


def decode_bits(blob: bytes) -> tuple[np.ndarray, int]:
    header, entries = parse_header(blob)
    out = np.empty(header.shape, dtype=np.uint16)
    coords = _block_coords(header)
    if len(coords) != len(entries):
        raise ValueError("block count mismatch")
    for i in range(len(entries)):
        (h0, nh, t0, nt, c0, nc), block = decode_block(blob, i)
        out[h0 : h0 + nh, t0 : t0 + nt, c0 : c0 + nc] = block
    return out, header.dtype_code


def decode(blob: bytes) -> np.ndarray:
    words, dtype_code = decode_bits(blob)
    if dtype_code == DTYPE_FP16:
        return words.view(np.float16)
    # NumPy has no portable native BF16 dtype; preserve/return raw words.
    return words


def verify_exact(original: np.ndarray, blob: bytes) -> bool:
    expected = _as_u16_words(original)
    actual, _ = decode_bits(blob)
    return np.array_equal(expected, actual)
