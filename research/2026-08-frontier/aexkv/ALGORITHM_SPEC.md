# AEX-KV v2 Algorithm and Binary Format Specification

Status: research prototype, stream version 2 (`AEXKV002`).

## 1. Contract

Input is a rank-3 tensor `X[H,S,D]` of 16-bit words. `dtype_code=1`
interprets the words as IEEE binary16; `dtype_code=2` interprets them as
bfloat16. Encoding and decoding operate on words, not numerical values. The
required property is

`decode_bits(encode(X)) == X` element-for-element as `uint16`.

This preserves `+0`, `-0`, all subnormals, infinities, every NaN sign/quiet bit,
and every NaN payload. No quality metric can substitute for this contract.

## 2. Blocking and canonicalization

The tensor is partitioned into blocks of at most `G` heads, `T` tokens, and `C`
channels. For a cube `X[h:h+g,t:t+n,c:c+d]`, AEX-KV forms the reversible matrix

`W = transpose(cube, token, head, channel).reshape(n, g*d)`.

Grouping heads increases compression context; smaller `G` gives finer random
access. Coordinates are implicit from global shape and `(G,T,C)`, so directory
entries do not repeat coordinates.

## 3. Reversible mode portfolio

Each block independently evaluates the following candidates. Every transformed
byte string is either stored directly or compressed as one zstd frame.

1. `RAW`: canonical uint16 words, little-endian.
2. `ZSTD_RAW`: raw words + zstd.
3. `ZSTD_CHANNEL`: channel-major word transpose + zstd.
4. `ZSTD_BYTE_SHUFFLE`: low bytes followed by high bytes + zstd.
5. `ZSTD_CHANNEL_BYTE_SHUFFLE`: channel-major byte shuffle + zstd (the FP16
   specialization of a TDT/Shuffle-style transform).
6. `ZSTD_XOR_TOKEN`: `R[0]=W[0]`, `R[t]=W[t] XOR W[t-1]`, channel byte shuffle.
7. `ZSTD_DELTA_TOKEN`: modulo-`2^16` first difference, channel byte shuffle.
8. `ZSTD_ORDERED_DELTA_TOKEN`: float-flip, modulo first difference, channel byte
   shuffle.
9. `ZSTD_ORDERED_DELTA2_TOKEN`: float-flip, second difference, channel byte
   shuffle.
10. `ZSTD_DELTA_2D`: reversible additive Lorenzo residual over token/channel.
11. `ZSTD_XOR_2D`: reversible XOR Lorenzo residual.
12. `ZSTD_BITPLANE`: 16-by-N bit-matrix transpose + zstd.
13. `ZSTD_FIELDS`: a bijective 1/5/10 bit-field partition + zstd.  This is the
    native FP16 sign/exponent/mantissa layout; for BF16 it remains exact but is
    only a reversible bit grouping, not the BF16 1/8/7 semantic partition.
14. `DICTIONARY` / `ZSTD_DICTIONARY`: exact local uint16 dictionary with
    fixed-width packed indices (one-channel blocks only).
15. `RLE_DICTIONARY` / `ZSTD_RLE_DICTIONARY`: dictionary indices plus exact
    varint run lengths.
16. `ZSTD_XOR_TOKEN_BITPLANE`: token XOR residual followed by bit-plane
    transpose and zstd.
17. `ZSTD_DELTA_TOKEN_BITPLANE`: modulo delta residual followed by bit planes.
18. `ZSTD_ORDERED_DELTA_BITPLANE`: float-flip delta followed by bit planes.
19. `ZSTD_TRACE_EXPDELTA_BITPLANE`: dtype-aware channel grouping,
    per-channel median base exponents, modular exponent deltas, separate
    sign/delta/mantissa bit planes, then zstd. FP16 uses 1/5/10 fields and BF16
    uses 1/8/7. This is a documented strengthened TRACE-style proxy, not an
    official reproduction of TRACE's undisclosed base-selection/packing rules.

### Float-flip

For raw word `u`, define the bijection

```
F(u) = (~u) & 0xffff,  if u & 0x8000
       u ^ 0x8000,     otherwise.
```

Its inverse is

```
F^-1(v) = v ^ 0x8000,  if v & 0x8000
          (~v) & 0xffff, otherwise.
```

For ordinary finite values this maps sign-magnitude IEEE words to monotone
unsigned keys; it remains a bijection for special patterns. Modulo residuals
therefore expose numerical locality without performing floating-point
arithmetic or losing payload bits.

## 4. Verified admission

For block `b` and candidate mode `m`, the encoder constructs the **actual** final
payload `P_{b,m}`. It chooses

`m_b = argmin_m |P_{b,m}|`.

`RAW` is always a candidate in production. Selection never uses entropy alone,
and it counts zstd frame overhead, dictionary contents, packed tails, and byte
padding. Ties retain the earlier/cheaper mode.

## 5. Stream format

All integers are little-endian.

### Header (`<8sBBBBIIIHHHIQ>`, 42 bytes)

- magic: 8 bytes, `AEXKV002`
- version, dtype code, rank (=3), flags: four uint8
- `H,S,D`: three uint32
- block heads `G`, tokens `T`, channels `C`: three uint16
- number of blocks: uint32
- absolute payload start: uint64

### Directory entry (`<QIBBII>`, 22 bytes each)

- absolute payload offset: uint64
- payload length: uint32
- mode ID: uint8
- flags: uint8
- uncompressed canonical block bytes: uint32
- CRC32 of uncompressed canonical block: uint32

Payloads follow the fixed directory with no hidden side metadata. Block
coordinates are derived from directory index and header geometry.

## 6. Exactness theorem

**Theorem.** For every rank-3 uint16 tensor `X`, valid block geometry, and mode
selected by the encoder, `decode_bits(encode(X)) = X`.

**Proof sketch.** Canonicalization is a permutation. Raw, byte shuffle,
channel transpose, field packing, fixed-width packing, bit-plane transpose,
XOR differencing, modulo differencing, float-flip, 2-D Lorenzo transforms,
dictionary indexing, and RLE all have explicit inverses. zstd decompression
returns the exact transformed byte sequence. The decoder applies the inverse
of the directory-selected transform and the inverse canonical permutation.
Blocks partition the tensor without overlap or omission. Composition of these
bijections reconstructs every original word. CRC32 detects, but is not needed
to prove, successful inversion.

The test suite also enumerates all 65,536 possible words, random tensors and
geometries, specials, corruption, random block access, FP16 captures, and BF16
word streams.

## 7. Size bound

Let `N` be words and `B` blocks. Because raw bytes are always an admissible
payload,

`|AEX(X)| <= 2N + 42 + 22B`.

Thus pathological incompressibility expands only by framing. With one block,
the relative expansion is `(42+22)/(2N)`; fine fragmentation increases this
term and is reported explicitly.

## 8. Online/cold-tier operation

A serving implementation keeps the active token tail raw. When `T` tokens fill,
it seals the block asynchronously, emits its directory entry and payload, and
starts a new raw tail. Dense attention would need to decode all cold blocks on
every step; the paper therefore treats hot-HBM acceleration as a falsifiable
systems hypothesis, not an assumed benefit. The primary defensible targets are
paused-session stores, exact prefix stores, CPU/NVMe offload, and sparse
retrieval tiers.
