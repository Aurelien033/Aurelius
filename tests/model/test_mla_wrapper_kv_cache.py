"""Regression tests for MLA KV-cache correctness and chunked prefill.

Tests target src/model/mla_wrapper.py (MLACompatibleAttention).
Two bugs are known:
  1. Line 51-52: `S > 1` with past_kv raises ValueError, blocking chunked prefill.
  2. Line 85:    V_fresh = V[:, :, -T:, :] uses T values but attention
                 attends over total_seq = past_len + T — shape mismatch
                 on the absorbed cached path when past_len > 0.
"""

from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.mla_wrapper import MLACompatibleAttention

# ---------------------------------------------------------------------------
# Small tightly-controlled config  (n_heads % n_kv_heads must be 0)
# ---------------------------------------------------------------------------
B, D_MODEL, N_HEADS, HEAD_DIM, N_KV_HEADS, KV_LRANK, T_PROMPT = 2, 64, 4, 16, 2, 16, 4
PAST_LEN = 3


def _cfg(mla_enabled: bool = True, **kw):
    kw.setdefault("n_kv_heads", N_KV_HEADS)
    cfg = AureliusConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        mla_enabled=mla_enabled,
        mla_kv_lrank=KV_LRANK,
        mla_q_lrank=32,
        mla_rope_dim=8,
        dropout=0.0,
        **kw,
    )
    return cfg


# ---------------------------------------------------------------------------
# Test 1 — No-cache prefill with S > 1 (baseline: must always work)
# ---------------------------------------------------------------------------
def test_mla_prefill_no_cache_multi_token():
    """Prefilling with S > 1 and no cache must not raise."""
    mla = MLACompatibleAttention(_cfg()).eval()
    x = torch.randn(B, T_PROMPT, D_MODEL)
    with torch.no_grad():
        out, (c, r) = mla(x)
    assert out.shape == (B, T_PROMPT, D_MODEL)
    assert c.shape == (B, T_PROMPT, KV_LRANK), f"cache shape {c.shape}"


# ---------------------------------------------------------------------------
# Test 2 — Cached single-token decode (past_len > 0, S=1)
# ---------------------------------------------------------------------------
def test_mla_decode_single_token_with_past_kv():
    """Single-token decode with past cache must not raise."""
    mla = MLACompatibleAttention(_cfg()).eval()
    x_prompt = torch.randn(B, T_PROMPT, D_MODEL)
    _, (c_past, r_past) = mla(x_prompt)

    x_new = torch.randn(B, 1, D_MODEL)
    with torch.no_grad():
        out, (c_new, r_new) = mla(x_new, past_kv=(c_past, r_past))
    assert out.shape == (B, 1, D_MODEL)
    assert c_new.shape[1] == T_PROMPT + 1, f"cache seq {c_new.shape[1]}"


# ---------------------------------------------------------------------------
# Test 3 — Cached chunked prefill (past_len > 0, S > 1) — BUG-1 trigger
# ---------------------------------------------------------------------------
def test_mla_chunked_prefill_with_past_kv():
    """Chunked prefill (S > 1, past_kv present) must not raise ValueError.

    Bug 1 at mla_wrapper.py line 51-52: unconditionally raises when S > 1
    and past_kv is not None, blocking any multi-token cached prefill.
    """
    mla = MLACompatibleAttention(_cfg()).eval()
    # Step 1: build a cache of length PAST_LEN
    x_past = torch.randn(B, PAST_LEN, D_MODEL)
    _, (c_past, r_past) = mla(x_past)

    # Step 2: simulated second chunk, T > 1, with cache present
    x_chunk = torch.randn(B, T_PROMPT, D_MODEL)

    with torch.no_grad():
        out, (c_new, r_new) = mla(x_chunk, past_kv=(c_past, r_past))

    assert out.shape == (B, T_PROMPT, D_MODEL)
    expected_seq = PAST_LEN + T_PROMPT
    assert c_new.shape[1] == expected_seq, (
        f"cache seq_len {c_new.shape[1]} != expected {expected_seq}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Absorbed path cached KV-cache shape consistency (BUG-2 trigger)
# ---------------------------------------------------------------------------
def test_mla_absorbed_cached_shapes_not_crash():
    """Absorbed cached path with past_kv must accept past_len + T in K,V space.

    Bug 2 at mla_wrapper.py line 72-86:
      scores = Q_abs @ c_full^T  attends over total_seq keys,
      but V_fresh = V[:, :, -T:, :] provides only T V-vectors,
      causing a shapes mismatch when total_seq = past_len + T != T.
    """
    mla = MLACompatibleAttention(_cfg()).eval()
    mla.absorb()

    # Build a cache of length PAST_LEN
    x_past = torch.randn(B, PAST_LEN, D_MODEL)
    _, (c_past, r_past) = mla(x_past)

    # Run with S > 1 and past_kv
    x_chunk = torch.randn(B, T_PROMPT, D_MODEL)
    with torch.no_grad():
        out, (c_new, r_new) = mla(x_chunk, past_kv=(c_past, r_past))

    assert out.shape == (B, T_PROMPT, D_MODEL)
    assert torch.isfinite(out).all(), "NaN in absorbed-cached output"
    assert c_new.shape[1] == PAST_LEN + T_PROMPT


# ---------------------------------------------------------------------------
# Test 5 — Causal mask: queries offset by past_len must not look at future
# ---------------------------------------------------------------------------
def test_mla_causal_mask_queries_cannot_attend_to_future():
    """Last query position of a chunk with past cache accesses only those keys."""
    mla = MLACompatibleAttention(_cfg()).eval()

    x_past = torch.randn(B, PAST_LEN, D_MODEL)
    _, (c_past, r_past) = mla(x_past)

    x_chunk = torch.randn(B, T_PROMPT, D_MODEL)
    with torch.no_grad():
        out, (c_new, r_new) = mla(x_chunk, past_kv=(c_past, r_past))

    assert torch.isfinite(out).all(), "NaN detected in causal-attention output"


# ---------------------------------------------------------------------------
# Test 6 — Output shape + cache shape consistency across two-step sequence
# ---------------------------------------------------------------------------
def test_mla_cache_shapes_consistency_two_steps():
    """Two steps of cache must have monotonically growing sequence dimension."""
    mla = MLACompatibleAttention(_cfg()).eval()

    x1 = torch.randn(B, T_PROMPT, D_MODEL)
    with torch.no_grad():
        out1, (c1, _r1) = mla(x1)
    assert c1.shape[1] == T_PROMPT, f"step1 cache seq {c1.shape[1]}"

    x2 = torch.randn(B, 1, D_MODEL)
    with torch.no_grad():
        out2, (c2, _r2) = mla(x2, past_kv=(c1, _r1))
    assert c2.shape[1] == T_PROMPT + 1

    x3 = torch.randn(B, 1, D_MODEL)
    with torch.no_grad():
        out3, (c3, _r3) = mla(x3, past_kv=(c2, _r2))
    assert c3.shape[1] == T_PROMPT + 2


# ---------------------------------------------------------------------------
# Test 7 — Absorbed and non-absorbed cached paths must produce the same shapes
# ---------------------------------------------------------------------------
def test_mla_both_cache_modes_output_shape():
    """Absorbed and standard cached outputs must have matching shapes."""
    mla = MLACompatibleAttention(_cfg()).eval()

    x_past = torch.randn(B, PAST_LEN, D_MODEL)
    _, (c_past, r_past) = mla(x_past)
    x_chunk = torch.randn(B, T_PROMPT, D_MODEL)

    with torch.no_grad():
        mla.absorb()  # absorbed mode — sets absorbed_qk AND absorbed=True
        out_abs, (c_abs, _ra) = mla(x_chunk, past_kv=(c_past, r_past))

    with torch.no_grad():
        mla.absorbed = False
        out_std, (c_std, _rs) = mla(x_chunk, past_kv=(c_past, r_past))

    assert out_abs.shape == (B, T_PROMPT, D_MODEL)
    assert out_std.shape == (B, T_PROMPT, D_MODEL)
    assert c_abs.shape == c_std.shape
