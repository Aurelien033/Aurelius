"""Parameter count audit — Solus-MoE 8B and Solus-7B Dense.

This module is the authoritative source for all model-size claims in
ARCHITECTURE.md.  It is importable without side effects and runnable as a script.

Architecture (verified)
========================
  H  (hidden)           = 4096
  Nh (heads)            = 32
  Nkv (key-value heads) = 7
  D (head_dim)          = H // Nh = 128

  Dense  per layer  = 24   (not in moe_layer_indices)
  MoE    per layer  =  8

  I_d  = intermediate_size          = 12,800    (dense FFN hidden)
  I_e  = expert_intermediate_size   =  3,072    (per-expert FFN hidden)
  E    = moe_num_experts            =  8

  V = vocab_size = 128,256
"""
from __future__ import annotations


def count_solus_7b_dense() -> dict:
    H = 4096; Nh = 32; Nkv = 7; D = H // Nh
    V = 128256; N = 30  # Solus-7B: 30 layers
    qkv = (Nh + 2 * Nkv) * H * D
    o   = H * Nh * D
    ffn = 3 * H * 22016
    norm = 2 * H
    per  = qkv + o + ffn + norm
    emb  = V * H
    total = emb + N * per
    print("=== Solus-7B Dense ===")
    print(f"  embeddings        : {emb:>15,} = {emb/1e6:6.1f}M")
    print(f"  dense FFN         : {3*H*22016:>15,} = {3*H*22016/1e6:6.1f}M per dense layer")
    print(f"  per_dense layer   : {per:>15,} = {per/1e6:6.1f}M")
    print(f"  30x dense layers  : {N*per:>15,} = {N*per/1e6:6.1f}M")
    print(f"  final norm        : {norm//2:>15,} = {norm//2/1e6:6.3f}M")
    print(f"  TOTAL             : {total:>15,} = {total/1e6:.1f}M")
    return dict(total=total, total_m=round(total/1e6,1))


def count_solus_moe_8b() -> dict:
    H = 4096; Nh = 32; Nkv = 7; D = H // Nh
    V = 128256; Nd = 24; Nm = 8; E = 8; Id = 12800; Ie = 3072

    qkv = (Nh + 2 * Nkv) * H * D  # Q+K+V (all dense and MoE layers)
    o   = H * Nh * D
    ffn_d = 3 * H * Id
    norm = 2 * H
    per_d = qkv + o + ffn_d + norm

    expert_ffn   = 3 * H * Ie
    pool_experts = E * expert_ffn
    router_gate  = H * E
    per_m = qkv + o + pool_experts + router_gate + norm

    emb     = V * H
    final_n = H
    total   = emb + Nd * per_d + Nm * per_m + final_n

    print(f"\n=== Solus-MoE 8B ===")
    print(f"  Token embeddings (VxH)     : {emb:>15,} = {emb/1e6:6.1f}M  ({emb:,})")
    print(f"  {'─'*55}")
    print(f"  24xDense layers (Nd)       : {Nd*per_d:>15,} = {Nd*per_d/1e6:6.1f}M")
    print(f"     per dense layer          : {per_d:>15,} = {per_d/1e6:6.3f}M")
    print(f"     attn (qkv+o)             : {qkv+o:>15,} = {(qkv+o)/1e6:.3f}M")
    print(f"     ffn (3xHxI)              : {ffn_d:>15,} = {ffn_d/1e6:.3f}M")
    print(f"     2xRMSNorm                : {norm:>15,} = {norm/1e6:.3f}M")
    print(f"  8xMoE layers (Nm)          : {Nm*per_m:>15,} = {Nm*per_m/1e6:6.1f}M")
    print(f"     per MoE layer            : {per_m:>15,} = {per_m/1e6:.3f}M")
    print(f"     attn (qkv+o)             : {qkv+o:>15,} = {(qkv+o)/1e6:.3f}M")
    print(f"     pool (8x experts)         : {pool_experts:>15,} = {pool_experts/1e6:.3f}M")
    print(f"       8 experts x 37,748,736 : {expert_ffn:>12,} each")
    print(f"     router gate (HxE)        : {router_gate:>15,} = {router_gate/1e6:.3f}M")
    print(f"     2xRMSNorm                : {norm:>15,} = {norm/1e6:.3f}M")
    print(f"  final RMSNorm (H)           : {final_n:>15,} = {final_n/1e6:.3f}M")
    print(f"  {'═'*55}")
    print(f"  GRAND TOTAL                 : {total:>15,} = {total/1e6:.1f}M")
    return dict(
        total=total, total_m=round(total/1e6,1),
        per_dense_m=round(per_d/1e6,3), per_moe_m=round(per_m/1e6,3),
        per_d=per_d, per_m=per_m,
        pool_experts=pool_experts, expert_per=expert_ffn,
        ffn_d=ffn_d, router_gate=router_gate,
    )


if __name__ == "__main__":
    count_solus_7b_dense()
    count_solus_moe_8b()
