#!/usr/bin/env python3
"""
Solus-MoE 8B — Parameter count verification and design matrix.
"""
hidden_size      = 4096
num_hidden_layers= 32
num_attn_heads   = 32
num_kv_heads     = 7
vocab_size       = 128256

inter_dense   = 12800     # dense FFN intermediate
expert_inter  = 3072      # expert FFN intermediate
num_experts   = 8
top_k         = 2
n_moe_layers  = 8
n_dense_layers= num_hidden_layers - n_moe_layers  # 24


def layer_params(is_moe: bool) -> dict:
    """Per-layer parameter breakdown (excluding embeddings)."""
    D = hidden_size // num_attn_heads         # head_dim = 128
    Q     = num_attn_heads  * hidden_size * D    # Q proj
    KV    = 2 * num_kv_heads * hidden_size * D  # K+V proj
    O     = num_attn_heads  * hidden_size * D    # O proj
    attn  = Q + KV + O                          # ≈ 40.9M

    norm  = hidden_size                          # 0.004M

    if is_moe:
        inter        = expert_inter
        ffn_per_expert = 3 * hidden_size * inter   # gate+up+down
        expert_total   = num_experts * ffn_per_expert
        router         = hidden_size * num_experts  # gating linear
        total          = attn + expert_total + norm + router
        return dict(attn=attn, experts=expert_total, norm=norm,
                    router=router, total=total, ffn_per_expert=ffn_per_expert)
    else:
        inter = inter_dense
        ffn   = 3 * hidden_size * inter   # gate+up+down
        total = attn + ffn + norm
        return dict(attn=attn, ffn=ffn, norm=norm, total=total)


dp  = layer_params(is_moe=False)
mp  = layer_params(is_moe=True )

embed    = vocab_size * hidden_size
lm_head  = 0   # tied to embedding

totals = (embed
          + n_dense_layers * dp["total"]
          + n_moe_layers  * mp["total"]
          + lm_head)

print("═"*62)
print("SOLUS-MoE 8B — Design matrix (expert_inter=3072)")
print("═"*62)
print(f"\n{'Component':<28}  {'Each':>11}  Count")
print("-"*46)

print(f"\n{'EMBEDDINGS':<28}  {embed/1e6:>10.1f}M  × 1")

print(f"\n{'DENSE LAYERS ('+str(n_dense_layers)+')':<28}  "
      f"{dp['total']/1e6:>10.2f}M  ×{n_dense_layers}")
print(f"  Attention (Q+KV+O)    {dp['attn']/1e6:>9.3f}M")
print(f"  FFN (3*hi inter=12800) {dp['ffn']/1e6:>9.3f}M")
print(f"  Norm                  {dp['norm']/1e6:>9.4f}M")

print(f"\n{'MoE LAYERS ('+str(n_moe_layers)+')':<28}  "
      f"{mp['total']/1e6:>10.2f}M  ×{n_moe_layers}")
print(f"  Attention (Q+KV+O)    {mp['attn']/1e6:>9.3f}M")
print(f"  Per expert (3*hi*3072) {mp['ffn_per_expert']/1e6:>8.3f}M  × {num_experts} experts")
print(f"  Expert sub-total      {mp['experts']/1e6:>9.2f}M")
print(f"  Router (hi*8)          {mp['router']/1e3:>8.1f}K")
print(f"  Norm                  {mp['norm']/1e6:>9.4f}M")

print(f"\n{'='*46}")
print(f"  TOTAL PARAMS        {totals/1e9:>10.3f}B  ({totals/1e6:.0f}M)")
print(f"  vs Solus-Dense      {totals/1e9/6.868:>10.3f}×")

mo  = n_dense_layers/num_hidden_layers
moo = n_moe_layers/num_hidden_layers * (top_k/num_experts)
dom = mp['total']/num_hidden_layers
dpm = dp['total']/num_hidden_layers
print(f"\n── Compute per token (top-{top_k}) ───────────────────")
print(f"  Dense token share    {mo:>5.0%}  (24 × {dpm/1e6:.1f}M)")
print(f"  MoE token share      {moo:>5.0%}  (8 × {dom/1e6:.1f}M × {top_k}/{num_experts})")
print(f"  Effective vs dense   {mo+moo:>5.0%}")
print(f"\n✅ Target: ~8.0B  →  actual: {totals/1e6:.0f}M ({totals/1e9:.3f}B)")
