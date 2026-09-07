"""Solus-7B — attention layer using half-dimension RoPE convention."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SolusConfig
from .rope import RotaryEmbedding, apply_rope


class SolusAttention(nn.Module):
    """GQA — 32 Q heads share 7 KV heads; RoPE applied to Q and K."""

    def __init__(self, cfg: SolusConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx    = layer_idx
        self.num_heads    = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        # head_dim_full = hidden_size // heads  (full per-head vector length)
        self.head_dim_full = cfg.hidden_size // cfg.num_attention_heads
        self.scale         = self.head_dim_full ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads    * self.head_dim_full, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim_full, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim_full, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim_full, cfg.hidden_size, bias=False)

        # pair_dim = head_dim_full // 2  (size of one rotation axis)
        pair_dim = self.head_dim_full // 2
        self.q_rope = RotaryEmbedding(
            pair_dim=pair_dim, max_seq_len=cfg.max_position_embeddings,
            theta=cfg.rope_theta, rope_scaling=cfg.rope_scaling
        )
        self.k_rope = RotaryEmbedding(
            pair_dim=pair_dim, max_seq_len=cfg.max_position_embeddings,
            theta=cfg.rope_theta, rope_scaling=cfg.rope_scaling
        )

        self.attn_dropout = nn.Dropout(cfg.attention_dropout) if cfg.attention_dropout > 0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        bsz, seq_len, _ = hidden_states.shape

        # Project → (B, S, heads * dim)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape → (B, heads, S, head_dim_full)
        q = q.view(bsz, seq_len, self.num_heads,      self.head_dim_full).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads,   self.head_dim_full).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads,   self.head_dim_full).transpose(1, 2)

        # RoPE (in-place-free)
        q = self.q_rope(q)
        k = self.k_rope(k)

        # GQA: replicate KV heads to match Q heads
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(bsz, self.num_kv_heads, n_rep, seq_len, self.head_dim_full).reshape(bsz, self.num_heads, seq_len, self.head_dim_full)
            v = v[:, :, None, :, :].expand(bsz, self.num_kv_heads, n_rep, seq_len, self.head_dim_full).reshape(bsz, self.num_heads, seq_len, self.head_dim_full)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=True,
        )  # (B, heads, S, head_dim_full)

        # Flatten heads → (B, S, hidden)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.cfg.hidden_size)
        return self.o_proj(attn_output), None
