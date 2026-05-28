"""Aurelius AMCTransformer — hybrid attention + SSM model with per-layer memory.

Architecture:
  - Even layers (0, 2, 4, ...): multi-head attention with RoPE
  - Odd layers (1, 3, 5, ...): AMC SSM working memory (Tier-1)
  - SSM layers emit memory blocks; promotion gates handle Tier-1 → Tier-2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.memory.amc_runtime_cache import AMCMemoryBlock
from src.model.amc_promotion import PromotionGate
from src.model.amc_ssm_layer import AMCForwardOutput, AMCSSMConfig, AMCSSMLayer
from src.model.attention import apply_rope, precompute_rope_frequencies
from src.model.config import AureliusConfig
from src.model.ffn import SwiGLUFFN
from src.model.hlm_bank_adapter import HLMPreferenceAdapter, HLMPreferenceAdapterConfig
from src.model.mla import MLAConfig, MultiheadLatentAttention
from src.model.rms_norm import RMSNorm

if TYPE_CHECKING:
    from src.memory.amc_tier2 import AMCTier2Hook
    from src.memory.amc_tier3 import AMCTier3Hook


@dataclass
class AMCTransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None = None
    d_ff: int | None = None
    head_dim: int | None = None
    kv_lrank: int = 64
    d_conv: int = 4
    ssm_d_state: int = 64
    ssm_expand: int = 2
    ssm_headdim: int = 64
    max_seq_len: int = 4096
    tie_embeddings: bool = True
    promotion_temperature: float = 0.5
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    ssm_layers_at: tuple[int, ...] | None = None
    hlm_bank_size: int = 14
    hlm_bank_dim: int | None = None
    hlm_bank_top_k: int = 4
    hlm_bank_inject_scale: float = 0.1
    hlm_bank_read_layers: tuple[int, ...] | None = None
    use_hlm_bank: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = max(1, self.n_heads // 2)
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        if self.d_ff is None:
            self.d_ff = self.d_model * 4

    def get_ssm_layer_indices(self) -> set[int]:
        if self.ssm_layers_at is not None:
            return set(self.ssm_layers_at)
        return set(range(1, self.n_layers, 2))

    def to_aurelius_config(self) -> AureliusConfig:
        return AureliusConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads or max(1, self.n_heads // 2),
            head_dim=self.head_dim or (self.d_model // self.n_heads),
            d_ff=self.d_ff or (self.d_model * 4),
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            dropout=self.dropout,
            tie_embeddings=self.tie_embeddings,
        )


@dataclass
class AMCModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    memory_blocks: list[AMCMemoryBlock] = field(default_factory=list)
    surprise_scores: torch.Tensor | None = None
    gate_outputs: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    promotion_loss: torch.Tensor | None = None
    bank_alpha: torch.Tensor | None = None
    bank_confidence: torch.Tensor | None = None
    bank_telemetry: dict[str, float | int] | None = None


class _AMCGroupedQueryAttention(nn.Module):
    """Lightweight GQA with RoPE for AMCTransformer attention layers."""

    def __init__(self, config: AMCTransformerConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or max(1, config.n_heads // 2)
        self.head_dim = config.head_dim or (config.d_model // config.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(attn)


class MLAAttentionLayer(nn.Module):
    """Pre-norm MLA + FFN block for even-index layers (compressed KV cache)."""

    def __init__(self, config: AMCTransformerConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        head_dim = config.head_dim or (config.d_model // config.n_heads)
        aurelius = config.to_aurelius_config()
        mla_cfg = MLAConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=head_dim,
            kv_lora_rank=config.kv_lrank,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mla = MultiheadLatentAttention(mla_cfg)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(aurelius)

    def forward(self, hidden: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        _ = freqs_cis  # legacy GQA freqs; MLA uses src.model.rope.RotaryEmbedding
        mla_out = self.mla(self.attn_norm(hidden))
        if isinstance(mla_out, tuple):
            mla_out = mla_out[0]
        hidden = hidden + mla_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class StandardAttentionLayer(nn.Module):
    """Pre-norm GQA + FFN with RoPE (legacy fallback; even layers use MLAAttentionLayer)."""

    def __init__(self, config: AMCTransformerConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        aurelius = config.to_aurelius_config()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = _AMCGroupedQueryAttention(config, layer_index)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(aurelius)

    def forward(self, hidden: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(hidden), freqs_cis)
        hidden = hidden + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class AMCTransformer(nn.Module):
    """Hybrid attention + SSM transformer with AMC memory integration."""

    def __init__(self, config: AMCTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.ssm_layer_indices = config.get_ssm_layer_indices()
        self._ssm_layer_indices_sorted = sorted(self.ssm_layer_indices)

        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx in self.ssm_layer_indices:
                ssm_cfg = AMCSSMConfig(
                    d_model=config.d_model,
                    d_state=config.ssm_d_state,
                    d_conv=config.d_conv,
                    expand=config.ssm_expand,
                    headdim=config.ssm_headdim,
                )
                self.layers.append(AMCSSMLayer(ssm_cfg, layer_index=layer_idx))
            else:
                self.layers.append(MLAAttentionLayer(config, layer_index=layer_idx))

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.promotion_gates = nn.ModuleDict(
            {
                str(layer_idx): PromotionGate(
                    config.d_model,
                    temperature=config.promotion_temperature,
                )
                for layer_idx in self.ssm_layer_indices
            }
        )

        head_dim = config.head_dim or (config.d_model // config.n_heads)
        freqs = precompute_rope_frequencies(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # DreamBank adapter (optional)
        bank_dim = config.hlm_bank_dim or config.kv_lrank
        adapter_cfg = HLMPreferenceAdapterConfig(
            d_model=config.d_model,
            bank_dim=bank_dim,
            inject_scale=config.hlm_bank_inject_scale,
            top_k=config.hlm_bank_top_k,
        )
        self.hlm_bank_adapter = HLMPreferenceAdapter(adapter_cfg)

        self._tier2_hook: AMCTier2Hook | None = None
        self._tier3_hook: AMCTier3Hook | None = None

    def wire_amc_hooks(
        self,
        tier2: AMCTier2Hook,
        tier3: AMCTier3Hook | None = None,
    ) -> None:
        self._tier2_hook = tier2
        self._tier3_hook = tier3

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        session_id: str | None = None,
        step: int = 0,
        use_amc: bool = True,
        return_memory: bool = False,
        preference_bank: object = None,
    ) -> AMCModelOutput:
        _ = session_id
        _ = self._tier3_hook
        batch, seq_len = input_ids.shape
        hidden = self.embed(input_ids)
        freqs = self.freqs_cis[:seq_len]

        memory_blocks: list[AMCMemoryBlock] = []
        surprise_scores: list[torch.Tensor] = []
        gate_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        promotion_loss: torch.Tensor | None = None

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.ssm_layer_indices and isinstance(layer, AMCSSMLayer):
                amc_out: AMCForwardOutput = layer(hidden, step=step + layer_idx)
                hidden = amc_out.hidden

                if use_amc:
                    memory_blocks.append(amc_out.memory_block)
                    surprise_scores.append(amc_out.surprise_scores)

                    gate = self.promotion_gates[str(layer_idx)]
                    last_hidden = hidden[:, -1, :]
                    last_surprise = amc_out.surprise_scores[:, -1]
                    store_hard, store_soft = gate(last_hidden, last_surprise)
                    gate_outputs.append((store_hard, store_soft))

                    if self._tier2_hook is not None:
                        for batch_idx in range(batch):
                            if store_hard[batch_idx].item() > 0.5:
                                self._tier2_hook.observe(
                                    role="working_memory",
                                    content=f"layer_{layer_idx}_step_{step}",
                                    surprise=float(last_surprise[batch_idx].item()),
                                    importance=float(store_soft[batch_idx].item()),
                                )
            else:
                hidden = layer(hidden, freqs)

        hidden = self.norm(hidden)

        # DreamBank: optional bank-biased residual (only when bank is provided and enabled)
        bank_alpha: torch.Tensor | None = None
        bank_confidence: torch.Tensor | None = None
        bank_telemetry: dict[str, float | int] | None = None
        if self.config.use_hlm_bank and preference_bank is not None:
            adapter_out = self.hlm_bank_adapter(hidden, preference_bank)
            hidden = adapter_out.hidden
            bank_alpha = adapter_out.alpha
            bank_confidence = adapter_out.confidence
            bank_telemetry = preference_bank.telemetry()

        logits = self.lm_head(hidden)

        if self.training and gate_outputs:
            total = hidden.new_zeros(())
            for gate_idx, (hard, soft) in enumerate(gate_outputs):
                layer_key = str(self._ssm_layer_indices_sorted[gate_idx])
                target = surprise_scores[gate_idx][:, -1].detach()
                total = total + self.promotion_gates[layer_key].promote_loss(soft, target)
            promotion_loss = total / len(gate_outputs)

        stacked_surprise = torch.stack(surprise_scores, dim=0) if surprise_scores else None

        return AMCModelOutput(
            logits=logits,
            hidden_states=hidden,
            memory_blocks=memory_blocks if return_memory else [],
            surprise_scores=stacked_surprise,
            gate_outputs=gate_outputs if return_memory else [],
            promotion_loss=promotion_loss,
            bank_alpha=bank_alpha,
            bank_confidence=bank_confidence,
            bank_telemetry=bank_telemetry,
        )

    def reset_amc_state(self) -> None:
        for layer in self.layers:
            if isinstance(layer, AMCSSMLayer):
                layer.reset_state()

    @property
    def ssm_layer_count(self) -> int:
        return len(self.ssm_layer_indices)

    @property
    def attention_layer_count(self) -> int:
        return self.config.n_layers - self.ssm_layer_count


__all__ = [
    "AMCTransformer",
    "AMCTransformerConfig",
    "AMCModelOutput",
    "MLAAttentionLayer",
    "StandardAttentionLayer",
]
