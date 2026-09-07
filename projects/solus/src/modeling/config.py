# Solus-MoE 8B - Model Configuration
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolusConfig:
    name: str = "solus-moe-8b"
    architecture: str = "solus-moe-transformer"

    # Core dims
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 7

    # Dense-only FFN intermediate
    intermediate_size: int = 12800

    # MoE-only (None = disable MoE)
    expert_intermediate_size: Optional[int] = None
    moe_num_experts: int = 8
    moe_top_k: int = 2

    # Position embeddings
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Normalization
    rms_norm_eps: float = 1e-5

    # Tokenizer
    vocab_size: int = 128256
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: Optional[int] = None

    # Attention
    attention_dropout: float = 0.0
    is_causal: bool = True

    # Initialization
    initializer_range: float = 0.02

    # MoE layer indices (0-indexed); 8 of 32, every 4th starting at 0
    moe_layer_indices: Optional[list[int]] = field(
        default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28]
    )

    # MoE router hyperparameters
    moe_router_jitter_eps: float = 0.01   # noise added to gate logits during training
    moe_aux_loss_alpha: float = 0.01       # weight for load-balance auxiliary loss

    # Derived
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def pair_dim(self) -> int:
        return self.head_dim // 2

    @property
    def moe_enabled(self) -> bool:
        return (self.expert_intermediate_size is not None
                and self.expert_intermediate_size > 0)

    @property
    def params_m(self) -> float:
        D = self.head_dim
        H = self.hidden_size
        V = self.vocab_size
        attn = (self.num_attention_heads + 2 * self.num_key_value_heads) * H * D

        if not self.moe_enabled:
            # Dense decoder: 2 × RMSNorm + attn + mlp
            ffn        = 3 * H * self.intermediate_size          # gate/up/down
            norm_dense = 2 * H                                    # pre_attn + post_attn
            per        = attn + ffn + norm_dense
            return float(V * H + self.num_hidden_layers * per) / 1e6

        moe_n   = len(self.moe_layer_indices)
        dense_n = self.num_hidden_layers - moe_n

        ffn_d     = 3 * H * self.intermediate_size             # gate/up/down (dense FFN)
        norm_dense = 2 * H                                      # 2 × RMSNorm per dense layer
        per_dt     = attn + ffn_d + norm_dense

        ex_per     = 3 * H * self.expert_intermediate_size     # 3 linear per expert
        expert_pool = self.moe_num_experts * ex_per             # num_experts × params_per_expert
        router      = H * self.moe_num_experts                  # gating linear
        norm_moe    = 2 * H                                     # 2 × RMSNorm per MoE layer
        per_moe     = attn + expert_pool + router + norm_moe

        return float(V * H + dense_n * per_dt + moe_n * per_moe) / 1e6

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> "SolusConfig":
        import json, os
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            return cls(**json.load(f))

    def to_pretrained(self, checkpoint_dir: str) -> None:
        import json, os
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)


def solus_moe_8b() -> SolusConfig:
    """Solus-MoE 8B - 8,025M parameters, 8 experts x inter=3072."""
    return SolusConfig(
        name="solus-moe-8b",
        architecture="solus-moe-transformer",
        expert_intermediate_size=3072,
    )


def solus_7b_dense() -> SolusConfig:
    """Solus-Dense 7B - 6,868M parameters."""
    return SolusConfig(
        name="solus-7b",
        architecture="dense-transformer",
        expert_intermediate_size=None,
    )
