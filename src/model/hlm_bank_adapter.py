"""HLM Preference Bank Adapter — differentiable adapter that reads from the bank.

Mechanism:
    query_t = normalize(query_proj(LN(hidden_t)))         # d_model -> bank_dim
    alpha_t = sigmoid(gate_mlp(LN(hidden_t)))             # d_model -> 1
    bank_context_t = bank.read(query_t).context           # bank_dim
    bias_t = out_proj(bank_context_t)                     # bank_dim -> d_model
    hidden'_t = hidden_t + inject_scale * alpha_t * bias_t

Invariants:
- If bank is None or empty, hidden' == hidden exactly.
- alpha_t in [0, 1], shape (B, T, 1).
- Gradients flow into adapter parameters, not bank buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.memory.hlm_bank import HLMPreferenceBank


@dataclass(frozen=True)
class HLMPreferenceAdapterConfig:
    d_model: int
    bank_dim: int = 64
    gate_hidden: int = 0  # 0 = linear gate only
    inject_scale: float = 0.1
    top_k: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.inject_scale <= 1.0:
            raise ValueError(f"inject_scale must be in [0, 1], got {self.inject_scale}")


@dataclass(frozen=True)
class HLMPreferenceAdapterOutput:
    hidden: torch.Tensor     # (B, T, d_model)
    alpha: torch.Tensor      # (B, T, 1)
    bank_context: torch.Tensor  # (B, T, bank_dim)
    confidence: torch.Tensor   # (B, T, 1)


class HLMPreferenceAdapter(nn.Module):
    def __init__(self, config: HLMPreferenceAdapterConfig) -> None:
        super().__init__()
        self.cfg = config
        self.norm = nn.LayerNorm(config.d_model, eps=1e-6)

        # Query projection: d_model -> bank_dim
        self.query_proj = nn.Linear(config.d_model, config.bank_dim, bias=False)

        # Gate: d_model -> hidden -> 1
        if config.gate_hidden > 0:
            self.gate_mlp = nn.Sequential(
                nn.Linear(config.d_model, config.gate_hidden),
                nn.GELU(),
                nn.Linear(config.gate_hidden, 1),
            )
        else:
            self.gate_mlp = nn.Linear(config.d_model, 1)

        # Output projection: bank_dim -> d_model
        self.out_proj = nn.Linear(config.bank_dim, config.d_model, bias=False)

        # Initialize out_proj small so bank influence is muted at init
        nn.init.kaiming_uniform_(self.out_proj.weight, a=0.0)

    def forward(
        self,
        hidden: torch.Tensor,
        bank: HLMPreferenceBank | None,
        *,
        read_only: bool = True,
    ) -> HLMPreferenceAdapterOutput:
        """Apply bank-biased residual to hidden states.

        Args:
            hidden: (B, T, d_model)
            bank: optional HLMPreferenceBank; if None or empty, returns identity.
            read_only: unused in MVP (kept for write-mode extension).

        Returns:
            HLMPreferenceAdapterOutput with modified hidden, gate alpha,
            bank context, and confidence.
        """
        B, T, D = hidden.shape

        # Short-circuit if no bank or empty
        if bank is None or bank.is_empty():
            alpha = torch.zeros(B, T, 1, device=hidden.device, dtype=hidden.dtype)
            ctx = torch.zeros(B, T, self.cfg.bank_dim, device=hidden.device, dtype=hidden.dtype)
            conf = torch.zeros(B, T, 1, device=hidden.device, dtype=hidden.dtype)
            return HLMPreferenceAdapterOutput(
                hidden=hidden,
                alpha=alpha,
                bank_context=ctx,
                confidence=conf,
            )

        normed = self.norm(hidden)  # (B, T, d_model)

        # Query projection + normalize
        query = self.query_proj(normed)  # (B, T, bank_dim)
        query = query / (query.norm(dim=-1, keepdim=True) + self.cfg.bank_dim**-0.5)

        # Gate alpha in [0, 1]
        alpha = torch.sigmoid(self.gate_mlp(normed))  # (B, T, 1)

        # Bank read — don't detach context so gradients flow into query_proj
        read_result = bank.read(query, top_k=self.cfg.top_k)
        # read_result.context: (B, T, bank_dim)
        ctx = read_result.context
        conf = read_result.confidence.detach()  # (B, T, 1)

        # Bias projection: bank_dim -> d_model
        bias = self.out_proj(ctx)  # (B, T, d_model)

        # Residual injection
        hidden_out = hidden + self.cfg.inject_scale * alpha * bias

        return HLMPreferenceAdapterOutput(
            hidden=hidden_out,
            alpha=alpha,
            bank_context=ctx,
            confidence=conf,
        )
