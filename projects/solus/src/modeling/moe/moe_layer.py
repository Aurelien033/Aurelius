"""
Solus-MoE - Sparse MoE Layer

Combines SolusMoERouter + expert pool into one sparse transformer block.
Dispatch: hidden (B,T,H) -> router (B,T,E) -> top-K experts -> weighted sum -> output.

References: Shazeer et al. 2017, Zoph et al. 2022, Fedus et al. 2022.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import SolusConfig
from .router  import SolusMoERouter, RouterOutput
from .expert  import SolusExpert
from ..norm import RMSNorm


class SolusMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer.

    Forward pass
    ------------
      hidden : (B, T, H)
      -> RouterOutput (weights, expert_indices, topk_weights, aux_loss)
      -> expert compute per (token, expert_k) pair
      -> weighted recombine -> (B, T, H)  residual
    """

    def __init__(self, cfg: SolusConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx  = layer_idx
        self.cfg        = cfg
        self.num_experts = cfg.moe_num_experts
        self.top_k      = cfg.moe_top_k
        self.hidden_size = cfg.hidden_size

        # Router produces gate probabilities and dispatches tokens to experts
        self.router = SolusMoERouter(cfg)

        # Expert pool -- all experts are identical in architecture, different init
        self.experts = nn.ModuleList([
            SolusExpert(cfg.hidden_size, cfg.expert_intermediate_size)
            for _ in range(self.num_experts)
        ])

    # ---------------------------------------------------------------
    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        hidden_states : (B, seq_len, hidden_size)

        Returns
        -------
        (output, aux_loss) both Tensors:
          output  : (B, seq_len, hidden_size)  -- sparse MoE output
          aux_loss: () scalar or None          -- routing load-balance loss
        """
        B, T, H = hidden_states.shape

        # Get routing decision per token
        router_out = self.router(hidden_states)
        topk_w     = router_out.topk_weights   # (B, T, K)
        expert_idx = router_out.expert_indices # (B, T, K) int64
        aux_loss   = router_out.aux_loss

        # --- Expert dispatch -----------------------------------------------
        # Build per-(token, k) hidden vectors  [N*K, H]  where  N = B*T
        N        = B * T
        flat_h   = hidden_states.reshape(N, H)        # (N, H)
        flat_idx = expert_idx.reshape(N, self.top_k)  # (N, K)  int
        flat_w   = topk_w.reshape(N, self.top_k)      # (N, K)

        K  = self.top_k
        # offset: token 0, 1, ..., B*T-1 each repeated K times for offset addition
        # Position within flat array:  i*K + k   for i-th token, k-th expert (0<=k<K)
        pos_base = torch.arange(N, device=hidden_states.device).unsqueeze(1) * K  # (B*T, 1)
        pos_k    = torch.arange(K, device=hidden_states.device).unsqueeze(0)       # (1, K)
        flat_pos = (pos_base + pos_k).reshape(-1)                                  # (N*K,)

        # Gather hidden vectors: for each of N*K slots, pick the hidden vector
        # of the token that owns that slot
        # The i-th token's hidden repeats K times (K different experts)
        repeated_h = flat_h.index_select(0, flat_pos // K)   # (N*K, H)

        # Per-expert output: expert processes its assigned tokens
        # Sort by expert index so that expert k gets contiguous runs
        expert_ids_flat = flat_idx.reshape(-1)  # (N*K,)
        sort_idx = torch.argsort(expert_ids_flat, stable=True)        # (N*K,)
        expert_ids_sorted = expert_ids_flat[sort_idx]

        # Dispatch hidden vectors to correct expert
        # gather: expert_out[i] for i-th expert
        # Run each expert on its token batch
        # We can just loop in eager mode (N*K / E tokens per expert, small overhead for E<=8)
        # expert_outputs: (N*K, H) reordered back into token order
        expert_output = torch.empty_like(repeated_h)    # (N*K, H)
        
        for e_idx in range(self.num_experts):
            mask = expert_ids_sorted == e_idx           # tokens assigned to expert e_idx
            if mask.any():
                token_indices = sort_idx[mask]          # positions in (N*K,) space
                expert_output[token_indices] = self.experts[e_idx](
                    repeated_h[token_indices]
                )
            else:
                # Zero out unassigned positions (shouldn't happen with top-k)
                pass

        # Sort expert outputs back to original (token, k) order
        inv_sort = torch.argsort(sort_idx, stable=True)
        expert_output = expert_output[inv_sort]   # (N*K, H)  in (token, k) order

        # Weighted combination:  y[token i] = topk_w[i,k] * expert_out[i,k]
        w_flat    = flat_w.reshape(-1, 1)        # (N*K, 1)
        weighted  = expert_output * w_flat        # (N*K, H)
        combined  = weighted.reshape(N, K, H)     # (N, K, H)
        output    = combined.sum(dim=1)           # (N, H)
        output    = output.reshape(B, T, H)       # (B, T, H)

        return output, aux_loss
