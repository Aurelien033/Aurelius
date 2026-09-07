"""
Solus-MoE 8B -- Full Causal Language Model

Mixes 24 standard SolusDecoderLayer with 8 SolusSparseDecoderLayer.
Embeddings and LM head are tied (identical to Solus-7B).

Forward
-------
    out = moe_model(input_ids=input_ids, labels=input_ids)
    loss = out["loss"]

Generate
--------
    model.eval()
    tokens = model.generate(input_ids, max_new_tokens=32)
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..config import SolusConfig
from ..norm import RMSNorm
from ..layers import SolusDecoderLayer        # dense
from .sparse_decoder_layer import SolusSparseDecoderLayer  # sparse


class Embeddings(nn.Module):
    """Token embeddings + mild dropout regularizer."""
    def __init__(self, cfg: SolusConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_min)
        self.dropout      = nn.Dropout(0.05)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.dropout(self.embed_tokens(input_ids))


class SolusMoEForCausalLM(nn.Module):
    """
    Complete Solus-MoE 8B model for causal language modeling.

    Architecture: 24 Dense SolusDecoderLayer + 8 Sparse SolusSparseDecoderLayer
                   Tied token embeddings
    """

    def __init__(self, cfg: SolusConfig):
        super().__init__()
        self.cfg = cfg

        moe_indices   = set(cfg.moe_layer_indices) if cfg.moe_enabled else set()
        self.moe_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()

        for i in range(cfg.num_hidden_layers):
            if i in moe_indices:
                self.moe_layers.append(SolusSparseDecoderLayer(cfg, layer_idx=i))
            else:
                self.dense_layers.append(SolusDecoderLayer(cfg, layer_idx=i))

        # Shared embedding table (tied with LM head)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # Final norm and tied LM head
        self.norm     = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head  = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Tie LM head to embedding table — weight is a single Tensor, not Module
        self.lm_head.weight = self.embed_tokens.weight
        self._init_weights()

    def _init_weights(self):
        self.apply(self._init_fn)

    @staticmethod
    def _init_fn(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layers)

    @property
    def num_dense_layers(self) -> int:
        return len(self.dense_layers)

    # ---------------------------------------------------------------
    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> dict:
        """
        Parameters
        ----------
        input_ids : (B, S) int64  token IDs
        labels    : (B  S)  -- same as input_ids; -100 ignored in loss

        Returns
        -------
        {"logits": (B, S, V)}  or
        {"logits": (B, S, V), "loss": scalar}
        """
        hidden = self.embed_tokens(input_ids)

        # Interleave dense / sparse layers in correct order
        moe_idx = 0
        dense_idx = 0
        moe_indices_set = set(self.cfg.moe_layer_indices) if self.cfg.moe_enabled else set()

        for i in range(self.cfg.num_hidden_layers):
            if i in moe_indices_set:
                hidden, aux = self.moe_layers[moe_idx](hidden)
                moe_idx += 1
            else:
                hidden = self.dense_layers[dense_idx](hidden)
                dense_idx += 1

        hidden       = self.norm(hidden)
        logits       = hidden @ self.embed_tokens.weight.T

        out: dict = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out

    # ----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: Tensor,
                 *, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, repetition_penalty: float = 1.1,
                 stop_token_ids: Optional[list[int]] = None) -> Tensor:
        """Minimal autoregressive generation loop."""
        self.eval()
        eos_ids   = stop_token_ids or [self.cfg.eos_token_id]

        generated = input_ids

        for _ in range(max_new_tokens):
            out    = self.forward(input_ids=generated)
            logits = out["logits"][:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature
            if repetition_penalty != 1.0:
                for seq_id in range(generated.shape[0]):
                    seen = generated[seq_id].unique()
                    logits[seq_id, seen] = logits[seq_id, seen] / repetition_penalty
            if top_k > 0:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth,
                                     torch.full_like(logits, float("-inf")), logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(descending=True)
                cumprobs  = sorted_logits.softmax(-1).cumsum(-1)
                nucleus   = cumprobs > top_p; nucleus[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(nucleus, float("-inf"))
                logits = torch.zeros_like(sorted_logits).scatter_(1, sorted_indices, sorted_logits)

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated   = torch.cat([generated, next_token], dim=1)

            if next_token.item() in eos_ids:
                break

        return generated
