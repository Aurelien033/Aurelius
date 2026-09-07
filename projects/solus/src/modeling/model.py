"""
Solus-7B — causal LM model
Eager implementation; no external deps beyond torch.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .config import SolusConfig
from .layers import SolusDecoderLayer
from .norm import RMSNorm


class Embeddings(nn.Module):
    def __init__(self, cfg: SolusConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.dropout      = nn.Dropout(0.05)          # mild regularizer during pretrain

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.dropout(self.embed_tokens(input_ids))


class SolusForCausalLM(nn.Module):
    """
    Complete Solus-7B model for causal language modeling.

    Training:
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out["loss"]   # mean cross-entropy, -100-ignored

    Inference:
        model.eval()
        with torch.no_grad():
            tokens = model.generate(input_ids, max_new_tokens=64)
    """

    def __init__(self, cfg: SolusConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens        = Embeddings(cfg)
        self.layers              = nn.ModuleList(
            [SolusDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm                = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head             = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.post_init()

    # ------------------------------------------------------------------
    def post_init(self):
        """Standard 0.02 truncated-normal init; RMSNorm weights start as ones."""
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> dict:
        """
        Parameters
        ----------
        input_ids : (B, S) int64 token IDs
        labels    : (B, S) int64 same as input_ids; -100 is ignored in loss

        Returns
        -------
        {"logits": (B, S, V)}
        or
        {"logits": (B, S, V), "loss": scalar}
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states   = self.norm(hidden_states)
        logits          = hidden_states @ self.embed_tokens.embed_tokens.weight.T

        out: dict = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: Tensor, *, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, repetition_penalty: float = 1.1,
                 stop_token_ids: Optional[list[int]] = None) -> Tensor:
        """
        Minimal autoregressive generation loop.
        For production serving, use the vLLM-based server (scripts/serving/serve_solus.py).
        """
        self.eval()
        eos_ids    = stop_token_ids or [self.cfg.eos_token_id]
        generated  = input_ids
        pad_token  = self.cfg.eos_token_id or 0   # placeholder; stop uses eos

        for _ in range(max_new_tokens):
            out    = self.forward(input_ids=generated)
            logits = out["logits"][:, -1, :]

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for seq_id in range(generated.shape[0]):
                    seen = generated[seq_id].unique()
                    logits[seq_id, seen] = logits[seq_id, seen] / repetition_penalty

            # Top-K
            if top_k > 0:
                kth = logits.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

            # Top-P (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cumprobs = sorted_logits.softmax(-1).cumsum(-1)
                nucleus  = cumprobs > top_p
                nucleus[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(nucleus, float("-inf"))
                logits = torch.empty_like(sorted_logits).scatter_(1, sorted_idx, sorted_logits)

            next_token = logits.argmax(dim=-1, keepdim=True)   # greedy
            generated  = torch.cat([generated, next_token], dim=1)

            if next_token.item() in eos_ids:
                break

        return generated
