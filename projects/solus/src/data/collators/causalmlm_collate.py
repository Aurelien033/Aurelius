"""Data collation — batches of token IDs with causal attention masking."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TokenizedDataset(Dataset):
    """
    In-memory token dataset for pre-training.
    Loads .npy shards produced by the data pipeline and sequences them to config.seq_len.
    No shuffling here — rely on DataLoader shuffle=True with num_workers > 0.
    """

    def __init__(self, shards: list[str], seq_len: int):
        import numpy as np
        self.seq_len = seq_len
        self.tokens: torch.Tensor | None = None
        for shard in shards:
            chunk = torch.from_numpy(np.load(shard, mmap_mode="r").astype("int64"))
            if self.tokens is None:
                self.tokens = chunk
            else:
                self.tokens = torch.cat([self.tokens, chunk])
        self.num_seqs = self.tokens.numel() // seq_len

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start:start + self.seq_len].clone()


class SFTDataset(Dataset):
    """
    JSONL instruction-tuning dataset.

    Required JSONL fields per line:
        {"prompt": str, "response": str, "system_prompt": str}
    """

    def __init__(self, jsonl_path: str, tokenizer, seq_len: int,
                 *, system_prompt: str = ""):
        import json
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.examples  = []

        with open(jsonl_path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append({
                    "system": ex.get("system_prompt", system_prompt),
                    "prompt": ex["prompt"],
                    "response": ex["response"],
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{ex['system']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{ex['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ex['response']}<|eot_id|>"
        )
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.seq_len)
        return torch.tensor(tokens, dtype=torch.long)


def causalmlm_collate(batch: list[torch.Tensor], pad_token_id: int = 0):
    """
    Left/right pad sequences to max(batch len), then flat strip to exactly max_len.
    Labels = inputs shifted right by 1 with -100 on pad tokens (ignored in CE loss).
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    B, S = padded.shape
    input_ids = padded
    labels     = padded.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1]  = -100          # last position shifted LOSE → -100
    attention_mask = (input_ids != pad_token_id).unsqueeze(1) * torch.triu(  # (B,1,S,S)
        torch.ones(S, S, device=input_ids.device, dtype=input_ids.dtype), diagonal=1
    )
    # set mask to 0 for pad tokens
    attention_mask = attention_mask & input_ids.ne(pad_token_id).unsqueeze(-1)
    # Dtype-upgrade to float for matmul down the road
    attention_mask = attention_mask.masked_fill(attention_mask.bool(), float(0.0))
    attention_mask = attention_mask.masked_fill(~attention_mask.bool(), float("-inf"))
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
