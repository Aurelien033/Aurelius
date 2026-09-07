"""Solus pretraining data pipeline — maps domain token shards on disk to a DataLoader."""
from __future__ import annotations
import glob, random, os, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.collators.causalmlm_collate import TokenizedDataset, causalmlm_collate


def build_pretrain_dataloader(
    data_root: str,
    split: str = "train",
    seq_len: int     = 2048,
    batch_size: int  = 256,
    tokenizer        = None,          # unused here — tokenize offline
    num_workers: int = 4,
    seed: int        = 42,
) -> DataLoader:
    """
    Parameters
    ----------
    data_root : Path to data/pretrain/ containing per-domain subdirectories.
                Each dir must contain .npy uint32 token arrays (pre-chunked).
    split     : 'train' or 'val' — corresponding shard subfolders
    seq_len   : training sequence length
    batch_size: global batch size (accepted for interface compatibility;
                actual micro-batch per GPU is divided by world size externally)

    Returns
    -------
    torch.utils.data.DataLoader  producing (input_ids, labels, attention_mask)
    """
    train_dir = Path(data_root) / split / "train"
    val_dir   = Path(data_root) / split / "val" or train_dir

    # Collect all shard paths
    shards = sorted(glob.glob(str(train_dir / "**/*.npy"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"No .npy shards found in {train_dir}")

    dataset = TokenizedDataset(shards=shards, seq_len=seq_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=causalmlm_collate,
    )
    return loader


def build_sft_dataloader(
    jsonl_path: str,
    tokenizer,
    seq_len: int      = 2048,
    batch_size: int   = 32,
    num_workers: int  = 2,
) -> DataLoader:
    """Build DataLoader for instruction-tuning (chat) JSONL data."""
    from src.data.collators.causalmlm_collate import SFTDataset, sft_collate_fn

    dataset = SFTDataset(jsonl_path=jsonl_path, tokenizer=tokenizer, seq_len=seq_len)  # type: ignore[name-defined]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=sft_collate_fn,
    )
    return loader
