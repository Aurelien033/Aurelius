"""PyTorch Dataset backed by numpy memmaps from prepare_training_data.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.amc_data import AMCTrainBatch


class AMCDataset(Dataset[AMCTrainBatch]):
    """Dataset reading train/eval memmap shards produced by the data prep script."""

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        if split not in ("train", "eval"):
            raise ValueError(f"unknown split {split!r}")
        self.split = split

        manifest_path = self.data_dir / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        split_manifest = self.manifest[split]
        self.n_sequences = int(split_manifest["n_sequences"])
        self.max_seq_len = int(split_manifest["max_seq_len"])

        input_path = Path(split_manifest["input_ids_path"])
        importance_path = Path(split_manifest["importance_path"])
        if not input_path.is_absolute():
            input_path = self.data_dir / input_path.name
        if not importance_path.is_absolute():
            importance_path = self.data_dir / importance_path.name

        self._input_ids = np.memmap(
            input_path,
            dtype=np.uint32,
            mode="r",
            shape=(self.n_sequences, self.max_seq_len),
        )
        self._importance = np.memmap(
            importance_path,
            dtype=np.float32,
            mode="r",
            shape=(self.n_sequences, self.max_seq_len),
        )

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> AMCTrainBatch:
        input_ids = torch.from_numpy(self._input_ids[idx].astype(np.int64))
        importance_labels = torch.from_numpy(self._importance[idx].copy())
        pad_id = 0
        target_ids = torch.cat([input_ids[1:], torch.tensor([pad_id], dtype=torch.long)])
        return AMCTrainBatch(
            input_ids=input_ids.unsqueeze(0),
            target_ids=target_ids.unsqueeze(0),
            importance_labels=importance_labels.unsqueeze(0),
            session_id=f"{self.split}-{idx}",
            step=idx,
            retrieved_embeddings=None,
        )


__all__ = ["AMCDataset"]
