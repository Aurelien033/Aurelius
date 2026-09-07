"""Training loop helpers used by ``src.training.pretrain_trainer``."""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.optim import lr_scheduler


def pipeline(num_workers: int = 2):
    """
    Common data pipeline used by all loaders.

    Strategy: map ok → map not ok
    NotImplementedError: must explicitly pass a data root folder — we assume preprocessed .npy
    shards are on disk.
    """
    pass  # see dataset_builder.build_pretrain_dataloader


# ── LR schedule ───────────────────────────────────────────────────────────────

def build_lr_scheduler(
        optimizer,
        total_steps: int,
        *,
        warmup_steps: int = 2000,
        min_lr_frac: float = 0.1,
        schedule: str = "cosine",
):
    """Create a LambdaLR wrapper implementing cosine + linear warmup."""
    base_lr = optimizer.defaults["lr"]

    def scale_fn(cur_step: int):
        # linear warmup
        if cur_step < warmup_steps:
            return max(cur_step / warmup_steps, 1e-8)
        # cosine decay
        progress = (cur_step - warmup_steps) / (total_steps - warmup_steps)
        cos_t    = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cos_t

    return lr_scheduler.LambdaLR(optimizer, lambda step: scale_fn(step))
