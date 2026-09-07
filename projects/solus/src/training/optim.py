"""Solus-7B — optimizer + LR scheduler utilities."""
from __future__ import annotations
import math
from torch.optim import Optimizer


def build_optimizer(model, lr: float = 3.0e-4, weight_decay: float = 0.1,
                    betas=(0.9, 0.95), *, use_8bit: bool = True) -> Optimizer:
    """
    AdamW with cosine LR schedule.

    8-bit AdamW (bitsandbytes) saves VRAM without meaningful convergence quality loss.
    Fallback: standard torch.optim.AdamW if bitsandbytes not installed.
    """
    from torch.optim import AdamW
    optim_class = __import__("bitsandbytes.optim", fromlist=["AdamW8bit"]).AdamW8bit \
                  if use_8bit else AdamW

    params = [p for n, p in model.named_parameters() if p.requires_grad]
    return optim_class(
        params,
        lr=lr,
        betas=betas,
        eps=1e-8,
        weight_decay=weight_decay,
    )


def cosine_lr_scheduler(optimizer, total_steps: int, warmup_steps: int = 2000,
                        lr_min: float | None = None) -> Optimizer:
    """
    Cosine decay with linear warmup.

    Warmup:  0 → base LR  in `warmup_steps`
    Cosine:  base → lr_min in `total_steps - warmup_steps`
    lr_min defaults to 10% of base LR.
    """
    base_lr = optimizer.defaults["lr"]
    lr_min  = lr_min if lr_min is not None else base_lr * 0.1

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min / base_lr + 0.5 * (1.0 - lr_min / base_lr) * (1.0 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)
