"""Accelerate-based trainer for Solus-7B pretraining / SFT."""
from __future__ import annotations
from pathlib import Path
import json, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.modeling.model import SolusForCausalLM
from src.modeling.config import SolusConfig
from src.training.optim import build_optimizer, cosine_lr_scheduler


# Cuda-friendly types for bf16
DTYPE_STRIDE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def train_solus(config_path: str, output_dir: str,
                *, resume_from: str | None = None, quick: bool = False):
    """
    Main training entry point (called from scripts/pretrain/train.py).

    Parameters
    ----------
    config_path  : YAML config file
    output_dir   : directory to write checkpoints + logs
    resume_from  : path to an existing checkpoint to resume training from
    quick        : if True, dummy-data smoke test (run < 10 steps and exit)
    """
    # Parse config
    from yaml import safe_load
    with open(config_path) as f:
        config = safe_load(f)

    model_cfg = SolusConfig(**config["model"])
    set_seed(42)

    # ── Accelerator ──────────────────────────────────────────────────────────
    accelerator = Accelerator(
        logging_dir=os.path.join(output_dir, "logs"),
        mixed_precision=config.get("training", {}).get("dtype", "bf16"),
        dynamic_loss_scale=True,
    )
    accelerator.print(f"  Devices : {accelerator.num_processes}× {accelerator.device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    dtype = DTYPE_STRIDE.get(config.get("training", {}).get("dtype", "bf16"), torch.bfloat16)
    model = SolusForCausalLM(model_cfg).to(dtype=dtype)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_cfg = config.get("pretrain", config.get("sft", {}))
    seq_len  = train_cfg.get("seq_len", 2048)
    batch_size = train_cfg.get("global_batch_size", 256)
    micro_batch = batch_size // accelerator.num_processes

    from src.data.dataset_builder import build_pretrain_dataloader
    data_loader = build_pretrain_dataloader(
        data_root=os.path.join(os.path.dirname(config_path), "..", "data", "pretrain"),
        seq_len=seq_len,
        batch_size=micro_batch,
        num_workers=4,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", {}).get("base", 3e-4)))
    optim = build_optimizer(model, lr=lr, weight_decay=0.1)
    total_steps = train_cfg.get("total_steps", int(train_cfg.get("total_tokens", 1e11) / 4192))
    lr_sched    = cosine_lr_scheduler(optim, total_steps=total_steps,
                                      warmup_steps=train_cfg.get("warmup_steps", 2000))

    # ── W&B ───────────────────────────────────────────────────────────────────
    if accelerator.is_main_process and not quick:
        os.environ.setdefault("WANDB_API_KEY", "")  # user must set in env
        import wandb
        wandb.init(
            project=config.get("logging", {}).get("project", "solus"),
            name=config.get("logging", {}).get("experiment_name", "pretrain"),
            config=config,
            dir=output_dir,
        )

    # ── Prepare with Accelerate ───────────────────────────────────────────────
    model, optim, data_loader, lr_sched = accelerator.prepare(
        model, optim, data_loader, lr_sched,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if resume_from:
        accelerator.print(f"  ← Resuming from {resume_from}")
        # accelerator.load_state handles all distributed shard resumption automatically
        # W&B will resume automatically and continue logging

    # ── Training loop ─────────────────────────────────────────────────────────
    max_steps = 10 if quick else total_steps
    log_every = train_cfg.get("log_every_steps", 10)
    save_every = train_cfg.get("save_every_steps", 1000)
    save_dir = Path(output_dir) / "checkpoints"

    for step in range(start_step, max_steps):
        model.train()
        batch = next(iter(data_loader))
        input_ids = batch["input_ids"].to(accelerator.device)
        labels    = batch["labels"].to(accelerator.device)
        # attention_mask dict key not always present in all collate versions → skip
        with accelerator.accumulate(model):
            loss = model(input_ids=input_ids, labels=labels)["loss"] / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            lr_sched.step()
            optim.zero_grad(set_to_none=True)

        step_loss = accelerator.gather(loss).mean().item()

        # Incremented *after* optimizer step, so counter should be dynamic
        effective_step = step
        if effective_step % log_every == 0 and accelerator.is_main_process:
            accelerator.print(
                f"  [{effective_step:7d}/{max_steps:7d}]  loss={step_loss:.4f}  lr={lr_sched.get_last_lr()[0]:.2e}")
            # wandb.log({"train/loss": step_loss, "train/lr": lr_sched.get_last_lr()[0],
            #            "train/step": effective_step}, step=effective_step)

        if accelerator.is_main_process and effective_step > 0 and effective_step % save_every == 0:
            accelerator.save_state(str(save_dir / f"step_{effective_step:07d}"))
            accelerator.print(f"  ✓ checkpoint step_{effective_step:07d}")

    accelerator.print("  Training complete ✓")
    return model
