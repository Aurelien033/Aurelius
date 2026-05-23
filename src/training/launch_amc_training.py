#!/usr/bin/env python3
"""Launch AMC training from a forge YAML config and memmap dataset.

Usage:
    deepspeed --num_gpus 4 src/training/launch_amc_training.py \\
        --config configs/amc_forge_1b.yaml \\
        --data data/tokenized/amc_forge_1b \\
        --log_dir logs/forge_1b_run_001 \\
        --deepspeed configs/deepspeed_zero2.json

    # Single-GPU / CPU sanity check:
    python src/training/launch_amc_training.py \\
        --config configs/amc_forge_1b.yaml \\
        --data data/tokenized/test_run \\
        --log_dir logs/sanity_run \\
        --max-steps 10 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.memory.amc_tier2 import AMCTier2Hook  # noqa: E402
from src.memory.amc_tier3 import AMCTier3Hook  # noqa: E402
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig  # noqa: E402
from src.training.amc_data import AMCDataCollator, AMCTrainBatch  # noqa: E402
from src.training.amc_dataset import AMCDataset  # noqa: E402
from src.training.amc_trainer import AMCTrainConfig, AMCTrainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch AMC training")
    parser.add_argument("--config", required=True, help="YAML model + training config")
    parser.add_argument("--data", required=True, help="Tokenized data directory (manifest.json)")
    parser.add_argument("--log_dir", required=True, help="Directory for logs and checkpoints")
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed JSON config (optional)")
    parser.add_argument("--resume-from", default=None, help="Checkpoint .pt to resume")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps (sanity runs)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--skip-config-validation", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by DeepSpeed launcher")
    parser.add_argument("--deepspeed_config", default=None, help="Alias for --deepspeed")
    return parser.parse_args(argv)


def log(msg: str, *, step: int | None = None, rank: int = 0) -> None:
    if rank != 0:
        return
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}]"
    if step is not None:
        prefix += f" [step {step:>6}]"
    print(f"{prefix} {msg}", flush=True)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_config_from_yaml(raw: dict[str, Any]) -> AMCTransformerConfig:
    model_raw = dict(raw["model"])
    model_raw.pop("name", None)
    return AMCTransformerConfig(**model_raw)


def train_config_from_yaml(
    raw: dict[str, Any],
    *,
    max_steps: int | None = None,
    batch_size: int | None = None,
) -> AMCTrainConfig:
    train_raw = raw.get("training", {})
    amc_weights = raw.get("amc", {}).get("loss_weights") or train_raw.get("loss_weights", {})
    return AMCTrainConfig(
        model_name=str(raw.get("model", {}).get("name", "amc-forge")),
        batch_size=batch_size or int(train_raw.get("batch_size", 16)),
        gradient_accumulation=int(train_raw.get("gradient_accumulation", 1)),
        learning_rate=float(train_raw.get("learning_rate", 3e-4)),
        promotion_gate_lr=float(train_raw.get("promotion_gate_lr", 1e-4)),
        surprise_head_lr=float(train_raw.get("surprise_head_lr", 1e-5)),
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        warmup_steps=int(train_raw.get("warmup_steps", 1000)),
        max_steps=max_steps or int(train_raw.get("max_steps", 50_000)),
        eval_every=int(train_raw.get("eval_every", 500)),
        checkpoint_every=int(train_raw.get("checkpoint_every", 2000)),
        grad_clip=float(train_raw.get("grad_clip", 1.0)),
        loss_weights={
            "sft": float(amc_weights.get("sft", 0.70)),
            "surprise": float(amc_weights.get("surprise", 0.15)),
            "consistency": float(amc_weights.get("consistency", 0.10)),
            "promotion": float(amc_weights.get("promotion", 0.05)),
        },
        precision=str(raw.get("compute", {}).get("precision", "bf16")),
    )


def validate_data_dir(data_dir: Path | str) -> list[str]:
    root = Path(data_dir)
    errors: list[str] = []
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return [f"missing manifest: {manifest}"]
    with manifest.open(encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    for split in ("train", "eval"):
        if split not in manifest_data:
            errors.append(f"manifest missing split {split!r}")
            continue
        split_info = manifest_data[split]
        for key in ("input_ids_path", "importance_path", "n_sequences", "max_seq_len"):
            if key not in split_info:
                errors.append(f"manifest[{split}] missing {key!r}")
        input_path = Path(split_info.get("input_ids_path", ""))
        if not input_path.is_absolute():
            input_path = root / input_path.name
        if split_info.get("n_sequences", 0) > 0 and not input_path.is_file():
            errors.append(f"{split} memmap missing: {input_path}")
    return errors


class _DeviceLoader:
    """Move collated batches to the training device."""

    def __init__(self, loader: DataLoader[AMCTrainBatch], device: torch.device) -> None:
        self._loader = loader
        self._device = device

    def __iter__(self) -> Iterator[AMCTrainBatch]:
        for batch in self._loader:
            yield batch.to(self._device)


class _DeviceTrainer(AMCTrainer):
    """AMCTrainer with device placement and optional BF16 autocast."""

    def __init__(
        self,
        *args: Any,
        device: torch.device,
        use_bf16: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._device = device
        self._use_bf16 = use_bf16 and device.type == "cuda"
        self.model.to(device)

    def train_step(self, batch: AMCTrainBatch) -> dict[str, float]:
        batch = batch.to(self._device)
        if self._use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return super().train_step(batch)
        return super().train_step(batch)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.eval_loader is None:
            return {"eval_sft_loss": 0.0, "eval_surprise_accuracy": 0.0}
        self.model.eval()
        losses: list[float] = []
        sup_accs: list[float] = []
        for batch in self.eval_loader:
            batch = batch.to(self._device)
            if self._use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.model(batch.input_ids, use_amc=True, return_memory=True)
            else:
                output = self.model(batch.input_ids, use_amc=True, return_memory=True)
            vocab_size = output.logits.shape[-1]
            sft = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, vocab_size),
                batch.target_ids.reshape(-1),
            )
            losses.append(float(sft.item()))
            if output.surprise_scores is not None:
                pred = (output.surprise_scores > 0.5).float()
                target = batch.importance_labels
                if pred.shape != target.shape and pred.dim() == 3:
                    pred = pred.mean(dim=0)
                min_len = min(pred.shape[-1], target.shape[-1])
                acc = (pred[..., :min_len] == target[..., :min_len]).float().mean().item()
                sup_accs.append(acc)
        self.model.train()
        return {
            "eval_sft_loss": sum(losses) / max(len(losses), 1),
            "eval_surprise_accuracy": sum(sup_accs) / max(len(sup_accs), 1),
        }


def build_dataloaders(
    data_dir: Path | str,
    train_cfg: AMCTrainConfig,
    *,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader[AMCTrainBatch], DataLoader[AMCTrainBatch]]:
    collator = AMCDataCollator()
    train_dataset = AMCDataset(data_dir, split="train")
    eval_dataset = AMCDataset(data_dir, split="eval")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size or max(1, train_cfg.batch_size // 2),
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    return train_loader, eval_loader


def run_training(args: argparse.Namespace) -> int:
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        rank = args.local_rank
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    ds_config_path = args.deepspeed or args.deepspeed_config
    if ds_config_path and world_size > 1:
        log(
            "DeepSpeed multi-GPU path is not integrated with AMCTrainer's three optimizers; "
            "running single-process training on this rank.",
            rank=rank,
        )

    log(f"rank={rank} world_size={world_size} device={device}", rank=rank)

    config_path = Path(args.config)
    raw_config = load_yaml_config(config_path)

    if not args.skip_config_validation and config_path.name == "amc_forge_1b.yaml":
        from scripts.validate_amc_forge_config import validate_amc_forge_config

        config_errors = validate_amc_forge_config(raw_config, repo_root=_REPO_ROOT)
        if config_errors:
            for err in config_errors:
                print(f"config error: {err}", file=sys.stderr)
            return 1

    data_errors = validate_data_dir(args.data)
    if data_errors:
        for err in data_errors:
            print(f"data error: {err}", file=sys.stderr)
        return 1

    model_cfg = model_config_from_yaml(raw_config)
    model = AMCTransformer(model_cfg)
    tier2 = AMCTier2Hook()
    tier3 = AMCTier3Hook()
    model.wire_amc_hooks(tier2, tier3)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"model params: {param_count:,}", rank=rank)

    train_cfg = train_config_from_yaml(
        raw_config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    eval_batch_size = int(raw_config.get("evaluation", {}).get("eval_batch_size", 8))
    train_loader, eval_loader = build_dataloaders(
        args.data,
        train_cfg,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
    )

    log_dir = Path(args.log_dir)
    trainer = _DeviceTrainer(
        model,
        _DeviceLoader(train_loader, device),
        _DeviceLoader(eval_loader, device),
        train_cfg,
        log_dir,
        device=device,
        use_bf16=train_cfg.precision == "bf16",
    )

    if args.resume_from:
        ckpt = trainer.checkpoint_manager.load(args.resume_from)
        model.load_state_dict(ckpt["model"])
        for name, state in ckpt.get("optimizers", {}).items():
            if name in trainer.optimizers:
                trainer.optimizers[name].load_state_dict(state)
        for name, state in ckpt.get("schedulers", {}).items():
            if name in trainer.schedulers:
                trainer.schedulers[name].load_state_dict(state)
        trainer.step = int(ckpt.get("step", 0))
        log(f"resumed from {args.resume_from} at step {trainer.step}", rank=rank)

    log(f"starting training: max_steps={train_cfg.max_steps}", rank=rank)
    trainer.train()
    log(f"training complete; logs in {log_dir}", rank=rank)
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_training(args))


if __name__ == "__main__":
    main()
