"""AMCTrainer — optimizer groups, schedulers, and training orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.model.amc_transformer import AMCModelOutput
from src.training.amc_data import AMCTrainBatch
from src.training.amc_losses import (
    memory_consistency_loss,
    promotion_reward_loss,
    surprise_prediction_loss,
    total_amc_loss,
)


@dataclass
class AMCTrainConfig:
    model_name: str = "amc-forge-1b"
    batch_size: int = 64
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    promotion_gate_lr: float = 1e-4
    surprise_head_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50_000
    eval_every: int = 500
    checkpoint_every: int = 2000
    grad_clip: float = 1.0
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sft": 0.70,
            "surprise": 0.15,
            "consistency": 0.10,
            "promotion": 0.05,
        }
    )
    precision: str = "bf16"


class AMCModelProtocol(Protocol):
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def named_parameters(self) -> Any: ...
    def parameters(self) -> Any: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        session_id: str | list[str] | None = None,
        step: int | list[int] = 0,
        use_amc: bool = True,
        return_memory: bool = False,
    ) -> AMCModelOutput: ...


def _batch_step(batch: AMCTrainBatch) -> int:
    if isinstance(batch.step, list):
        return int(batch.step[0]) if batch.step else 0
    return int(batch.step)


def _batch_session_id(batch: AMCTrainBatch) -> str | None:
    if isinstance(batch.session_id, list):
        return str(batch.session_id[0]) if batch.session_id else None
    return batch.session_id


class JSONLLogger:
    """Append-only JSONL metrics log."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kwargs: Any) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(kwargs, default=str) + "\n")


def build_optimizer_groups(
    model: nn.Module,
    *,
    lr_main: float,
    lr_gate: float,
    lr_surprise: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
) -> tuple[dict[str, AdamW], dict[str, SequentialLR]]:
    """Build three separate optimizers and cosine schedulers."""
    main_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []
    surprise_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "promotion_gate" in name:
            gate_params.append(param)
        elif "surprise_head" in name:
            surprise_params.append(param)
        else:
            main_params.append(param)

    def _optimizer(params: list[nn.Parameter], lr: float) -> AdamW:
        if params:
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        dummy = torch.zeros(1, requires_grad=True)
        return AdamW([dummy], lr=lr, weight_decay=0.0)

    optimizers = {
        "main": _optimizer(main_params, lr_main),
        "gate": _optimizer(gate_params, lr_gate),
        "surprise": _optimizer(surprise_params, lr_surprise),
    }

    def make_scheduler(optimizer: AdamW, lr: float) -> SequentialLR:
        start_factor = 1e-7 / max(lr, 1e-7)
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_steps - warmup_steps),
            eta_min=1e-7,
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    schedulers = {
        "main": make_scheduler(optimizers["main"], lr_main),
        "gate": make_scheduler(optimizers["gate"], lr_gate),
        "surprise": make_scheduler(optimizers["surprise"], lr_surprise),
    }
    return optimizers, schedulers


class CheckpointManager:
    """Save/load training checkpoints with simple rotation."""

    def __init__(
        self,
        save_dir: Path | str,
        *,
        keep_last_n: int = 5,
        keep_best: bool = True,
        best_metric: str = "eval_surprise_accuracy",
        best_mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self._best_value = float("-inf") if best_mode == "max" else float("inf")
        self._saved: list[Path] = []

    def save(
        self,
        tag: str,
        *,
        model: nn.Module,
        optimizers: dict[str, AdamW],
        schedulers: dict[str, SequentialLR],
        step: int,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "model": model.state_dict(),
            "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
            "schedulers": {key: sched.state_dict() for key, sched in schedulers.items()},
            "step": step,
            "metrics": metrics or {},
        }
        path = self.save_dir / f"checkpoint-{tag}.pt"
        torch.save(payload, path)
        self._saved.append(path)
        while len(self._saved) > self.keep_last_n:
            oldest = self._saved.pop(0)
            if oldest.exists() and "best" not in oldest.name:
                oldest.unlink()
        if self.keep_best and metrics and self.best_metric in metrics:
            value = float(metrics[self.best_metric])
            if self.best_mode == "max":
                is_better = value > self._best_value
            else:
                is_better = value < self._best_value
            if is_better:
                self._best_value = value
                torch.save(payload, self.save_dir / "checkpoint-best.pt")
        return path

    def load(self, path: Path | str) -> dict[str, Any]:
        return torch.load(Path(path), map_location="cpu", weights_only=True)


class AMCTrainer:
    """Full training loop for AMC-aware models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterator[AMCTrainBatch] | Any,
        eval_loader: Iterator[AMCTrainBatch] | Any | None,
        config: AMCTrainConfig,
        log_dir: Path | str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self._micro_step = 0

        self.optimizers, self.schedulers = build_optimizer_groups(
            model,
            lr_main=config.learning_rate,
            lr_gate=config.promotion_gate_lr,
            lr_surprise=config.surprise_head_lr,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        self.main_opt = self.optimizers["main"]
        self.gate_opt = self.optimizers["gate"]
        self.surprise_opt = self.optimizers["surprise"]
        self.scheduler = self.schedulers["main"]

        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        self.logger = JSONLLogger(self.log_dir / "training.jsonl")
        self.checkpoint_manager = CheckpointManager(self.log_dir / "checkpoints")

    def train_step(self, batch: AMCTrainBatch) -> dict[str, float]:
        self.model.train()
        output = self.model(
            batch.input_ids,
            session_id=_batch_session_id(batch),
            step=_batch_step(batch),
            use_amc=True,
            return_memory=True,
        )

        vocab_size = output.logits.shape[-1]
        sft = F.cross_entropy(
            output.logits.reshape(-1, vocab_size),
            batch.target_ids.reshape(-1),
        )

        surprise_scores = output.surprise_scores
        if surprise_scores is None:
            surprise_scores = torch.zeros_like(batch.importance_labels, dtype=torch.float32)
        sup = surprise_prediction_loss(surprise_scores, batch.importance_labels)

        hidden = output.hidden_states
        if hidden is None:
            hidden = output.logits  # fallback for lightweight mocks
        con = memory_consistency_loss(hidden, batch.retrieved_embeddings)

        if output.promotion_loss is not None:
            pro = output.promotion_loss
        elif output.gate_outputs:
            soft = torch.stack([gate[1].mean() for gate in output.gate_outputs])
            rewards = batch.importance_labels.mean(dim=-1).detach()
            pro = promotion_reward_loss(soft, rewards)
        else:
            pro = torch.zeros((), device=sft.device, dtype=sft.dtype)

        weights = self.config.loss_weights
        total, _loss_metrics = total_amc_loss(
            sft,
            sup,
            con,
            pro,
            alpha=weights["sft"],
            beta=weights["surprise"],
            gamma=weights["consistency"],
            delta=weights["promotion"],
        )
        scaled_total = total / max(1, self.config.gradient_accumulation)
        scaled_total.backward()

        self._micro_step += 1
        stepped = False
        if self._micro_step % self.config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers.values():
                scheduler.step()
            stepped = True

        promotion_rate = 0.0
        if output.gate_outputs:
            promotion_rate = float(
                torch.stack([gate[0].mean() for gate in output.gate_outputs]).mean().item()
            )

        tier2_entries = 0
        tier2_hook = getattr(self.model, "_tier2_hook", None)
        if tier2_hook is not None and hasattr(tier2_hook, "episodic"):
            tier2_entries = len(tier2_hook.episodic)

        metrics = {
            "sft_loss": float(sft.detach().item()),
            "surprise_loss": float(sup.detach().item()),
            "consistency_loss": float(con.detach().item()),
            "promotion_loss": float(pro.detach().item()),
            "total_loss": float(total.detach().item()),
            "lr": float(self.main_opt.param_groups[0]["lr"]),
            "promotion_rate": promotion_rate,
            "mean_surprise": float(surprise_scores.detach().mean().item()),
            "tier2_entries": float(tier2_entries),
            "optimizer_stepped": float(stepped),
        }
        self.logger.log(step=self.step, **metrics)
        return metrics

    def train(self) -> None:
        for step, batch in enumerate(self.train_loader):
            self.train_step(batch)
            self.step = step
            if self.eval_loader is not None and step > 0 and step % self.config.eval_every == 0:
                eval_metrics = self.evaluate()
                self.logger.log(step=step, phase="eval", **eval_metrics)
            if step > 0 and step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)
            if step >= self.config.max_steps:
                break
        self.save_checkpoint("final")

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.eval_loader is None:
            return {"eval_sft_loss": 0.0, "eval_surprise_accuracy": 0.0}

        self.model.eval()
        losses: list[float] = []
        sup_accs: list[float] = []
        for batch in self.eval_loader:
            output = self.model(batch.input_ids, use_amc=True, return_memory=True)
            vocab_size = output.logits.shape[-1]
            sft = F.cross_entropy(
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

        return {
            "eval_sft_loss": sum(losses) / max(len(losses), 1),
            "eval_surprise_accuracy": sum(sup_accs) / max(len(sup_accs), 1),
        }

    def save_checkpoint(self, tag: str | int) -> Path:
        return self.checkpoint_manager.save(
            str(tag),
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            step=self.step,
            metrics={"step": self.step},
        )


__all__ = [
    "AMCTrainConfig",
    "AMCTrainer",
    "CheckpointManager",
    "JSONLLogger",
    "build_optimizer_groups",
]
