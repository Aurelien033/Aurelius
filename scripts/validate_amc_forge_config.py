#!/usr/bin/env python3
"""Validate configs/amc_forge_1b.yaml (model + training + data + compute).

Usage:
  python scripts/validate_amc_forge_config.py
  python scripts/validate_amc_forge_config.py --config configs/amc_forge_1b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]

# T17 / T18 invariants
_REQUIRED_LRS = {
    "learning_rate": 3e-4,
    "promotion_gate_lr": 1e-4,
    "surprise_head_lr": 1e-5,
}
_PARAM_MIN = 800_000_000
_PARAM_MAX = 1_200_000_000
_LOSS_WEIGHT_TOLERANCE = 1e-6


def validate_amc_forge_config(raw: dict[str, Any], *, repo_root: Path | None = None) -> list[str]:
    """Return validation errors (empty list means pass)."""
    errors: list[str] = []
    root = repo_root or _REPO_ROOT

    if "model" not in raw:
        return ["missing top-level 'model' section"]

    model_raw = raw["model"]
    if not isinstance(model_raw, dict):
        return ["'model' section must be a mapping"]

    if "ssm_d_conv" in model_raw and "d_conv" not in model_raw:
        errors.append("use 'd_conv' (not 'ssm_d_conv') for AMCTransformerConfig")

    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.count_params import validate_config
        from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig

        cfg = AMCTransformerConfig(**{k: v for k, v in model_raw.items() if k != "name"})
        errors.extend(validate_config(cfg))

        model = AMCTransformer(cfg)
        total = sum(param.numel() for param in model.parameters())
        if not (_PARAM_MIN < total < _PARAM_MAX):
            errors.append(
                f"parameter count {total:,} outside [{_PARAM_MIN:,}, {_PARAM_MAX:,}]"
            )

        if model.ssm_layer_count + model.attention_layer_count != cfg.n_layers:
            errors.append("SSM + attention layer counts do not sum to n_layers")

        if model.ssm_layer_count != model.attention_layer_count:
            errors.append(
                f"expected 50/50 SSM/MLA split, got "
                f"SSM={model.ssm_layer_count} MLA={model.attention_layer_count}"
            )

        data_section = raw.get("data", {})
        if isinstance(data_section, dict):
            data_seq = int(data_section.get("max_seq_len", cfg.max_seq_len))
            if data_seq > cfg.max_seq_len:
                errors.append(
                    f"data.max_seq_len ({data_seq}) exceeds model.max_seq_len ({cfg.max_seq_len})"
                )

    except Exception as exc:  # noqa: BLE001 — surface as config error
        errors.append(f"model build failed: {exc}")

    training = raw.get("training")
    if training is None:
        errors.append("missing top-level 'training' section")
    elif not isinstance(training, dict):
        errors.append("'training' section must be a mapping")
    else:
        errors.extend(_validate_training_section(training))

    data = raw.get("data")
    if data is None:
        errors.append("missing top-level 'data' section")
    elif not isinstance(data, dict):
        errors.append("'data' section must be a mapping")
    elif int(data.get("max_seq_len", 0)) <= 0:
        errors.append("data.max_seq_len must be positive")

    compute = raw.get("compute")
    if compute is None:
        errors.append("missing top-level 'compute' section")
    elif not isinstance(compute, dict):
        errors.append("'compute' section must be a mapping")
    else:
        errors.extend(_validate_compute_section(compute, training if isinstance(training, dict) else {}))

    deepspeed_path = raw.get("deepspeed_config") or (
        compute.get("deepspeed_config") if isinstance(compute, dict) else None
    )
    if deepspeed_path:
        path = Path(deepspeed_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"deepspeed config not found: {path}")

    return errors


def _validate_training_section(training: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected in _REQUIRED_LRS.items():
        if key not in training:
            errors.append(f"training.{key} is required")
            continue
        value = float(training[key])
        if abs(value - expected) > 1e-12:
            errors.append(f"training.{key} must be {expected:g}, got {value:g}")

    weights = training.get("loss_weights")
    if weights is None:
        errors.append("training.loss_weights is required")
    elif not isinstance(weights, dict):
        errors.append("training.loss_weights must be a mapping")
    else:
        required = ("sft", "surprise", "consistency", "promotion")
        missing = [name for name in required if name not in weights]
        if missing:
            errors.append(f"training.loss_weights missing keys: {missing}")
        total = sum(float(weights[k]) for k in required if k in weights)
        if abs(total - 1.0) > _LOSS_WEIGHT_TOLERANCE:
            errors.append(f"training.loss_weights must sum to 1.0, got {total:.6f}")

    for key in ("batch_size", "gradient_accumulation", "warmup_steps", "max_steps"):
        if key not in training:
            errors.append(f"training.{key} is required")
        elif int(training[key]) <= 0:
            errors.append(f"training.{key} must be positive")

    batch_size = int(training.get("batch_size", 0))
    grad_accum = int(training.get("gradient_accumulation", 0))
    gpus = int(training.get("gpus", training.get("num_gpus", 1)))
    if "effective_batch_size" in training:
        expected = batch_size * grad_accum * max(1, gpus)
        declared = int(training["effective_batch_size"])
        if declared != expected:
            errors.append(
                f"training.effective_batch_size ({declared}) != "
                f"batch_size*grad_accum*gpus ({expected})"
            )

    return errors


def _validate_compute_section(compute: dict[str, Any], training: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(compute.get("gpus", 0)) <= 0:
        errors.append("compute.gpus must be positive")
    precision = str(compute.get("precision", "bf16")).lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        errors.append(f"unsupported compute.precision: {precision}")
    strategy = str(compute.get("strategy", ""))
    if strategy and "deepspeed" in strategy:
        ds_name = compute.get("deepspeed_config", "configs/deepspeed_zero2.json")
        if not ds_name:
            errors.append("compute.strategy requires deepspeed_config path")
    if training:
        train_gpus = int(training.get("gpus", training.get("num_gpus", 0)) or 0)
        compute_gpus = int(compute.get("gpus", 0))
        if train_gpus and compute_gpus and train_gpus != compute_gpus:
            errors.append(
                f"training.gpus ({train_gpus}) must match compute.gpus ({compute_gpus})"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AMC Forge 1B YAML config")
    parser.add_argument(
        "--config",
        default="configs/amc_forge_1b.yaml",
        help="Path to forge config YAML",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    errors = validate_amc_forge_config(raw, repo_root=_REPO_ROOT)
    if errors:
        print(f"FAIL: {config_path}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {config_path}")
    model_name = raw.get("model", {}).get("name", "amc-forge")
    print(f"  model: {model_name}")
    print(f"  training max_steps: {raw['training']['max_steps']}")
    print(f"  data max_seq_len: {raw['data']['max_seq_len']}")
    print(f"  compute: {raw['compute']['gpus']}x {raw['compute'].get('gpu_type', 'GPU')}")


if __name__ == "__main__":
    main()
