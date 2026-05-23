#!/usr/bin/env python3
"""Count parameters of an AMCTransformer by submodule category.

Usage:
  python scripts/count_params.py --config configs/amc_forge_1b.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def count_by_category(model: AMCTransformer) -> dict[str, int]:
    """Count parameters grouped by submodule category."""
    counts = {
        "embed": 0,
        "mla_layers": 0,
        "ssm_layers": 0,
        "promotion_gates": 0,
        "surprise_heads": 0,
        "norm_final": 0,
        "lm_head": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        numel = param.numel()
        if "embed" in name and "lm_head" not in name:
            counts["embed"] += numel
        elif "promotion_gate" in name or "promotion_gates" in name:
            counts["promotion_gates"] += numel
        elif "surprise_head" in name:
            counts["surprise_heads"] += numel
        elif "layers." in name:
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx in model.ssm_layer_indices:
                    counts["ssm_layers"] += numel
                else:
                    counts["mla_layers"] += numel
            else:
                counts["other"] += numel
        elif "norm" in name and "layer" not in name:
            counts["norm_final"] += numel
        elif "lm_head" in name:
            counts["lm_head"] += numel
        else:
            counts["other"] += numel

    return counts


def count_parameters(model: AMCTransformer, *, trainable_only: bool = True) -> dict[str, int]:
    """Return parameter counts by category and total."""
    if trainable_only:
        counts = count_by_category(model)
    else:
        counts = count_by_category(model)
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def validate_config(config: AMCTransformerConfig) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []

    if config.d_model % config.n_heads != 0:
        errors.append(
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )

    head_dim = config.d_model // config.n_heads
    if head_dim < 32:
        errors.append(f"head_dim ({head_dim}) too small (min 32)")

    if config.ssm_headdim and config.d_model % config.ssm_headdim != 0:
        errors.append(
            f"d_model ({config.d_model}) must be divisible by ssm_headdim ({config.ssm_headdim})"
        )

    if config.n_layers < 4:
        errors.append(f"n_layers ({config.n_layers}) too small for hybrid model")

    if config.vocab_size < 100:
        errors.append(f"vocab_size ({config.vocab_size}) suspiciously small")

    if config.max_seq_len < 64:
        errors.append(f"max_seq_len ({config.max_seq_len}) too small")

    if config.kv_lrank > config.d_model:
        errors.append(f"kv_lrank ({config.kv_lrank}) larger than d_model ({config.d_model})")

    return errors


validate_amc_config = validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Count AMCTransformer parameters")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import yaml

    from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig

    config_path = Path(args.config)
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    config = AMCTransformerConfig(**raw.get("model", raw))

    errors = validate_config(config)
    if errors:
        print("CONFIG ERRORS:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    model = AMCTransformer(config)
    counts = count_by_category(model)
    total = sum(counts.values())

    print(f"\n=== {config.__class__.__name__} Parameter Count ===\n")

    for category, count in counts.items():
        if count == 0:
            continue
        pct = 100 * count / total
        print(f"  {category:<20} {count:>14,} ({pct:5.2f}%)")

    print(f"\n  {'TOTAL':<20} {total:>14,}")
    print(f"\n  SSM layers:     {model.ssm_layer_count}")
    print(f"  Attention (MLA): {model.attention_layer_count}")
    print(f"  Ratio: SSM/total = {model.ssm_layer_count / config.n_layers:.2f}")
    print()

    weights_gb = total * 2 / 1e9
    print(f"  Model weights (BF16): {weights_gb:.2f} GB")
    print(f"  Full state (Adam + grads): ~{weights_gb * 4:.2f} GB")


if __name__ == "__main__":
    main()
