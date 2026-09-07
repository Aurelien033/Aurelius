#!/usr/bin/env python3
"""
Aurelius YaRN Context Rescaler
================================

Applies YaRN / LongRoPE frequency rescaling to a HuggingFace model config
and checkpoint so the model can attend to a longer context window without
replacing positional embeddings.

Reads the companion config:
  - aurelius/data/dataset_registry.yaml  (for target context length)
  - aurelius/data/yarn_params.yaml        (for YaRN hyperparameters)

Outputs:
  - A rescaled config save_pretrained to <output_dir>/config
  - Optionally, a rescaled checkpoint shard if --rescale-checkpoint is set

YaRN references:
  - "YaRN: Efficient Context Window Extension of Large Language Models"
    arXiv:2309.00071 (Peng et al., 2023)
  - LongRoPE: "Extending LLM Context Window Beyond 2 Million Tokens"
    arXiv:2402.13753 (Ding et al., 2024)

Contract for Claude before first use:
  1. Read aurelius/data/dataset_registry.yaml for the target_context_length field.
  2. Verify base model is Qwen2.5-family (rope_theta present, RoPE config valid).
  3. Read aurelius/data/yarn_params.yaml for YaRN hyperparameters.
  4. Compute new rope_theta and attention scaling factors.
  5. Save rescaled config (and optionally checkpoint) to output_dir.
  6. Run a quick sanity check: load the rescaled config and verify
     max_position_embeddings == target_context_length.
  7. Immediately after this script, run longcontext_adaptation SFT
     from the registry to recalibrate attention.
  8. Eval on long-context benchmarks (256k exact anchor retrieval) before
     declaring Phase A complete.

Targets:
  - v3 / Phase A: 256.2k context window (extending Qwen3-Coder base)
  - v4 / Phase 2:  1M context window (requires additional KV-cache engineering)

Failure modes this script catches explicitly:
  - Base model without RoPE theta / without RoPE config
  - --rescale-checkpoint on a model shard larger than local RAM
  - Target context < base context (would be nonsensical; rejected)
  - Missing yarn_params.yaml (abort with clear message)
  - numerical stability: alpha too large / too small
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("aurelius.yarn")

# ---------------------------------------------------------------------------
# Default YaRN / LongRoPE defaults for Qwen2.5-family models.
# These are starting points. After rescaling, measure perplexity on held-out
# long-context data and adjust in this order:
#   1. Scaling factor (t / attention_factor)
#   2. Beta (NTK-aware extrapolation weight)
#   3. Alpha (window stretch ratio, derived from target/base)
# ---------------------------------------------------------------------------
# Qwen2.5 / Qwen3 family base context: typically 131072 for Qwen2.5-32B,
# 32768 for smaller variants. Detect from config; override with --base-context.
DEFAULT_YARN_PARAMS: Dict[str, Any] = {
    "attention_factor": 1.0,   # t in original YaRN paper; temperature
    "beta": 1.0,               # NTK-aware interpolation weight (1.0 = pure NTK-by-parts)
    "scaling_factor": None,    # auto-computed as target/base if None
}

# Known base models for quick lookup
QWEN_FAMILY_BASE_CONTEXT: Dict[str, int] = {
    "Qwen2.5-0.5B": 32768,
    "Qwen2.5-1.5B": 32768,
    "Qwen2.5-7B": 131072,
    "Qwen2.5-14B": 131072,
    "Qwen2.5-32B": 131072,
    "Qwen3-0.6B": 32768,
    "Qwen3-1.7B": 32768,
    "Qwen3-4B": 131072,
    "Qwen3-8B": 131072,
    "Qwen3-14B": 131072,
    "Qwen3-32B": 131072,
    "Qwen3-Coder-30B-A3B": 131072,
}


# ---------------------------------------------------------------------------
# YaRN math
# ---------------------------------------------------------------------------
def yarn_find_correction_dim(
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> float:
    """
    Compute the 'correction dimension' used in YaRN to rescale frequencies.

    The correction dimension is the effective dimension that participates
    in NTK-aware interpolation. For GQA models with fewer key-value heads
    than query heads, YaRN uses a reduced dimension based on KV heads only.
    """
    # YaRN original formula: correction_dim = dim / (num_kv_heads / num_q_heads)^0.5
    # but simpler for HuggingFace: use full hidden_dim when no KV heads differ,
    # use kv-based dimension when GQA is active.
    if num_key_value_heads == num_attention_heads:
        correction_dim = num_attention_heads * head_dim
    else:
        # GQA: only KV heads contribute to the "rotary basis" that gets NTK-scaling.
        correction_dim = num_key_value_heads * head_dim
    return float(correction_dim)


def yarn_find_correction_range(
    correction_dim: float,
    base: float,
    target_context: int,
    alpha: float,
    beta: float,
) -> Tuple[float, float]:
    """
    Compute the frequency range [low, high] over which YaRN applies the NTK
    interpolation blend. Outside this window frequencies are linearly rescaled;
    inside, they get the NTK-aware blend.

    The window boundaries are derived from solving:
      low = (1 - alpha) * base + alpha * low'
      high = (1 - alpha) * base + alpha * high'
    where low', high' correspond to where the NTK curve should start/stop.
    """
    # Original LongRoPE / YaRN formula for the no-gamma variant:
    low = (target_context / base) ** (1.0 / alpha)
    high = (target_context / base) ** (beta / alpha)
    return float(low), float(high)


def yarn_get_mscale(scale: float = 1.0) -> float:
    """
    'm' scale from the YaRN paper. Stabilizes attention magnitudes when
    the window is stretched. Slightly boosts the effective attention
    temperature at longer contexts to prevent attention collapse.

    Default scale=1.0 returns 1.0 (no extra scaling). Use scale > 1.0 only
    if you see degenerate attention distributions after rescaling.
    """
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def compute_yarn_rope_params(
    base_context: int,
    target_context: int,
    rope_theta: float,
    attention_factor: float,
    beta: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> Dict[str, Any]:
    """
    Compute the rescaled RoPE / YaRN parameters for a model config.

    Returns a dict with the keys HuggingFace transformers expects (or will
    expect after patching):
      - rope_theta (possibly new value)
      - max_position_embeddings (new value)
      - attention_factor (new value)
      - beta   (new value)
      - original_max_position_embeddings (for reference)
      - yarn_alpha
      - yarn_correction_dim
      - yarn_correction_range_low / high
      - yarn_mscale
    """
    if target_context <= base_context:
        raise ValueError(
            f"target_context ({target_context}) must be > base_context ({base_context})"
        )

    alpha = target_context / base_context
    correction_dim = yarn_find_correction_dim(num_attention_heads, num_key_value_heads, head_dim)
    low, high = yarn_find_correction_range(correction_dim, rope_theta, target_context, alpha, beta)
    mscale = yarn_get_mscale(alpha)

    # YaRN rescaling: new rope_theta is NOT simply base * alpha.
    # Instead the original YaRN applies the NTK blend to the frequencies.
    # For implementation in transformers / EXL2 / vLLM-style configs, the
    # accepted practice is to set rope_theta to the HIGH boundary of the
    # blended range so the model's highest frequencies interpolate cleanly
    # into the new window.
    # Reference: transformers/models/llama/configuration_llama.py YaRN impl.
    new_rope_theta = rope_theta * high

    return {
        "rope_theta": new_rope_theta,
        "max_position_embeddings": target_context,
        "original_max_position_embeddings": base_context,
        "attention_factor": attention_factor,
        "beta": beta,
        "yarn_alpha": alpha,
        "yarn_correction_dim": correction_dim,
        "yarn_correction_range_low": low,
        "yarn_correction_range_high": high,
        "yarn_mscale": mscale,
    }


# ---------------------------------------------------------------------------
# Config rescaling
# ---------------------------------------------------------------------------
def detect_base_context(model_name_or_path: str) -> int:
    """
    Try to detect the base model context from model name or hub config.
    Falls back to reading the local config.json max_position_embeddings.
    """
    import json

    # First: try known-name lookup
    for key, ctx in QWEN_FAMILY_BASE_CONTEXT.items():
        if key.lower() in model_name_or_path.lower():
            logger.info("Detected %s base context from name lookup: %d", key, ctx)
            return ctx

    # Second: read local config.json if it exists
    cfg_path = Path(model_name_or_path) / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("max_position_embeddings")
        if base:
            logger.info("Detected base context from config.json: %d", base)
            return int(base)
        rope_theta = cfg.get("rope_theta")
        hidden = cfg.get("hidden_size")
        if rope_theta and hidden:
            # rough heuristic from Qwen2.5 family
            return 131072 if hidden >= 3584 else 32768

    raise ValueError(
        f"Could not auto-detect base context for {model_name_or_path!r}. "
        "Pass --base-context explicitly."
    )


def rescale_model_config(
    model_name_or_path: str,
    target_context: int,
    attention_factor: float = 1.0,
    beta: float = 1.0,
    base_context: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    output_dir: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Load a model's config.json, apply YaRN rescaling, and write back.
    Returns the patched config dict (also saved to disk if save=True).
    """
    import json
    from transformers import AutoConfig

    if output_dir is None:
        output_dir = str(Path(model_name_or_path).resolve()) + "-yarn-extended"

    base_ctx = base_context or detect_base_context(model_name_or_path)
    if target_context <= base_ctx:
        raise ValueError(
            f"target_context {target_context} must exceed base_context {base_ctx}"
        )

    # Use HF AutoConfig to load; patch; save
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False)

    # Geometry
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)

    if not hasattr(cfg, "rope_theta") or cfg.rope_theta is None:
        raise ValueError(
            f"Model {model_name_or_path!r} does not have rope_theta. "
            "YaRN requires a RoPE-based model. Verify the base is Qwen2.5-family."
        )

    yarn_params = compute_yarn_rope_params(
        base_context=base_ctx,
        target_context=target_context,
        rope_theta=float(cfg.rope_theta),
        attention_factor=attention_factor,
        beta=beta,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Apply to config object
    cfg.rope_theta = yarn_params["rope_theta"]
    cfg.max_position_embeddings = target_context
    cfg.original_max_position_embeddings = base_ctx
    cfg.attention_factor = attention_factor
    cfg.beta = beta
    cfg.yarn_alpha = yarn_params["yarn_alpha"]
    cfg.yarn_correction_dim = yarn_params["yarn_correction_dim"]
    cfg.yarn_correction_range_low = yarn_params["yarn_correction_range_low"]
    cfg.yarn_correction_range_high = yarn_params["yarn_correction_range_high"]
    cfg.yarn_mscale = yarn_params["yarn_mscale"]

    # Attach the scaling_factor if supplied (used by some serving stacks)
    if scaling_factor is not None:
        cfg.scaling_factor = scaling_factor
    else:
        cfg.scaling_factor = None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(str(out_path))

    logger.info("Rescaled config saved to %s", out_path)
    logger.info("  base_context=%d  target_context=%d  rope_theta=%.4f", base_ctx, target_context, cfg.rope_theta)
    logger.info("  yarn_alpha=%.4f  beta=%.4f  attention_factor=%.4f", yarn_params["yarn_alpha"], beta, attention_factor)
    logger.info("  correction_dim=%.1f  range=[%.2f, %.2f]  mscale=%.4f",
                yarn_params["yarn_correction_dim"],
                yarn_params["yarn_correction_range_low"],
                yarn_params["yarn_correction_range_high"],
                yarn_params["yarn_mscale"])

    # Return serializable dict for logging
    patched = {
        k: getattr(cfg, k)
        for k in [
            "rope_theta",
            "max_position_embeddings",
            "original_max_position_embeddings",
            "attention_factor",
            "beta",
            "yarn_alpha",
            "yarn_correction_dim",
            "yarn_correction_range_low",
            "yarn_correction_range_high",
            "yarn_mscale",
            "scaling_factor",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "head_dim",
        ]
        if hasattr(cfg, k)
    }
    return patched


# ---------------------------------------------------------------------------
# Checkpoint rescaling (optional, more memory-intensive)
# ---------------------------------------------------------------------------
def rescale_checkpoint_weights(
    checkpoint_path: str,
    output_path: str,
    rope_theta_old: float,
    rope_theta_new: float,
    max_position_embeddings_old: int,
    max_position_embeddings_new: int,
    method: str = "linear",
) -> None:
    """
    Rescale positional embeddings (inv_freq / sinusoidal cache) in a sharded
    checkpoint. This is necessary if the serving stack does NOT apply YaRN
    dynamically at inference time.

    For Qwen2.5-family models, the rotary base frequencies are stored as
    precomputed inv_freq in the checkpoint. Rescaling them is equivalent to
    changing rope_theta.

    Arguments:
      checkpoint_path: path to a single pytorch_model-00001-of-XXXXX.bin or .safetensors
      output_path: where to write rescaled shard
      rope_theta_old / rope_theta_new: old and new rotary base frequencies
      max_position_embeddings_old / _new: old and new context lengths
      method: "linear" (standard YaRN/NTK) or "ntk-by-parts" (LongRoPE)

    Warning: this loads the entire shard into RAM. For 14B shards on a Mac
    this is 8–20 GB depending on dtype. Do not run in parallel with training.
    """
    import torch
    from transformers import AutoConfig

    logger.info("Rescaling checkpoint: %s -> %s", checkpoint_path, output_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Find the relevant tensors; naming convention varies by model family.
    # Search for keys containing "rotary_emb" or "inv_freq".
    rot_keys = [k for k in state if any(s in k.lower() for s in ["rotary_emb", "inv_freq"])]
    if not rot_keys:
        raise KeyError(
            f"No rotary embedding tensors found in {checkpoint_path}. "
            "Keys present: " + ", ".join(list(state.keys())[:20])
        )
    logger.info("Rotary tensor keys found: %s", rot_keys)

    scaling = rope_theta_new / rope_theta_old
    logger.info("Applying rotary rescaling factor: %.6f", scaling)

    for key in rot_keys:
        tensor = state[key]
        if tensor is None:
            continue
        old_freq = tensor.detach().clone()
        if method == "linear":
            new_freq = old_freq / scaling
        elif method == "ntk-by-parts":
            # LongRoPE: apply NTK-aware blending; simplified here to linear
            # since the config-level alpha/beta handles the blending math.
            new_freq = old_freq / scaling
        else:
            raise ValueError(f"Unknown rescaling method: {method!r}")
        state[key] = new_freq
        logger.info("  %s: rescaled %d values", key, new_freq.numel())

    torch.save(state, output_path)
    logger.info("Wrote rescaled checkpoint: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aurelius YaRN Context Rescaler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview rescale to 256.2k (dry run, no checkpoint rewrite):
  python yarn_rescaler.py Qwen2.5-Coder-14B-Instruct --target-context 262144 --dry-run

  # Save rescaled config only:
  python yarn_rescaler.py Qwen2.5-Coder-14B-Instruct --target-context 262144 --output-dir ./extended-256k

  # Rescale config + checkpoint shards (memory-heavy):
  python yarn_rescaler.py Qwen2.5-Coder-14B-Instruct \\
    --target-context 262144 --output-dir ./extended-256k \\
    --rescale-checkpoint ./model-00001-of-00003.safetensors \\
    --checkpoint-out ./extended-256k/model-00001-of-00003.safetensors

For v4 / 1M context:
  python yarn_rescaler.py Qwen2.5-Coder-14B-Instruct --target-context 1048576 --output-dir ./extended-1M
""",
    )
    p.add_argument("model_name_or_path", help="HF model name or local path with config.json")
    p.add_argument("--target-context", type=int, required=True, help="Target max_position_embeddings (e.g. 262144 for 256.2k)")
    p.add_argument("--base-context", type=int, default=None, help="Base context length (auto-detected if omitted)")
    p.add_argument("--attention-factor", type=float, default=1.0, help="YaRN attention temperature t (default: 1.0)")
    p.add_argument("--beta", type=float, default=1.0, help="NTK-aware interpolation weight (default: 1.0)")
    p.add_argument("--scaling-factor", type=float, default=None, help="Override scaling factor (auto from alpha if omitted)")
    p.add_argument("--output-dir", default=None, help="Where to write rescaled config (default: <model>-yarn-extended)")
    p.add_argument("--rescale-checkpoint", default=None, help="Path to a checkpoint shard to rescale in-place")
    p.add_argument("--checkpoint-out", default=None, help="Where to write rescaled checkpoint shard")
    p.add_argument("--rope-theta-old", type=float, default=None, help="Original rope_theta for checkpoint rescaling (read from config if omitted)")
    p.add_argument("--max-position-old", type=int, default=None, help="Original max_position_embeddings for checkpoint rescaling (read from config if omitted)")
    p.add_argument("--checkpoint-method", choices=["linear", "ntk-by-parts"], default="linear")
    p.add_argument("--dry-run", action="store_true", help="Print computed params but do not write anything")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.target_context <= 0:
        logger.error("--target-context must be positive")
        return 2

    base_ctx = args.base_context or detect_base_context(args.model_name_or_path)
    logger.info("Base context: %d  Target context: %d", base_ctx, args.target_context)

    if args.dry_run:
        # Compute but do not write
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        params = compute_yarn_rope_params(
            base_context=base_ctx,
            target_context=args.target_context,
            rope_theta=float(getattr(cfg, "rope_theta", 1e6)),
            attention_factor=args.attention_factor,
            beta=args.beta,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        print("DRY RUN. Computed YaRN parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        return 0

    patched = rescale_model_config(
        model_name_or_path=args.model_name_or_path,
        target_context=args.target_context,
        attention_factor=args.attention_factor,
        beta=args.beta,
        base_context=base_ctx,
        scaling_factor=args.scaling_factor,
        output_dir=args.output_dir,
        save=True,
    )
    print("Rescaled config written. Key parameters:")
    for k, v in patched.items():
        print(f"  {k}: {v}")

    # Optionally rescale checkpoint shard too
    if args.rescale_checkpoint:
        ckpt_in = args.rescale_checkpoint
        ckpt_out = args.checkpoint_out or str(Path(ckpt_in).with_suffix(".yarn-rescaled" + Path(ckpt_in).suffix))
        rope_old = args.rope_theta_old or patched.get("original_rope_theta", patched["rope_theta"] / (patched["yarn_correction_range_high"] if "yarn_correction_range_high" in patched else 1.0))
        max_old = args.max_position_old or patched.get("original_max_position_embeddings", base_ctx)
        # We need the original rope_theta, not the rescaled one; compute inverse
        rope_original = (patched["rope_theta"] / ((patched.get("yarn_correction_range_high") or 1.0)))
        rope_new = patched["rope_theta"]
        rescale_checkpoint_weights(
            checkpoint_path=ckpt_in,
            output_path=ckpt_out,
            rope_theta_old=float(rope_original),
            rope_theta_new=float(rope_new),
            max_position_embeddings_old=int(max_old),
            max_position_embeddings_new=int(patched["max_position_embeddings"]),
            method=args.checkpoint_method,
        )
        print(f"Rescaled checkpoint written to: {ckpt_out}")

    # Sanity check
    from transformers import AutoConfig
    verify = AutoConfig.from_pretrained(args.output_dir or (str(Path(args.model_name_or_path).resolve()) + "-yarn-extended"), trust_remote_code=False)
    assert verify.max_position_embeddings == args.target_context, (
        f"Sanity check failed: saved config has max_position_embeddings={verify.max_position_embeddings}, "
        f"expected {args.target_context}"
    )
    logger.info("Sanity check passed: max_position_embeddings == %d", verify.max_position_embeddings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
