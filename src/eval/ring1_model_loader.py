"""Ring 1 checkpoint loader — resolve legacy step dirs and infer AMC config from weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, load_model

from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


def resolve_checkpoint_weights_path(checkpoint_path: str | Path | None) -> Path | None:
    """Return a model.safetensors path from a file or checkpoint directory."""
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if path.is_file() and path.suffix == ".safetensors":
        return path
    if not path.is_dir():
        return None

    direct = path / "model.safetensors"
    if direct.is_file():
        return direct

    candidates: list[Path] = []
    for pattern in ("checkpoint-*", "step-*"):
        candidates.extend(sorted(path.glob(pattern)))
    for candidate in reversed(candidates):
        weights = candidate / "model.safetensors"
        if weights.is_file():
            return weights
    return None


def checkpoint_sha256_for_path(checkpoint_path: str | Path | None) -> str:
    """Hash checkpoint weights file if present."""
    weights = resolve_checkpoint_weights_path(checkpoint_path)
    if weights is None:
        return "dummy-checkpoint-not-loaded"
    digest = hashlib.sha256(usedforsecurity=False)
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_amc_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> AMCTransformerConfig:
    """Infer a minimal AMCTransformerConfig from safetensors keys and shapes."""
    embed = state_dict["embed.weight"]
    vocab_size, d_model = int(embed.shape[0]), int(embed.shape[1])

    layer_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("layers.") and key.split(".")[2] in {"attn", "ssm", "ffn"}
        }
    )
    n_layers = max(layer_indices) + 1 if layer_indices else 1

    ssm_layers: set[int] = set()
    for index in layer_indices:
        if any(key.startswith(f"layers.{index}.ssm") for key in state_dict):
            ssm_layers.add(index)

    gate = state_dict.get("layers.0.ffn.gate_proj.weight")
    d_ff = int(gate.shape[0]) if gate is not None else d_model * 4

    q_proj = state_dict.get("layers.0.attn.q_proj.weight")
    if q_proj is not None and q_proj.shape[0] == d_model:
        n_heads = max(1, d_model // 16)
        head_dim = d_model // n_heads
    else:
        n_heads = 4
        head_dim = d_model // n_heads

    bank_dim = d_model if d_model <= 128 else 64
    return AMCTransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        ssm_layers_at=tuple(sorted(ssm_layers)),
        use_hlm_bank=True,
        hlm_bank_size=14,
        hlm_bank_dim=bank_dim,
    )


def load_ring1_amc_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_hlm_bank: bool = True,
) -> tuple[AMCTransformer, Path, str]:
    """Load AMCTransformer weights for Ring 1 measured evaluation."""
    weights_path = resolve_checkpoint_weights_path(checkpoint_path)
    if weights_path is None:
        raise FileNotFoundError(f"No model.safetensors found under {checkpoint_path}")

    state_dict = load_file(str(weights_path))
    config = infer_amc_config_from_state_dict(state_dict)
    config.use_hlm_bank = use_hlm_bank
    model = AMCTransformer(config)
    load_model(model, str(weights_path), strict=False, device=device)
    model.to(device)
    model.eval()
    return model, weights_path, checkpoint_sha256_for_path(checkpoint_path)
