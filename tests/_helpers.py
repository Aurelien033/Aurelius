"""Shared test factories.

These small configuration builders were previously copy-pasted into dozens of
test modules with identical bodies and differing local names. They live here
once so a single change is visible everywhere.
"""

from src.model.config import AureliusConfig


def make_small_cfg() -> AureliusConfig:
    """Minimal model config: small enough to run in milliseconds."""
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
