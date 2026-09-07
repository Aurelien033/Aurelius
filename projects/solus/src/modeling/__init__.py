"""
Solus-7B -- Package init (SolusConfig, SolusForCausalLM).
MoE module is lazily imported to avoid circular dependency cycles.
"""
from __future__ import annotations

from .config import SolusConfig
from .model import SolusForCausalLM

__all__ = ["SolusForCausalLM", "SolusConfig"]

# ── Lazy MoE import ──────────────────────────────────────────────────
# SolusMoEForCausalLM is imported on first access, not during package init.
# This avoids circular imports: sparse_model -> src.modeling.config -> ...
# mapping __getattr__ to actual import
_moe_names = {"SolusMoEForCausalLM"}
def __getattr__(name: str):
    if name in _moe_names:
        from ..moe.sparse_model import SolusMoEForCausalLM
        # Do not cache in globals() here -- let stdlib cache via sys.modules
        return globals().get(name, SolusMoEForCausalLM)
    raise AttributeError(f"module 'src.modeling' has no attribute {name!r}")
