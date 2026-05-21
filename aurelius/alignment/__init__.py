"""Compatibility shim — imports from src.alignment."""

from src.alignment import *  # noqa: F401,F403

try:
    from src.alignment import __all__ as _src_all  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
else:
    __all__ = list(_src_all)
