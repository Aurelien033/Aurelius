"""Compatibility shim for ``gateway.streaming_handler``.

Canonical implementation:
``src.serving.streaming_handler``

This module is retained during the ``gateway`` -> ``src.serving`` migration so that
existing ``from gateway.streaming_handler import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'gateway' is deprecated. Use 'src.serving' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.serving.streaming_handler import *  # noqa: E402, F401, F403
from src.serving import streaming_handler as _src_module  # noqa: E402

try:
    from src.serving.streaming_handler import __all__ as _src_all  # noqa: E402
except ImportError:
    __all__ = [name for name in dir(_src_module) if not name.startswith("__")]
else:
    __all__ = list(_src_all)


def __getattr__(name: str) -> object:
    """Forward private/undecorated names to the canonical implementation."""
    try:
        return getattr(_src_module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
