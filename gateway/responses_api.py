"""Compatibility shim for ``gateway.responses_api``.

Canonical implementation:
``src.serving.responses_api``

This module is retained during the ``gateway`` -> ``src.serving`` migration so that
existing ``from gateway.responses_api import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'gateway' is deprecated. Use 'src.serving' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.serving.responses_api import *  # noqa: E402, F401, F403
from src.serving import responses_api as _src_module  # noqa: E402

try:
    from src.serving.responses_api import __all__ as _src_all  # noqa: E402
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
