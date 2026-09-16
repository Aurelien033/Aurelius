"""Compatibility shim for ``agent.session_journal``.

Canonical implementation:
``src.agent.session_journal``

This module is retained during the ``agent`` -> ``src.agent`` migration so that
existing ``from agent.session_journal import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'agent' is deprecated. Use 'src.agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.agent.session_journal import *  # noqa: E402, F401, F403
from src.agent import session_journal as _src_module  # noqa: E402

try:
    from src.agent.session_journal import __all__ as _src_all  # noqa: E402
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
