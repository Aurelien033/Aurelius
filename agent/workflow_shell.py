"""Compatibility shim for ``agent.workflow_shell``.

Canonical implementation:
``src.agent.workflow_shell``

This module is retained during the ``agent`` -> ``src.agent`` migration so that
existing ``from agent.workflow_shell import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'agent' is deprecated. Use 'src.agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.agent.workflow_shell import *  # noqa: E402, F401, F403
from src.agent import workflow_shell as _src_module  # noqa: E402

try:
    from src.agent.workflow_shell import __all__ as _src_all  # noqa: E402
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
