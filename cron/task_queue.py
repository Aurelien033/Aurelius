"""Compatibility shim for ``cron.task_queue``.

Canonical implementation:
``src.workflow.task_queue``

This module is retained during the ``cron`` -> ``src.workflow`` migration so that
existing ``from cron.task_queue import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'cron' is deprecated. Use 'src.workflow' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.workflow.task_queue import *  # noqa: E402, F401, F403
from src.workflow import task_queue as _src_module  # noqa: E402

try:
    from src.workflow.task_queue import __all__ as _src_all  # noqa: E402
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
