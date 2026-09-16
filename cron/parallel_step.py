"""Compatibility shim for ``cron.parallel_step``.

Canonical implementation:
``src.workflow.parallel_step``

This module is retained during the ``cron`` -> ``src.workflow`` migration so that
existing ``from cron.parallel_step import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'cron' is deprecated. Use 'src.workflow' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.workflow.parallel_step import *  # noqa: E402, F401, F403
from src.workflow import parallel_step as _src_module  # noqa: E402

try:
    from src.workflow.parallel_step import __all__ as _src_all  # noqa: E402
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
