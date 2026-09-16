"""Compatibility shim for ``cron.dag_executor``.

Canonical implementation:
``src.workflow.dag_executor``

This module is retained during the ``cron`` -> ``src.workflow`` migration so that
existing ``from cron.dag_executor import ...`` statements keep working.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'cron' is deprecated. Use 'src.workflow' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.workflow.dag_executor import *  # noqa: E402, F401, F403
from src.workflow import dag_executor as _src_module  # noqa: E402

try:
    from src.workflow.dag_executor import __all__ as _src_all  # noqa: E402
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
