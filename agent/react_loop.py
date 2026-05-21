"""Compatibility shim for ``agent.react_loop``.

Prefer :mod:`src.agent.react_loop` in all new code.
This module is retained only so that existing ``from agent.react_loop import …``
imports continue to work during the ``agent`` → ``src.agent`` migration.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'agent' is deprecated. Use 'src.agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.agent.react_loop import *  # noqa: E402, F401, F403

try:
    from src.agent.react_loop import __all__ as _src_all  # noqa: E402
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_") and name != "warnings"]
else:
    __all__ = list(_src_all)
