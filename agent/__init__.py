# DEPRECATED: Use src.agent instead.
import warnings

warnings.warn(
    "Importing from 'agent' is deprecated. Use 'src.agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.agent import *  # noqa: F401,F403

try:
    from src.agent import __all__ as _src_all  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_") and name != "warnings"]
else:
    __all__ = list(_src_all)
