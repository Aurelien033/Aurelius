# DEPRECATED: Use src.tools instead.
import warnings

warnings.warn(
    "Importing from 'tools' is deprecated. Use 'src.tools' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.tools import *  # noqa: F401,F403

try:
    from src.tools import __all__ as _src_all  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_") and name != "warnings"]
else:
    __all__ = list(_src_all)
