# DEPRECATED: Use src.serving instead.
import warnings

warnings.warn(
    "Importing from 'gateway' is deprecated. Use 'src.serving' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.serving import *  # noqa: F401,F403

try:
    from src.serving import __all__ as _src_all  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_") and name != "warnings"]
else:
    __all__ = list(_src_all)
