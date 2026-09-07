# DEPRECATED: Use src.agent instead.
import warnings

warnings.warn(
    "Importing from 'agent' is deprecated. Use 'src.agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.agent import *  # noqa: F403


def __getattr__(name: str) -> object:
    """Lazy re-export of __all__ to avoid circular import at init time."""
    if name == "__all__":
        import src.agent as _src

        return _src.__all__
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
