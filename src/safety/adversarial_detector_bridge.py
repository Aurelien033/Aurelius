"""Bridge: re-exports AdversarialDetector from src.security into the safety surface."""

from __future__ import annotations

from src.security.adversarial_detector import (  # noqa: F401
    AdversarialDetector,
    AdversarialPattern,
    AdversarialResult,
)
