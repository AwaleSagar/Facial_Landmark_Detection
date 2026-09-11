"""Face detector backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from facial_landmarks.detectors.base import FaceDetector, ensure_bgr
from facial_landmarks.detectors.haar import HaarCascadeDetector
from facial_landmarks.detectors.yunet import YuNetDetector

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DETECTOR_BACKENDS",
    "FaceDetector",
    "HaarCascadeDetector",
    "YuNetDetector",
    "create_detector",
    "ensure_bgr",
]

#: Mapping of CLI-facing backend names to their constructors.
DETECTOR_BACKENDS: Final[dict[str, Callable[..., FaceDetector]]] = {
    "yunet": YuNetDetector,
    "haar": HaarCascadeDetector,
}


def create_detector(backend: str = "yunet", **kwargs: object) -> FaceDetector:
    """Instantiate a detector backend by name.

    Args:
        backend: Either ``"yunet"`` (default, recommended) or ``"haar"``.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A ready-to-use detector.

    Raises:
        KeyError: If ``backend`` is unknown.
    """
    try:
        factory = DETECTOR_BACKENDS[backend]
    except KeyError:
        msg = f"unknown detector backend {backend!r}; choose from {sorted(DETECTOR_BACKENDS)}"
        raise KeyError(msg) from None
    return factory(**kwargs)
