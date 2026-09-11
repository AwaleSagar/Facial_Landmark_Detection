"""Landmark localisation backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from facial_landmarks.landmarks.base import Landmarker
from facial_landmarks.landmarks.lbf import LBFLandmarker

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["LANDMARKER_BACKENDS", "LBFLandmarker", "Landmarker", "create_landmarker"]

#: Mapping of CLI-facing backend names to their constructors.
LANDMARKER_BACKENDS: Final[dict[str, Callable[..., Landmarker]]] = {"lbf": LBFLandmarker}


def create_landmarker(backend: str = "lbf", **kwargs: object) -> Landmarker:
    """Instantiate a landmark backend by name.

    Args:
        backend: Currently only ``"lbf"``.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A ready-to-use landmarker.

    Raises:
        KeyError: If ``backend`` is unknown.
    """
    try:
        factory = LANDMARKER_BACKENDS[backend]
    except KeyError:
        msg = f"unknown landmarker backend {backend!r}; choose from {sorted(LANDMARKER_BACKENDS)}"
        raise KeyError(msg) from None
    return factory(**kwargs)
