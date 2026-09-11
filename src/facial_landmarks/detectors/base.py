"""The face detector interface and shared input validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from facial_landmarks.types import BGRImage, FaceBox

__all__ = ["FaceDetector", "ensure_bgr"]

_BGR_NDIM = 3
_BGR_CHANNELS = 3


@runtime_checkable
class FaceDetector(Protocol):
    """Locates faces in an image.

    Implementations must be safe to reuse across many images, and must return
    an empty list rather than ``None`` when nothing is found.
    """

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. ``"yunet"``."""
        ...

    @property
    def box_convention(self) -> str:
        """Name of this backend's bounding-box convention.

        Different detectors frame a face differently - a Haar cascade returns a
        near-square box covering the whole head, while YuNet returns a tighter,
        taller crop. Downstream models trained against one convention need the
        other remapped, so each backend declares which it produces.
        """
        ...

    @property
    def model_path(self) -> Path:
        """Path to the weight file backing this detector."""
        ...

    def detect(self, image: BGRImage) -> list[FaceBox]:
        """Detect every face in ``image``.

        Args:
            image: A BGR image of shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns:
            Detected boxes, ordered by descending confidence.
        """
        ...


def ensure_bgr(image: object) -> BGRImage:
    """Validate that ``image`` is a non-empty 8-bit BGR array.

    The parameter is typed ``object`` deliberately: this is the boundary where
    unvalidated input becomes a known-good :data:`BGRImage`, and the runtime
    guards below exist precisely for callers a type checker never saw.

    Args:
        image: The candidate image.

    Returns:
        The image unchanged, as a contiguous ``uint8`` array.

    Raises:
        TypeError: If ``image`` is not a NumPy array.
        ValueError: If the shape or dtype is not a usable BGR image.
    """
    if not isinstance(image, np.ndarray):
        msg = f"expected a numpy array, got {type(image).__name__}"
        raise TypeError(msg)
    if image.ndim != _BGR_NDIM or image.shape[2] != _BGR_CHANNELS:
        msg = f"expected a BGR image of shape (H, W, 3), got {image.shape}"
        raise ValueError(msg)
    if image.size == 0:
        msg = "image is empty"
        raise ValueError(msg)
    if image.dtype != np.uint8:
        msg = f"expected dtype uint8, got {image.dtype}"
        raise ValueError(msg)
    return cast("BGRImage", np.ascontiguousarray(image))
