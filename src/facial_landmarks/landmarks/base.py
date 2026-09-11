"""The landmark-fitting interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from facial_landmarks.types import BGRImage, FaceBox, FaceLandmarks

__all__ = ["Landmarker"]


@runtime_checkable
class Landmarker(Protocol):
    """Fits landmark points inside already-detected face boxes."""

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. ``"lbf"``."""
        ...

    @property
    def num_points(self) -> int:
        """Number of points this backend produces per face."""
        ...

    @property
    def box_convention(self) -> str:
        """The detector box convention this backend expects to be fed.

        Set by the caller so the backend can remap incoming boxes onto the
        convention its own model was trained against.
        """
        ...

    @property
    def model_path(self) -> Path:
        """Path to the weight file backing this landmarker."""
        ...

    def fit(self, image: BGRImage, faces: Sequence[FaceBox]) -> list[FaceLandmarks]:
        """Localise landmarks for each box in ``faces``.

        Args:
            image: The BGR image the boxes were detected in.
            faces: Boxes to fit. An empty sequence yields an empty result.

        Returns:
            One :class:`~facial_landmarks.types.FaceLandmarks` per input box,
            in the same order.
        """
        ...
