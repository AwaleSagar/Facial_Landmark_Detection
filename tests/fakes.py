"""Lightweight backend doubles.

Loading the real weights costs about a second, so every test that only needs
pipeline behaviour uses these instead of the genuine backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from facial_landmarks import FaceLandmarks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from facial_landmarks import FaceBox
    from facial_landmarks.types import BGRImage

__all__ = ["FakeDetector", "FakeLandmarker"]


class FakeDetector:
    """A detector that returns a fixed list of boxes and counts its calls."""

    def __init__(self, boxes: Sequence[FaceBox] = (), convention: str = "haar") -> None:
        self.boxes = list(boxes)
        self._convention = convention
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def box_convention(self) -> str:
        return self._convention

    @property
    def model_path(self) -> Path:
        return Path("/fake/detector")

    def detect(self, image: BGRImage) -> list[FaceBox]:
        self.calls += 1
        return list(self.boxes)


class FakeLandmarker:
    """A landmarker returning deterministic points along each box's diagonal."""

    def __init__(self, num_points: int = 68) -> None:
        self._num_points = num_points
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def box_convention(self) -> str:
        return "haar"

    @property
    def model_path(self) -> Path:
        return Path("/fake/landmarker")

    def fit(self, image: BGRImage, faces: Sequence[FaceBox]) -> list[FaceLandmarks]:
        self.calls += 1
        results = []
        for face in faces:
            xs = np.linspace(face.x, face.right, self._num_points, dtype=np.float32)
            ys = np.linspace(face.y, face.bottom, self._num_points, dtype=np.float32)
            results.append(FaceLandmarks(np.column_stack([xs, ys]), face))
        return results
