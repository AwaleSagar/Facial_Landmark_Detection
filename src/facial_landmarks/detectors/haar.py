"""Haar cascade face detector - the legacy baseline kept for comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from facial_landmarks.assets import resolve_asset
from facial_landmarks.detectors.base import ensure_bgr
from facial_landmarks.types import FaceBox

if TYPE_CHECKING:
    from facial_landmarks.types import BGRImage

__all__ = ["HaarCascadeDetector"]

logger = logging.getLogger(__name__)


class HaarCascadeDetector:
    """Viola-Jones frontal-face detection via ``cv2.CascadeClassifier``.

    This is the detector the original notebook used. It is retained so the
    ``benchmark`` command can quantify what the YuNet default buys, but it is
    markedly weaker on non-frontal faces and reports no real confidence, so
    every box is scored ``1.0``.

    Args:
        model_path: Explicit path to the cascade XML. Resolved from the asset
            cache when omitted.
        scale_factor: Image pyramid step; smaller is slower but finds more.
        min_neighbours: Higher values suppress more false positives at the
            cost of recall. Defaults to 3, OpenCV's own default and the
            best-scoring setting on this repository's sample images, so the
            benchmark compares against a fairly tuned baseline.
        min_size: Smallest face to consider, in pixels.
        allow_download: Whether missing weights may be fetched over the network.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbours: int = 3,
        min_size: tuple[int, int] = (30, 30),
        allow_download: bool = True,
    ) -> None:
        if scale_factor <= 1.0:
            msg = f"scale_factor must be greater than 1.0, got {scale_factor}"
            raise ValueError(msg)
        self._model_path = resolve_asset(
            "haarcascade", path=model_path, allow_download=allow_download
        )
        self._cascade = cv2.CascadeClassifier(str(self._model_path))
        if self._cascade.empty():  # pragma: no cover - corrupt asset
            msg = f"OpenCV could not load the cascade at {self._model_path}"
            raise ValueError(msg)
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours
        self.min_size = min_size

    @property
    def name(self) -> str:
        """Short backend identifier."""
        return "haar"

    @property
    def box_convention(self) -> str:
        """This backend's bounding-box convention."""
        return "haar"

    @property
    def model_path(self) -> Path:
        """Path to the cascade XML backing this detector."""
        return self._model_path

    def detect(self, image: BGRImage) -> list[FaceBox]:
        """Detect faces in ``image``.

        Args:
            image: A BGR ``uint8`` image.

        Returns:
            Detected boxes, largest first. Empty when nothing is found.
        """
        image = ensure_bgr(image)
        height, width = image.shape[:2]
        # Cascades operate on a single channel; equalising first is the
        # conventional preparation and steadies detection under uneven
        # lighting, though on this sample set it leaves recall unchanged.
        grey = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        raw = self._cascade.detectMultiScale(
            grey,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbours,
            minSize=self.min_size,
        )
        # detectMultiScale returns an empty *tuple* when there are no hits and
        # an ndarray otherwise. Comparing that result to () - as the original
        # notebook did - raises ValueError once it is an array, so normalise
        # through np.asarray and test the length instead.
        boxes = np.asarray(raw, dtype=np.float64).reshape(-1, 4)
        faces: list[FaceBox] = []
        for x, y, w, h in boxes:
            box = FaceBox(float(x), float(y), float(w), float(h), score=1.0)
            try:
                faces.append(box.clipped_to(width, height))
            except ValueError:  # pragma: no cover - defensive
                logger.debug("discarding out-of-frame detection %s", box)
        faces.sort(key=lambda box: box.area, reverse=True)
        return faces
