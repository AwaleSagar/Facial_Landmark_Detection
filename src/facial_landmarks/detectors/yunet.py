"""YuNet CNN face detector - the default backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import cv2

from facial_landmarks.assets import resolve_asset
from facial_landmarks.detectors.base import ensure_bgr
from facial_landmarks.types import FaceBox

if TYPE_CHECKING:
    from facial_landmarks.types import BGRImage

__all__ = ["YuNetDetector"]

logger = logging.getLogger(__name__)

#: Names of the five keypoints YuNet emits, in model output order.
YUNET_KEYPOINTS: Final = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "right_mouth_corner",
    "left_mouth_corner",
)

_ROW_LENGTH: Final = 15  # 4 box + 10 keypoint + 1 score


class YuNetDetector:
    """Face detection with the YuNet CNN shipped by the OpenCV Zoo.

    YuNet is a ~230 KB single-shot network that replaces the Haar cascade used
    by the original notebook. On this repository's 113 sample images it finds a
    face in 108 against the cascade's 87, and it also returns a confidence
    score and five coarse keypoints per face.

    Args:
        model_path: Explicit path to the ONNX weights. Resolved from the asset
            cache (downloading if needed) when omitted.
        score_threshold: Minimum confidence for a detection to be kept.
        nms_threshold: IoU threshold for non-maximum suppression.
        top_k: Maximum number of boxes retained before NMS.
        allow_download: Whether missing weights may be fetched over the network.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        allow_download: bool = True,
    ) -> None:
        if not 0.0 < score_threshold <= 1.0:
            msg = f"score_threshold must be in (0, 1], got {score_threshold}"
            raise ValueError(msg)
        self._model_path = resolve_asset("yunet", path=model_path, allow_download=allow_download)
        self.score_threshold = score_threshold
        self._detector = cv2.FaceDetectorYN.create(
            str(self._model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self._input_size: tuple[int, int] | None = None
        logger.debug("initialised YuNet from %s", self._model_path)

    @property
    def name(self) -> str:
        """Short backend identifier."""
        return "yunet"

    @property
    def box_convention(self) -> str:
        """This backend's bounding-box convention."""
        return "yunet"

    @property
    def model_path(self) -> Path:
        """Path to the ONNX weights backing this detector."""
        return self._model_path

    def detect(self, image: BGRImage) -> list[FaceBox]:
        """Detect faces in ``image``.

        Args:
            image: A BGR ``uint8`` image.

        Returns:
            Boxes sorted by descending confidence; empty if no face is found.
        """
        image = ensure_bgr(image)
        height, width = image.shape[:2]
        # The network is fully convolutional but OpenCV needs to be told the
        # frame size whenever it changes; skipping the redundant call keeps
        # video throughput up.
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, raw = self._detector.detect(image)
        if raw is None or len(raw) == 0:
            return []

        faces: list[FaceBox] = []
        for row in raw:
            if len(row) < _ROW_LENGTH:  # pragma: no cover - guards a model change
                logger.warning("unexpected YuNet row of length %d, skipping", len(row))
                continue
            x, y, w, h = (float(v) for v in row[:4])
            if w <= 0 or h <= 0:  # pragma: no cover - defensive
                continue
            keypoints = tuple(
                (name, (float(row[4 + 2 * i]), float(row[5 + 2 * i])))
                for i, name in enumerate(YUNET_KEYPOINTS)
            )
            box = FaceBox(x, y, w, h, score=float(row[14]), keypoints=keypoints)
            try:
                faces.append(box.clipped_to(width, height))
            except ValueError:  # pragma: no cover - box fully outside the frame
                logger.debug("discarding out-of-frame detection %s", box)
        faces.sort(key=lambda box: box.score, reverse=True)
        return faces
