"""The end-to-end detect-then-fit pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from facial_landmarks.detectors import FaceDetector, create_detector
from facial_landmarks.drawing import annotate
from facial_landmarks.images import load_image
from facial_landmarks.landmarks import Landmarker, create_landmarker

if TYPE_CHECKING:
    from pathlib import Path

    from facial_landmarks.types import BGRImage, FaceBox, FaceLandmarks

__all__ = ["FaceAnalysis", "FaceAnalyzer"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    """The result of running the pipeline over one image.

    Attributes:
        boxes: Every detected face box.
        landmarks: Fitted landmark sets. Empty when landmark fitting is
            disabled; otherwise parallel to :attr:`boxes`.
    """

    boxes: list[FaceBox] = field(default_factory=list)
    landmarks: list[FaceLandmarks] = field(default_factory=list)

    @property
    def num_faces(self) -> int:
        """Number of faces detected."""
        return len(self.boxes)

    def __bool__(self) -> bool:
        """True when at least one face was detected."""
        return bool(self.boxes)

    def draw_on(self, image: BGRImage, **kwargs: bool) -> BGRImage:
        """Render this analysis onto a copy of ``image``.

        Args:
            image: The image the analysis came from.
            **kwargs: Forwarded to :func:`facial_landmarks.drawing.annotate`.

        Returns:
            An annotated copy of ``image``.
        """
        return annotate(image, self.landmarks, self.boxes, **kwargs)


class FaceAnalyzer:
    """Detects faces and fits landmarks to them.

    The detector and landmarker are injected, so a caller can mix backends or
    substitute fakes in tests without touching this class.

    Args:
        detector: The face detector to use.
        landmarker: The landmark backend, or ``None`` to detect boxes only.
    """

    def __init__(self, detector: FaceDetector, landmarker: Landmarker | None = None) -> None:
        # Unreachable for a type-checked caller; kept for everyone else, who
        # otherwise hits an AttributeError deep inside analyze().
        if not isinstance(detector, FaceDetector):
            msg = (  # type: ignore[unreachable]
                f"detector must implement the FaceDetector protocol, got {type(detector).__name__}"
            )
            raise TypeError(msg)
        if landmarker is not None and not isinstance(landmarker, Landmarker):
            msg = (  # type: ignore[unreachable]
                "landmarker must implement the Landmarker protocol, "
                f"got {type(landmarker).__name__}"
            )
            raise TypeError(msg)
        self.detector = detector
        self.landmarker = landmarker

    @classmethod
    def from_backends(
        cls,
        detector: str = "yunet",
        landmarker: str | None = "lbf",
        *,
        allow_download: bool = True,
        **detector_kwargs: object,
    ) -> FaceAnalyzer:
        """Build an analyzer from backend names.

        Args:
            detector: Detector backend name, ``"yunet"`` or ``"haar"``.
            landmarker: Landmark backend name, or ``None`` for boxes only.
            allow_download: Whether missing weights may be fetched.
            **detector_kwargs: Forwarded to the detector constructor.

        Returns:
            A configured analyzer.
        """
        det = create_detector(detector, allow_download=allow_download, **detector_kwargs)
        marks = None
        if landmarker is not None:
            # Tell the landmarker which box convention it will be fed, so it
            # can remap onto the one its model was trained with.
            marks = create_landmarker(
                landmarker,
                allow_download=allow_download,
                box_convention=det.box_convention,
            )
        return cls(det, marks)

    @property
    def description(self) -> str:
        """Human-readable summary of the configured backends."""
        landmark_name = self.landmarker.name if self.landmarker else "none"
        return f"detector={self.detector.name} landmarker={landmark_name}"

    def analyze(self, image: BGRImage) -> FaceAnalysis:
        """Detect faces in ``image`` and fit landmarks to them.

        Args:
            image: A BGR ``uint8`` image.

        Returns:
            The detections and landmarks found. Both lists are empty when the
            image contains no detectable face - callers never get ``None``,
            which is what made the original notebook's guard clause necessary.
        """
        boxes = self.detector.detect(image)
        if not boxes:
            return FaceAnalysis()
        if self.landmarker is None:
            return FaceAnalysis(boxes=boxes)
        landmarks = self.landmarker.fit(image, boxes)
        return FaceAnalysis(boxes=boxes, landmarks=landmarks)

    def analyze_path(self, path: Path | str) -> tuple[BGRImage, FaceAnalysis]:
        """Load an image from disk and analyse it.

        Args:
            path: Path to the image file.

        Returns:
            A ``(image, analysis)`` pair, so the caller can annotate without
            re-reading the file.
        """
        image = load_image(path)
        return image, self.analyze(image)
