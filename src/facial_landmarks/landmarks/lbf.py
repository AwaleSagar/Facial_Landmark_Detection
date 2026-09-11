"""68-point landmark localisation with OpenCV's LBF facemark regressor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from facial_landmarks._native import suppress_native_stdout
from facial_landmarks.assets import resolve_asset
from facial_landmarks.detectors.base import ensure_bgr
from facial_landmarks.types import NUM_LANDMARKS_68, FaceLandmarks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from facial_landmarks.types import BGRImage, FaceBox

__all__ = ["CONVENTION_TO_HAAR", "LBFLandmarker"]

logger = logging.getLogger(__name__)

#: Transforms mapping each detector's box convention onto the Haar-cascade
#: convention, as ``(scale_x, scale_y, shift_x, shift_y)``.
#:
#: The LBF regressor was trained on Haar-cascade rectangles, so feeding it the
#: tighter box YuNet produces puts the model off its operating point and drags
#: the jaw contour downward. The YuNet figures below were fitted on the 77
#: images of this repository's sample set where both detectors fire and agree
#: (IoU > 0.3): Haar boxes are consistently 1.22x wider and 0.93x as tall,
#: about a shared centre (sd 0.09 and 0.07 respectively).
#:
#: Applying the transform halves the median disagreement between landmarks
#: fitted from a YuNet box and those fitted from the Haar box on the same face
#: (normalised mean error 0.054 -> 0.022). That measures agreement with the
#: model's native operating point rather than accuracy against ground truth,
#: which this dataset does not carry - but it is the operating point the
#: regressor was trained for.
CONVENTION_TO_HAAR: Final[dict[str, tuple[float, float, float, float]]] = {
    "haar": (1.0, 1.0, 0.0, 0.0),
    "yunet": (1.221, 0.932, 0.002, -0.017),
}


class LBFLandmarker:
    """Local Binary Features regressor producing the iBUG 300-W 68 points.

    Requires ``opencv-contrib-python``; the ``cv2.face`` module is absent from
    the plain ``opencv-python`` wheel.

    Args:
        model_path: Explicit path to ``lbfmodel.yaml``. Resolved from the asset
            cache when omitted.
        box_convention: The box convention of the detector feeding this
            landmarker, as a key of :data:`CONVENTION_TO_HAAR`. Incoming boxes
            are remapped onto the Haar convention the model was trained with.
            :class:`~facial_landmarks.pipeline.FaceAnalyzer` sets this from the
            detector automatically.
        allow_download: Whether missing weights may be fetched over the network.

    Raises:
        RuntimeError: If the installed OpenCV has no ``cv2.face`` module.
        KeyError: If ``box_convention`` is unknown.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        box_convention: str = "haar",
        allow_download: bool = True,
    ) -> None:
        if not hasattr(cv2, "face"):  # pragma: no cover - depends on the wheel
            msg = (
                "cv2.face is unavailable. Install opencv-contrib-python "
                "instead of opencv-python to use the LBF landmarker."
            )
            raise RuntimeError(msg)
        if box_convention not in CONVENTION_TO_HAAR:
            msg = (
                f"unknown box convention {box_convention!r}; "
                f"choose from {sorted(CONVENTION_TO_HAAR)}"
            )
            raise KeyError(msg)
        self.box_convention = box_convention
        self._to_haar = CONVENTION_TO_HAAR[box_convention]
        self._model_path = resolve_asset("lbfmodel", path=model_path, allow_download=allow_download)
        self._facemark = cv2.face.createFacemarkLBF()
        # loadModel prints straight to C++ stdout; keep it out of CLI output.
        with suppress_native_stdout():
            self._facemark.loadModel(str(self._model_path))
        logger.debug("initialised LBF landmarker from %s", self._model_path)

    @property
    def name(self) -> str:
        """Short backend identifier."""
        return "lbf"

    @property
    def num_points(self) -> int:
        """Number of points produced per face."""
        return NUM_LANDMARKS_68

    @property
    def model_path(self) -> Path:
        """Path to the model backing this landmarker."""
        return self._model_path

    def fit(self, image: BGRImage, faces: Sequence[FaceBox]) -> list[FaceLandmarks]:
        """Fit 68 landmarks inside each box.

        Args:
            image: The BGR image the boxes came from.
            faces: Boxes to fit.

        Returns:
            One landmark set per box, in input order. Empty if ``faces`` is
            empty or the regressor fails to converge.
        """
        if not faces:
            return []
        image = ensure_bgr(image)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remap onto the convention the regressor was trained with, then clip
        # so a widened box cannot run off the edge of the frame.
        height, width = image.shape[:2]
        adapted: list[FaceBox] = []
        for box in faces:
            remapped = box.remapped(*self._to_haar)
            try:
                adapted.append(remapped.clipped_to(width, height))
            except ValueError:  # pragma: no cover - defensive
                adapted.append(box)

        # The facemark API takes an (N, 4) int32 array of x/y/w/h rectangles.
        rects = np.array([box.to_int_xywh() for box in adapted], dtype=np.int32)
        ok, raw = self._facemark.fit(grey, rects)
        if not ok or raw is None:  # pragma: no cover - regressor rarely fails
            logger.warning("LBF fit failed for %d face(s)", len(faces))
            return []

        results: list[FaceLandmarks] = []
        # Results carry the detector's original box, not the adapted one: the
        # adaptation is an implementation detail of this regressor.
        for box, points in zip(faces, raw, strict=False):
            # The regressor returns (1, 68, 2); FaceLandmarks reshapes to (68, 2).
            results.append(FaceLandmarks(np.asarray(points, dtype=np.float32), box))
        return results
