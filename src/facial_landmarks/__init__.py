"""Face detection and 68-point facial landmark localisation.

The quickest way in:

    >>> from facial_landmarks import FaceAnalyzer
    >>> analyzer = FaceAnalyzer.from_backends()          # YuNet + LBF
    >>> image, analysis = analyzer.analyze_path("data/image_0.jpg")
    >>> analysis.num_faces
    1
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from facial_landmarks.assets import REGISTRY, ModelAsset, cache_dir, resolve_asset
from facial_landmarks.detectors import (
    FaceDetector,
    HaarCascadeDetector,
    YuNetDetector,
    create_detector,
)
from facial_landmarks.drawing import annotate, draw_boxes, draw_landmarks
from facial_landmarks.images import contact_sheet, load_image, save_image, to_rgb
from facial_landmarks.landmarks import Landmarker, LBFLandmarker, create_landmarker
from facial_landmarks.pipeline import FaceAnalysis, FaceAnalyzer
from facial_landmarks.types import (
    FACE_REGIONS_68,
    NUM_LANDMARKS_68,
    FaceBox,
    FaceLandmarks,
)

try:
    __version__ = version("facial-landmarks")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "FACE_REGIONS_68",
    "NUM_LANDMARKS_68",
    "REGISTRY",
    "FaceAnalysis",
    "FaceAnalyzer",
    "FaceBox",
    "FaceDetector",
    "FaceLandmarks",
    "HaarCascadeDetector",
    "LBFLandmarker",
    "Landmarker",
    "ModelAsset",
    "YuNetDetector",
    "__version__",
    "annotate",
    "cache_dir",
    "contact_sheet",
    "create_detector",
    "create_landmarker",
    "draw_boxes",
    "draw_landmarks",
    "load_image",
    "resolve_asset",
    "save_image",
    "to_rgb",
]
