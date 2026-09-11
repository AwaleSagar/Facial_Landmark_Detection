"""Shared fixtures.

Model-backed fixtures are session-scoped because loading ``lbfmodel.yaml``
costs about a second; the fast unit tests use the fakes defined here instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from facial_landmarks import FaceAnalyzer, FaceBox, load_image

from .fakes import FakeDetector, FakeLandmarker

if TYPE_CHECKING:
    from facial_landmarks.types import BGRImage

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory of sample portraits shipped with the repository."""
    if not DATA_DIR.is_dir():  # pragma: no cover - checkout without samples
        pytest.skip("sample data directory is missing")
    return DATA_DIR


@pytest.fixture(scope="session")
def sample_path(data_dir: Path) -> Path:
    """Path to a single sample portrait containing exactly one face."""
    path = data_dir / "image_0.jpg"
    if not path.is_file():  # pragma: no cover
        pytest.skip("sample image is missing")
    return path


@pytest.fixture(scope="session")
def sample_image(sample_path: Path) -> BGRImage:
    """A decoded sample portrait."""
    return load_image(sample_path)


@pytest.fixture
def blank_image() -> BGRImage:
    """A flat grey image that contains no face."""
    return np.full((240, 320, 3), 127, dtype=np.uint8)


@pytest.fixture(scope="session")
def analyzer() -> FaceAnalyzer:
    """A real YuNet + LBF analyzer, built once for the whole session."""
    try:
        return FaceAnalyzer.from_backends("yunet", "lbf")
    except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"model weights unavailable: {exc}")


@pytest.fixture
def box() -> FaceBox:
    """A plain 40x60 box at (10, 20)."""
    return FaceBox(10.0, 20.0, 40.0, 60.0, score=0.75)


@pytest.fixture
def fake_detector(box: FaceBox) -> FakeDetector:
    """A detector that always finds one face."""
    return FakeDetector([box])


@pytest.fixture
def fake_landmarker() -> FakeLandmarker:
    """A landmarker producing 68 deterministic points."""
    return FakeLandmarker()


@pytest.fixture
def image_file(tmp_path: Path, blank_image: BGRImage) -> Path:
    """A small PNG written to a temporary directory."""
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), blank_image)
    return path
