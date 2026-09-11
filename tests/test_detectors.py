"""Tests for the detector backends."""

from __future__ import annotations

import numpy as np
import pytest

from facial_landmarks import FaceDetector, create_detector
from facial_landmarks.assets import resolve_asset
from facial_landmarks.detectors import DETECTOR_BACKENDS
from facial_landmarks.detectors.base import ensure_bgr
from facial_landmarks.landmarks.lbf import CONVENTION_TO_HAAR

ALL_BACKENDS = sorted(DETECTOR_BACKENDS)


class TestEnsureBgr:
    def test_accepts_valid_image(self, blank_image):
        assert ensure_bgr(blank_image).shape == blank_image.shape

    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="expected a numpy array"):
            ensure_bgr([[0, 0, 0]])

    @pytest.mark.parametrize("shape", [(10, 10), (10, 10, 1), (10, 10, 4)])
    def test_rejects_wrong_channel_count(self, shape):
        with pytest.raises(ValueError, match="shape"):
            ensure_bgr(np.zeros(shape, dtype=np.uint8))

    def test_rejects_wrong_dtype(self):
        with pytest.raises(ValueError, match="dtype uint8"):
            ensure_bgr(np.zeros((10, 10, 3), dtype=np.float32))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            ensure_bgr(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_returns_contiguous(self):
        view = np.zeros((10, 20, 3), dtype=np.uint8)[:, ::2]
        assert ensure_bgr(view).flags["C_CONTIGUOUS"]


class TestFactory:
    def test_unknown_backend(self):
        with pytest.raises(KeyError, match="unknown detector backend"):
            create_detector("nope")

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_satisfies_protocol(self, backend):
        assert isinstance(create_detector(backend), FaceDetector)

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_declares_a_known_convention(self, backend):
        assert create_detector(backend).box_convention in CONVENTION_TO_HAAR


@pytest.mark.slow
class TestDetection:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_finds_the_face_in_a_portrait(self, backend, sample_image):
        faces = create_detector(backend).detect(sample_image)
        assert len(faces) >= 1
        height, width = sample_image.shape[:2]
        for face in faces:
            assert 0 <= face.x < face.right <= width
            assert 0 <= face.y < face.bottom <= height
            assert 0.0 < face.score <= 1.0

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_blank_image_yields_empty_list(self, backend, blank_image):
        # A list, never None - callers must not need a falsy-vs-None guard.
        assert create_detector(backend).detect(blank_image) == []

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_is_reusable_across_images(self, backend, sample_image, blank_image):
        detector = create_detector(backend)
        first = detector.detect(sample_image)
        detector.detect(blank_image)
        # Differently sized inputs must not corrupt detector state.
        assert len(detector.detect(sample_image)) == len(first)

    def test_yunet_reports_five_keypoints(self, sample_image):
        faces = create_detector("yunet").detect(sample_image)
        names = [name for name, _ in faces[0].keypoints]
        assert names == [
            "right_eye",
            "left_eye",
            "nose_tip",
            "right_mouth_corner",
            "left_mouth_corner",
        ]

    def test_yunet_sorts_by_descending_score(self, sample_image):
        scores = [f.score for f in create_detector("yunet").detect(sample_image)]
        assert scores == sorted(scores, reverse=True)

    def test_haar_reports_no_keypoints(self, sample_image):
        assert create_detector("haar").detect(sample_image)[0].keypoints == ()


class TestConfiguration:
    def test_yunet_rejects_bad_threshold(self):
        with pytest.raises(ValueError, match="score_threshold"):
            create_detector("yunet", score_threshold=0.0)

    def test_haar_rejects_bad_scale_factor(self):
        with pytest.raises(ValueError, match="scale_factor"):
            create_detector("haar", scale_factor=1.0)

    def test_explicit_model_path_is_honoured(self):
        path = resolve_asset("yunet")
        assert create_detector("yunet", model_path=path).model_path == path
