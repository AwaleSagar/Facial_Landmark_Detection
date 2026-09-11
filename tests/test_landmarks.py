"""Tests for the LBF landmark backend and box-convention adaptation."""

from __future__ import annotations

import numpy as np
import pytest

from facial_landmarks import FaceBox, Landmarker, create_detector, create_landmarker
from facial_landmarks.detectors import DETECTOR_BACKENDS
from facial_landmarks.landmarks.lbf import CONVENTION_TO_HAAR
from facial_landmarks.types import NUM_LANDMARKS_68


@pytest.fixture(scope="session")
def landmarker():
    try:
        return create_landmarker("lbf")
    except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"LBF model unavailable: {exc}")


class TestFactory:
    def test_unknown_backend(self):
        with pytest.raises(KeyError, match="unknown landmarker backend"):
            create_landmarker("nope")

    def test_satisfies_protocol(self, landmarker):
        assert isinstance(landmarker, Landmarker)

    def test_reports_68_points(self, landmarker):
        assert landmarker.num_points == NUM_LANDMARKS_68

    def test_rejects_unknown_convention(self):
        with pytest.raises(KeyError, match="unknown box convention"):
            create_landmarker("lbf", box_convention="sideways")


class TestConventionTable:
    def test_haar_entry_is_the_identity(self):
        assert CONVENTION_TO_HAAR["haar"] == (1.0, 1.0, 0.0, 0.0)

    def test_every_detector_backend_has_an_entry(self):
        for backend in DETECTOR_BACKENDS:
            assert create_detector(backend).box_convention in CONVENTION_TO_HAAR

    def test_yunet_entry_widens_and_shortens(self):
        scale_x, scale_y, _, _ = CONVENTION_TO_HAAR["yunet"]
        assert scale_x > 1.0
        assert scale_y < 1.0


class TestFit:
    def test_empty_faces_returns_empty(self, landmarker, sample_image):
        assert landmarker.fit(sample_image, []) == []

    @pytest.mark.slow
    def test_produces_68_points_inside_the_image(self, landmarker, sample_image):
        faces = create_detector("haar").detect(sample_image)
        results = landmarker.fit(sample_image, faces)
        assert len(results) == len(faces)
        height, width = sample_image.shape[:2]
        for marks in results:
            assert len(marks) == NUM_LANDMARKS_68
            assert marks.points[:, 0].min() > -width
            assert marks.points[:, 0].max() < 2 * width
            assert marks.points[:, 1].max() < 2 * height

    @pytest.mark.slow
    def test_result_carries_the_original_detector_box(self, sample_image):
        """Adaptation is internal; callers must get the box they passed in."""
        adapting = create_landmarker("lbf", box_convention="yunet")
        faces = create_detector("yunet").detect(sample_image)
        results = adapting.fit(sample_image, faces)
        assert results[0].box == faces[0]

    @pytest.mark.slow
    def test_adaptation_changes_the_fit(self, sample_image):
        """The yunet transform must actually reach the regressor."""
        faces = create_detector("yunet").detect(sample_image)
        naive = create_landmarker("lbf", box_convention="haar").fit(sample_image, faces)
        adapted = create_landmarker("lbf", box_convention="yunet").fit(sample_image, faces)
        assert not np.allclose(naive[0].points, adapted[0].points)

    @pytest.mark.slow
    def test_handles_a_box_that_widens_past_the_frame_edge(self, landmarker, sample_image):
        """A box at the edge widens out of frame and must be clipped, not crash."""
        adapting = create_landmarker("lbf", box_convention="yunet")
        height, width = sample_image.shape[:2]
        edge = FaceBox(width - 60, height - 60, 58, 58)
        assert len(adapting.fit(sample_image, [edge])) == 1
