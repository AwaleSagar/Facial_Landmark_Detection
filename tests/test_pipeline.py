"""Tests for the detect-then-fit pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from facial_landmarks import FaceAnalysis, FaceAnalyzer, FaceBox, YuNetDetector

from .fakes import FakeDetector


class TestFaceAnalysis:
    def test_empty_is_falsy(self):
        analysis = FaceAnalysis()
        assert not analysis
        assert analysis.num_faces == 0

    def test_populated_is_truthy(self, box: FaceBox):
        assert FaceAnalysis(boxes=[box])

    def test_draw_on_does_not_mutate(self, box: FaceBox, blank_image):
        before = blank_image.copy()
        FaceAnalysis(boxes=[box]).draw_on(blank_image)
        np.testing.assert_array_equal(blank_image, before)


class TestWithFakes:
    def test_detects_and_fits(self, blank_image, fake_detector, fake_landmarker):
        analysis = FaceAnalyzer(fake_detector, fake_landmarker).analyze(blank_image)
        assert analysis.num_faces == 1
        assert len(analysis.landmarks) == 1
        assert len(analysis.landmarks[0]) == 68

    def test_no_detections_skips_the_landmarker(self, blank_image, fake_landmarker):
        detector = FakeDetector([])
        analysis = FaceAnalyzer(detector, fake_landmarker).analyze(blank_image)
        assert analysis.num_faces == 0
        assert analysis.landmarks == []
        # Fitting an empty box list is pointless work; it must be skipped.
        assert fake_landmarker.calls == 0

    def test_landmarker_is_optional(self, blank_image, fake_detector):
        analysis = FaceAnalyzer(fake_detector).analyze(blank_image)
        assert analysis.num_faces == 1
        assert analysis.landmarks == []

    def test_description(self, fake_detector, fake_landmarker):
        assert FaceAnalyzer(fake_detector, fake_landmarker).description == (
            "detector=fake landmarker=fake"
        )
        assert "landmarker=none" in FaceAnalyzer(fake_detector).description

    def test_rejects_bad_detector(self):
        with pytest.raises(TypeError, match="FaceDetector protocol"):
            FaceAnalyzer("not a detector")  # type: ignore[arg-type]

    def test_rejects_bad_landmarker(self, fake_detector):
        with pytest.raises(TypeError, match="Landmarker protocol"):
            FaceAnalyzer(fake_detector, "not a landmarker")  # type: ignore[arg-type]


class TestFromBackends:
    def test_passes_detector_convention_to_landmarker(self):
        analyzer = FaceAnalyzer.from_backends("yunet", "lbf")
        assert analyzer.landmarker is not None
        assert analyzer.landmarker.box_convention == "yunet"

    def test_haar_path_uses_haar_convention(self):
        analyzer = FaceAnalyzer.from_backends("haar", "lbf")
        assert analyzer.landmarker is not None
        assert analyzer.landmarker.box_convention == "haar"

    def test_landmarker_none_is_honoured(self):
        assert FaceAnalyzer.from_backends("yunet", None).landmarker is None

    def test_detector_kwargs_are_forwarded(self):
        analyzer = FaceAnalyzer.from_backends("yunet", None, score_threshold=0.95)
        assert isinstance(analyzer.detector, YuNetDetector)
        assert analyzer.detector.score_threshold == 0.95


@pytest.mark.slow
class TestEndToEnd:
    def test_analyze_path_returns_image_and_analysis(self, analyzer, sample_path):
        image, analysis = analyzer.analyze_path(sample_path)
        assert image.ndim == 3
        assert analysis.num_faces == 1
        assert len(analysis.landmarks[0]) == 68

    def test_landmarks_lie_within_the_detected_box(self, analyzer, sample_path):
        _, analysis = analyzer.analyze_path(sample_path)
        marks, face = analysis.landmarks[0], analysis.boxes[0]
        # Allow a margin: the jaw contour legitimately hugs the box edge.
        margin = 0.5 * max(face.width, face.height)
        assert marks.points[:, 0].min() > face.x - margin
        assert marks.points[:, 0].max() < face.right + margin

    def test_blank_image_produces_an_empty_analysis(self, analyzer, blank_image):
        analysis = analyzer.analyze(blank_image)
        assert not analysis
        assert analysis.boxes == []
        assert analysis.landmarks == []

    def test_annotated_output_differs_from_input(self, analyzer, sample_path):
        image, analysis = analyzer.analyze_path(sample_path)
        assert (analysis.draw_on(image) != image).any()

    def test_recall_over_the_sample_set(self, analyzer, data_dir):
        """A regression guard on end-to-end quality across all 113 portraits."""
        paths = sorted(data_dir.glob("*.jpg"))
        hits = sum(bool(analyzer.analyze_path(p)[1]) for p in paths)
        assert hits / len(paths) > 0.90
