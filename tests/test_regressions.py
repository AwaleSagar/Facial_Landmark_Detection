"""Regression guards for the defects in the original notebook.

Each test first demonstrates that the raw OpenCV behaviour the notebook relied
on really does fail on a current stack, then shows the rebuilt library handling
the same situation. If a future refactor reintroduces one of these patterns,
the corresponding test fails.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from facial_landmarks import (
    FaceAnalyzer,
    FaceBox,
    FaceLandmarks,
    annotate,
    create_detector,
    load_image,
)


class TestImreadFlag:
    """``cv2.imread(path, cv2.COLOR_BGR2RGB)`` passes a conversion code as a flag."""

    def test_the_constant_is_not_an_imread_flag(self):
        # The notebook's call only ever worked by coincidence: COLOR_BGR2RGB is
        # the integer 4, which imread reads as IMREAD_ANYCOLOR.
        assert cv2.COLOR_BGR2RGB == cv2.IMREAD_ANYCOLOR
        assert cv2.COLOR_BGR2RGB != cv2.IMREAD_COLOR

    def test_the_wrong_flag_breaks_on_a_greyscale_source(self, tmp_path):
        """IMREAD_ANYCOLOR keeps a greyscale JPEG two-dimensional."""
        path = tmp_path / "grey.jpg"
        cv2.imwrite(str(path), np.full((64, 64), 128, dtype=np.uint8))
        wrongly_loaded = cv2.imread(str(path), cv2.COLOR_BGR2RGB)
        assert wrongly_loaded is not None
        assert wrongly_loaded.ndim == 2
        # The loader normalises it to three channels instead.
        assert load_image(path).shape == (64, 64, 3)

    def test_loader_uses_the_correct_flag(self, sample_path):
        # Whatever the source, the loader yields 8-bit BGR with three channels.
        image = load_image(sample_path)
        assert image.dtype == np.uint8
        assert image.shape[2] == 3

    def test_loader_does_not_silently_return_none(self, tmp_path):
        # imread returns None for an unreadable file; the notebook then crashed
        # much later with an opaque error. The loader raises at the source.
        bad = tmp_path / "broken.jpg"
        bad.write_bytes(b"\xff\xd8not really a jpeg")
        with pytest.raises(ValueError, match="could not be decoded"):
            load_image(bad)


class TestEmptyDetectionGuard:
    """``if faces == ():`` raises once detectMultiScale returns an ndarray."""

    def test_the_original_comparison_still_raises(self, sample_image):
        cascade = cv2.CascadeClassifier(str(create_detector("haar").model_path))
        grey = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(grey)
        # The cv2 stubs declare Sequence[Sequence[int]], but at runtime this is
        # an ndarray - which is precisely why the notebook's `== ()` blew up.
        assert isinstance(faces, np.ndarray)  # type: ignore[unreachable]
        assert len(faces) > 0  # type: ignore[unreachable]
        with pytest.raises(ValueError, match="broadcast"):
            _ = bool(faces == ())

    @pytest.mark.parametrize("backend", ["haar", "yunet"])
    def test_backends_return_a_plain_list(self, backend, sample_image, blank_image):
        detector = create_detector(backend)
        # Both the found and not-found paths yield a list, so `if faces:` is
        # always safe and no ndarray comparison is ever needed.
        assert isinstance(detector.detect(sample_image), list)
        assert detector.detect(blank_image) == []

    @pytest.mark.parametrize("backend", ["haar", "yunet"])
    def test_no_face_path_returns_a_value(self, backend, blank_image):
        """The notebook's functions fell through to an unbound local here."""
        analysis = FaceAnalyzer.from_backends(backend, None).analyze(blank_image)
        assert analysis is not None
        assert analysis.num_faces == 0


class TestFloatDrawingCoordinates:
    """``cv2.circle`` rejects the float32 points the facemark API returns."""

    def test_opencv_still_rejects_float_centres(self, blank_image):
        point = np.array([10.5, 20.5], dtype=np.float32)
        with pytest.raises(cv2.error):
            cv2.circle(blank_image, (point[0], point[1]), 1, (0, 0, 255), 2)

    def test_drawing_rounds_points_first(self, blank_image):
        marks = FaceLandmarks(
            np.array([[10.5, 20.5], [30.7, 40.2]], dtype=np.float32),
            FaceBox(5, 5, 60, 60),
        )
        assert annotate(blank_image, [marks]).any()

    def test_box_conversion_yields_ints(self):
        assert all(isinstance(v, int) for v in FaceBox(1.7, 2.3, 9.9, 8.1).to_int_xywh())


class TestInPlaceMutation:
    """The notebook drew onto its input, so successive figures accumulated."""

    def test_repeated_annotation_is_stable(self, blank_image):
        marks = FaceLandmarks(np.array([[10.0, 20.0]], dtype=np.float32), FaceBox(5, 5, 60, 60))
        first = annotate(blank_image, [marks])
        second = annotate(blank_image, [marks])
        # Annotating the same source twice must give identical output.
        np.testing.assert_array_equal(first, second)

    def test_source_image_is_never_modified(self, sample_image):
        before = sample_image.copy()
        annotate(sample_image, [], [FaceBox(10, 10, 50, 50)])
        np.testing.assert_array_equal(sample_image, before)
