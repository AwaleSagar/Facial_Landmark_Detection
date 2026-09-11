"""Tests for the geometry types."""

from __future__ import annotations

import numpy as np
import pytest

from facial_landmarks import FACE_REGIONS_68, NUM_LANDMARKS_68, FaceBox, FaceLandmarks


class TestFaceBox:
    def test_derived_geometry(self, box: FaceBox):
        assert box.right == 50.0
        assert box.bottom == 80.0
        assert box.area == 2400.0
        assert box.center == (30.0, 50.0)

    @pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0), (-5, 10), (10, -5)])
    def test_rejects_degenerate_extent(self, width, height):
        with pytest.raises(ValueError, match="positive extent"):
            FaceBox(0, 0, width, height)

    def test_is_immutable(self, box: FaceBox):
        with pytest.raises(AttributeError):
            box.x = 99  # type: ignore[misc]

    def test_to_int_xywh_rounds_and_preserves_extent(self):
        # A float box must not lose a pixel of width when rounded.
        box = FaceBox(10.6, 20.4, 30.9, 40.2)
        x, y, w, h = FaceBox(10.6, 20.4, 30.9, 40.2).to_int_xywh()
        assert (x, y) == (11, 20)
        assert x + w == round(box.right)
        assert y + h == round(box.bottom)
        assert all(isinstance(v, int) for v in (x, y, w, h))

    def test_clipped_to_confines_box(self):
        clipped = FaceBox(-10, -10, 50, 50).clipped_to(30, 30)
        assert (clipped.x, clipped.y) == (0.0, 0.0)
        assert (clipped.right, clipped.bottom) == (30.0, 30.0)

    def test_clipped_to_rejects_disjoint_box(self):
        with pytest.raises(ValueError, match="does not intersect"):
            FaceBox(200, 200, 10, 10).clipped_to(50, 50)

    def test_clipped_keeps_score_and_keypoints(self):
        box = FaceBox(0, 0, 10, 10, 0.5, (("eye", (1.0, 2.0)),))
        clipped = box.clipped_to(5, 5)
        assert clipped.score == 0.5
        assert clipped.keypoints == (("eye", (1.0, 2.0)),)

    def test_scaled_keeps_centre(self, box: FaceBox):
        grown = box.scaled(2.0)
        assert grown.center == box.center
        assert grown.area == pytest.approx(box.area * 4)

    @pytest.mark.parametrize("factor", [0.0, -1.0])
    def test_scaled_rejects_non_positive(self, box: FaceBox, factor):
        with pytest.raises(ValueError, match="must be positive"):
            box.scaled(factor)

    def test_remapped_applies_scale_and_shift(self):
        box = FaceBox(0, 0, 100, 100)
        out = box.remapped(1.2, 0.9, 0.0, -0.05)
        assert out.width == pytest.approx(120)
        assert out.height == pytest.approx(90)
        # Centre moves only by the requested shift.
        assert out.center == pytest.approx((50.0, 45.0))

    def test_remapped_identity_is_a_no_op(self, box: FaceBox):
        assert box.remapped(1.0, 1.0) == box

    @pytest.mark.parametrize(("sx", "sy"), [(0, 1), (1, 0), (-1, 1)])
    def test_remapped_rejects_bad_scale(self, box: FaceBox, sx, sy):
        with pytest.raises(ValueError, match="must be positive"):
            box.remapped(sx, sy)

    def test_iou_bounds(self, box: FaceBox):
        assert box.iou(box) == pytest.approx(1.0)
        assert box.iou(FaceBox(1000, 1000, 5, 5)) == 0.0

    def test_iou_partial_overlap(self):
        a = FaceBox(0, 0, 10, 10)
        b = FaceBox(5, 0, 10, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_iou_is_symmetric(self):
        a, b = FaceBox(0, 0, 10, 10), FaceBox(4, 4, 10, 10)
        assert a.iou(b) == pytest.approx(b.iou(a))


class TestFaceLandmarks:
    def test_reshapes_opencv_output(self, box: FaceBox):
        # The facemark API hands back (1, 68, 2); it must flatten to (68, 2).
        marks = FaceLandmarks(np.zeros((1, NUM_LANDMARKS_68, 2), dtype=np.float32), box)
        assert marks.points.shape == (NUM_LANDMARKS_68, 2)
        assert len(marks) == NUM_LANDMARKS_68

    def test_points_are_read_only(self, box: FaceBox):
        marks = FaceLandmarks(np.zeros((4, 2), dtype=np.float32), box)
        with pytest.raises(ValueError, match="read-only"):
            marks.points[0, 0] = 1.0

    def test_to_int_points_is_int32(self, box: FaceBox):
        marks = FaceLandmarks(np.array([[1.6, 2.4]], dtype=np.float32), box)
        assert marks.to_int_points().dtype == np.int32
        assert marks.to_int_points().tolist() == [[2, 2]]

    def test_iteration_yields_float_pairs(self, box: FaceBox):
        marks = FaceLandmarks(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), box)
        assert list(marks) == [(1.0, 2.0), (3.0, 4.0)]

    def test_regions_partition_all_68_points(self):
        covered = sorted(
            index for start, stop in FACE_REGIONS_68.values() for index in range(start, stop)
        )
        assert covered == list(range(NUM_LANDMARKS_68))

    def test_subset_returns_named_region(self, box: FaceBox):
        marks = FaceLandmarks(np.arange(NUM_LANDMARKS_68 * 2, dtype=np.float32), box)
        assert marks.subset("jaw").shape == (17, 2)
        assert marks.subset("left_eye").shape == (6, 2)

    def test_subset_rejects_unknown_region(self, box: FaceBox):
        marks = FaceLandmarks(np.zeros((NUM_LANDMARKS_68, 2), dtype=np.float32), box)
        with pytest.raises(KeyError):
            marks.subset("nostril")

    def test_subset_requires_68_points(self, box: FaceBox):
        marks = FaceLandmarks(np.zeros((5, 2), dtype=np.float32), box)
        with pytest.raises(ValueError, match="68-point scheme"):
            marks.subset("jaw")
