"""Tests for annotation rendering."""

from __future__ import annotations

import numpy as np
import pytest

from facial_landmarks import FaceBox, FaceLandmarks, annotate, draw_boxes, draw_landmarks
from facial_landmarks.drawing import REGION_COLORS
from facial_landmarks.types import FACE_REGIONS_68


@pytest.fixture
def canvas():
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def landmarks_68(box: FaceBox):
    points = np.column_stack(
        [
            np.linspace(box.x + 1, box.right - 1, 68, dtype=np.float32),
            np.linspace(box.y + 1, box.bottom - 1, 68, dtype=np.float32),
        ]
    )
    return FaceLandmarks(points, box)


class TestDoesNotMutateInput:
    """The original notebook drew in place, so figures bled into each other."""

    def test_draw_boxes_copies(self, canvas, box):
        before = canvas.copy()
        draw_boxes(canvas, [box])
        np.testing.assert_array_equal(canvas, before)

    def test_draw_landmarks_copies(self, canvas, landmarks_68):
        before = canvas.copy()
        draw_landmarks(canvas, [landmarks_68])
        np.testing.assert_array_equal(canvas, before)

    def test_annotate_copies(self, canvas, landmarks_68):
        before = canvas.copy()
        annotate(canvas, [landmarks_68])
        np.testing.assert_array_equal(canvas, before)

    def test_annotate_returns_a_copy_even_with_nothing_to_draw(self, canvas):
        out = annotate(canvas, [], [])
        assert out is not canvas
        np.testing.assert_array_equal(out, canvas)


class TestDrawBoxes:
    def test_draws_something(self, canvas, box):
        assert draw_boxes(canvas, [box]).any()

    def test_score_label_can_be_suppressed(self, canvas, box):
        with_label = draw_boxes(canvas, [box], show_score=True)
        without = draw_boxes(canvas, [box], show_score=False)
        assert with_label.sum() > without.sum()

    def test_label_stays_in_frame_for_a_box_at_the_top_edge(self, canvas):
        # A label drawn above y=0 would be clipped away entirely.
        top_box = FaceBox(10, 0, 50, 50, score=0.9)
        assert draw_boxes(canvas, [top_box], show_score=True).any()

    def test_empty_input_is_a_no_op(self, canvas):
        np.testing.assert_array_equal(draw_boxes(canvas, []), canvas)


class TestDrawLandmarks:
    def test_region_colouring_uses_multiple_colours(self, canvas, landmarks_68):
        out = draw_landmarks(canvas, [landmarks_68], color_by_region=True)
        colours = {tuple(px) for px in out.reshape(-1, 3) if tuple(px) != (0, 0, 0)}
        assert len(colours) > 1

    def test_single_colour_mode(self, canvas, landmarks_68):
        out = draw_landmarks(canvas, [landmarks_68], color_by_region=False)
        colours = {tuple(px) for px in out.reshape(-1, 3) if tuple(px) != (0, 0, 0)}
        assert colours == {(0, 0, 255)}

    def test_non_68_point_sets_fall_back_to_one_colour(self, canvas, box):
        marks = FaceLandmarks(np.array([[20.0, 30.0], [25.0, 35.0]], dtype=np.float32), box)
        out = draw_landmarks(canvas, [marks], color_by_region=True)
        colours = {tuple(px) for px in out.reshape(-1, 3) if tuple(px) != (0, 0, 0)}
        assert colours == {(0, 0, 255)}

    def test_float_coordinates_do_not_raise(self, canvas, box):
        """OpenCV 5 rejects float centres; every point must be rounded first."""
        marks = FaceLandmarks(np.array([[10.7, 20.3]], dtype=np.float32), box)
        assert draw_landmarks(canvas, [marks]).any()

    def test_every_region_has_a_colour(self):
        assert set(REGION_COLORS) == set(FACE_REGIONS_68)


class TestAnnotate:
    def test_uses_boxes_attached_to_landmarks(self, canvas, landmarks_68):
        out = annotate(canvas, [landmarks_68])
        assert out.any()

    def test_show_boxes_false_omits_rectangle(self, canvas, landmarks_68):
        boxed = annotate(canvas, [landmarks_68], show_boxes=True)
        plain = annotate(canvas, [landmarks_68], show_boxes=False)
        assert boxed.sum() > plain.sum()
