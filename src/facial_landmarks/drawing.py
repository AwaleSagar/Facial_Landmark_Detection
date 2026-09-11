"""Rendering of detections and landmarks onto images.

Every function here returns a new array. The original notebook drew onto the
image it was handed, so each figure silently accumulated the annotations of
the figure before it; copying once per call removes that whole class of bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from facial_landmarks.types import FACE_REGIONS_68, NUM_LANDMARKS_68

if TYPE_CHECKING:
    from collections.abc import Sequence

    from facial_landmarks.types import BGRImage, FaceBox, FaceLandmarks

__all__ = ["REGION_COLORS", "annotate", "draw_boxes", "draw_landmarks"]

#: BGR colour per 68-point facial region, chosen to stay distinguishable
#: against skin tones and in greyscale print.
REGION_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "jaw": (255, 176, 0),
    "right_eyebrow": (0, 200, 255),
    "left_eyebrow": (0, 200, 255),
    "nose_bridge": (120, 255, 120),
    "nose_tip": (120, 255, 120),
    "right_eye": (255, 80, 200),
    "left_eye": (255, 80, 200),
    "outer_lips": (60, 60, 255),
    "inner_lips": (160, 160, 255),
}
_DEFAULT_POINT_COLOR: Final = (0, 0, 255)
_BOX_COLOR: Final = (0, 255, 0)
_TEXT_COLOR: Final = (20, 20, 20)


def _scale_for(image: BGRImage) -> float:
    """Return a stroke scale so annotations read the same on any image size."""
    return max(1.0, float(min(image.shape[:2])) / 400.0)


def draw_boxes(
    image: BGRImage,
    boxes: Sequence[FaceBox],
    *,
    color: tuple[int, int, int] = _BOX_COLOR,
    show_score: bool = True,
) -> BGRImage:
    """Draw face bounding boxes on a copy of ``image``.

    Args:
        image: The BGR image to annotate.
        boxes: Boxes to draw.
        color: BGR stroke colour.
        show_score: Whether to label each box with its confidence.

    Returns:
        An annotated copy of ``image``.
    """
    canvas: BGRImage = image.copy()
    scale = _scale_for(image)
    thickness = max(1, round(2 * scale))

    for box in boxes:
        x, y, w, h = box.to_int_xywh()
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        if not show_score:
            continue
        label = f"{box.score:.2f}"
        font_scale = 0.5 * scale
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Keep the label inside the frame when the box touches the top edge.
        top = y - text_h - baseline
        if top < 0:
            top = y + h
        cv2.rectangle(canvas, (x, top), (x + text_w, top + text_h + baseline), color, cv2.FILLED)
        cv2.putText(
            canvas,
            label,
            (x, top + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def draw_landmarks(
    image: BGRImage,
    landmarks: Sequence[FaceLandmarks],
    *,
    color_by_region: bool = True,
    radius: int | None = None,
) -> BGRImage:
    """Draw landmark points on a copy of ``image``.

    Args:
        image: The BGR image to annotate.
        landmarks: Landmark sets to draw.
        color_by_region: Colour the 68-point scheme per facial region. Ignored
            for other point counts, which are drawn in a single colour.
        radius: Point radius in pixels; scaled from the image size when omitted.

    Returns:
        An annotated copy of ``image``.
    """
    canvas: BGRImage = image.copy()
    scale = _scale_for(image)
    point_radius = radius if radius is not None else max(1, round(1.5 * scale))

    for landmark in landmarks:
        points = landmark.to_int_points()
        use_regions = color_by_region and len(landmark) == NUM_LANDMARKS_68
        if use_regions:
            for region, (start, stop) in FACE_REGIONS_68.items():
                colour = REGION_COLORS[region]
                for x, y in points[start:stop]:
                    cv2.circle(canvas, (int(x), int(y)), point_radius, colour, cv2.FILLED)
        else:
            for x, y in points:
                cv2.circle(canvas, (int(x), int(y)), point_radius, _DEFAULT_POINT_COLOR, cv2.FILLED)
    return canvas


def annotate(
    image: BGRImage,
    landmarks: Sequence[FaceLandmarks] = (),
    boxes: Sequence[FaceBox] | None = None,
    *,
    show_boxes: bool = True,
    show_scores: bool = True,
    color_by_region: bool = True,
) -> BGRImage:
    """Draw boxes and landmarks in one pass.

    Args:
        image: The BGR image to annotate.
        landmarks: Landmark sets to draw.
        boxes: Boxes to draw. Defaults to the box attached to each landmark set.
        show_boxes: Whether to draw bounding boxes at all.
        show_scores: Whether to label boxes with confidence.
        color_by_region: Colour 68-point landmarks per facial region.

    Returns:
        An annotated copy of ``image``.
    """
    canvas: BGRImage = np.ascontiguousarray(image)
    if boxes is None:
        boxes = [landmark.box for landmark in landmarks]
    if show_boxes and boxes:
        canvas = draw_boxes(canvas, boxes, show_score=show_scores)
    if landmarks:
        canvas = draw_landmarks(canvas, landmarks, color_by_region=color_by_region)
    return canvas if canvas is not image else image.copy()
