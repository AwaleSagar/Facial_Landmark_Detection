"""Core geometry types shared by every detector and landmark backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

#: An image in OpenCV's native BGR channel order, shape ``(H, W, 3)``.
BGRImage: TypeAlias = "NDArray[np.uint8]"
#: A single-channel greyscale image, shape ``(H, W)``.
GrayImage: TypeAlias = "NDArray[np.uint8]"
#: An ``(N, 2)`` array of ``(x, y)`` point coordinates.
PointArray: TypeAlias = "NDArray[np.float32]"

#: Number of points produced by the iBUG 300-W 68-point annotation scheme.
NUM_LANDMARKS_68: Final = 68


@dataclass(frozen=True, slots=True)
class FaceBox:
    """An axis-aligned face bounding box in pixel coordinates.

    Coordinates are stored as floats so that a detector's sub-pixel output is
    preserved; use :meth:`to_int_xywh` when an integer rectangle is required.

    Attributes:
        x: Left edge of the box.
        y: Top edge of the box.
        width: Box width in pixels.
        height: Box height in pixels.
        score: Detector confidence in ``[0, 1]``. Backends without a
            meaningful confidence (Haar cascades) report ``1.0``.
        keypoints: Optional coarse keypoints supplied by the detector itself,
            as ``(name, (x, y))`` pairs. YuNet provides five; Haar provides none.
    """

    x: float
    y: float
    width: float
    height: float
    score: float = 1.0
    keypoints: tuple[tuple[str, tuple[float, float]], ...] = field(default=())

    def __post_init__(self) -> None:
        """Reject degenerate boxes early rather than deep inside a backend."""
        if self.width <= 0 or self.height <= 0:
            msg = f"FaceBox requires positive extent, got {self.width}x{self.height}"
            raise ValueError(msg)

    @property
    def right(self) -> float:
        """The right edge (``x + width``)."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """The bottom edge (``y + height``)."""
        return self.y + self.height

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """The box centre as ``(x, y)``."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def to_int_xywh(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` rounded to ints, as OpenCV drawing calls need.

        OpenCV 5 rejects float coordinates outright, which is exactly the bug
        that made the original notebook crash. Funnelling every conversion
        through this method keeps that failure mode from coming back.
        """
        x, y = round(self.x), round(self.y)
        return (x, y, round(self.right) - x, round(self.bottom) - y)

    def clipped_to(self, width: int, height: int) -> FaceBox:
        """Clamp the box to an image of ``width`` x ``height``.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            A new box confined to the image bounds.

        Raises:
            ValueError: If the box lies entirely outside the image.
        """
        left = min(max(self.x, 0.0), float(width))
        top = min(max(self.y, 0.0), float(height))
        right = min(max(self.right, 0.0), float(width))
        bottom = min(max(self.bottom, 0.0), float(height))
        if right <= left or bottom <= top:
            msg = f"box {self} does not intersect a {width}x{height} image"
            raise ValueError(msg)
        return FaceBox(left, top, right - left, bottom - top, self.score, self.keypoints)

    def scaled(self, factor: float) -> FaceBox:
        """Grow or shrink the box about its centre by ``factor``."""
        if factor <= 0:
            msg = f"scale factor must be positive, got {factor}"
            raise ValueError(msg)
        cx, cy = self.center
        w, h = self.width * factor, self.height * factor
        return FaceBox(cx - w / 2, cy - h / 2, w, h, self.score, self.keypoints)

    def remapped(
        self,
        scale_x: float,
        scale_y: float,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ) -> FaceBox:
        """Rescale about the centre, then shift, in units of the current size.

        Used to convert between the differing box conventions of detectors;
        see :data:`facial_landmarks.landmarks.lbf.CONVENTION_TO_HAAR`.

        Args:
            scale_x: Width multiplier.
            scale_y: Height multiplier.
            shift_x: Horizontal shift as a fraction of the original width.
            shift_y: Vertical shift as a fraction of the original height.

        Returns:
            The remapped box, carrying the original score and keypoints.

        Raises:
            ValueError: If either scale is non-positive.
        """
        if scale_x <= 0 or scale_y <= 0:
            msg = f"scales must be positive, got ({scale_x}, {scale_y})"
            raise ValueError(msg)
        cx, cy = self.center
        width = self.width * scale_x
        height = self.height * scale_y
        return FaceBox(
            cx - width / 2 + shift_x * self.width,
            cy - height / 2 + shift_y * self.height,
            width,
            height,
            self.score,
            self.keypoints,
        )

    def iou(self, other: FaceBox) -> float:
        """Intersection-over-union with ``other``, in ``[0, 1]``."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return 0.0
        overlap = (right - left) * (bottom - top)
        return overlap / (self.area + other.area - overlap)


@dataclass(frozen=True, slots=True)
class FaceLandmarks:
    """A set of 2D landmark points localised inside a :class:`FaceBox`.

    The point array is stored read-only so that a caller cannot mutate results
    that another consumer still holds.

    Attributes:
        points: ``(N, 2)`` float32 array of ``(x, y)`` coordinates.
        box: The detection the points were fitted to.
    """

    points: PointArray
    box: FaceBox

    def __post_init__(self) -> None:
        """Normalise the point array's shape, dtype and writability."""
        pts = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        pts.setflags(write=False)
        object.__setattr__(self, "points", pts)

    def __len__(self) -> int:
        """Number of landmark points."""
        return int(self.points.shape[0])

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Iterate over points as ``(x, y)`` float tuples."""
        for x, y in self.points:
            yield (float(x), float(y))

    def to_int_points(self) -> NDArray[np.int32]:
        """Return the points rounded to int32, ready for OpenCV drawing."""
        return np.rint(self.points).astype(np.int32)

    def subset(self, region: str) -> PointArray:
        """Return the points belonging to a named facial region.

        Args:
            region: One of the keys in :data:`FACE_REGIONS_68`.

        Returns:
            An ``(M, 2)`` view of the requested points.

        Raises:
            KeyError: If ``region`` is unknown.
            ValueError: If this landmark set is not the 68-point scheme.
        """
        if len(self) != NUM_LANDMARKS_68:
            msg = f"named regions require the 68-point scheme, got {len(self)} points"
            raise ValueError(msg)
        start, stop = FACE_REGIONS_68[region]
        return self.points[start:stop]


#: Index ranges of the iBUG 300-W 68-point scheme, as ``(start, stop)`` slices.
FACE_REGIONS_68: Final[dict[str, tuple[int, int]]] = {
    "jaw": (0, 17),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "nose_bridge": (27, 31),
    "nose_tip": (31, 36),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "outer_lips": (48, 60),
    "inner_lips": (60, 68),
}
