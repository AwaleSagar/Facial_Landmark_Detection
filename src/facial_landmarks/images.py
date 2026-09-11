"""Image loading, saving and contact-sheet helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from facial_landmarks.types import BGRImage

__all__ = ["IMAGE_SUFFIXES", "contact_sheet", "iter_images", "load_image", "save_image", "to_rgb"]

logger = logging.getLogger(__name__)

#: File extensions treated as images when expanding a directory.
IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def load_image(path: Path | str) -> BGRImage:
    """Read an image from disk as 8-bit BGR.

    The original notebook called ``cv2.imread(path, cv2.COLOR_BGR2RGB)``, which
    passes a colour-conversion constant where an ``IMREAD_*`` flag belongs.
    ``cv2.COLOR_BGR2RGB`` is the integer 4, which imread reads as
    ``IMREAD_ANYCOLOR``: the image comes back in whatever channel count the file
    happens to carry, so a greyscale JPEG decodes to a 2D array and every later
    ``shape[2]`` or colour conversion falls over. ``IMREAD_COLOR`` is the
    correct flag and always yields three 8-bit BGR channels.

    Args:
        path: Path to the image file.

    Returns:
        The decoded image, shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file exists but cannot be decoded.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"no image at {file_path}"
        raise FileNotFoundError(msg)
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"{file_path} exists but could not be decoded as an image"
        raise ValueError(msg)
    return cast("BGRImage", np.ascontiguousarray(image, dtype=np.uint8))


def save_image(image: BGRImage, path: Path | str) -> Path:
    """Write a BGR image to disk, creating parent directories as needed.

    Args:
        image: The BGR image to write.
        path: Destination path; the suffix selects the encoder.

    Returns:
        The path written to.

    Raises:
        OSError: If OpenCV fails to encode or write the file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # imwrite signals failure two different ways: it returns False for a write
    # error, but raises cv2.error when no encoder matches the suffix. Normalise
    # both into OSError so callers have one thing to catch.
    try:
        written = cv2.imwrite(str(out_path), image)
    except cv2.error as exc:
        msg = f"failed to write image to {out_path}: {exc}"
        raise OSError(msg) from exc
    if not written:
        msg = f"failed to write image to {out_path}"
        raise OSError(msg)
    return out_path


def to_rgb(image: BGRImage) -> BGRImage:
    """Convert BGR to RGB, e.g. before handing an image to matplotlib."""
    return cast("BGRImage", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def iter_images(paths: Iterable[Path | str], *, recursive: bool = False) -> list[Path]:
    """Expand files and directories into a sorted list of image paths.

    Args:
        paths: Files and/or directories to expand.
        recursive: Recurse into sub-directories.

    Returns:
        Sorted, de-duplicated image paths.

    Raises:
        FileNotFoundError: If any entry does not exist.
    """
    found: set[Path] = set()
    for entry in paths:
        path = Path(entry)
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            found.update(
                child
                for child in path.glob(pattern)
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            msg = f"no such file or directory: {path}"
            raise FileNotFoundError(msg)
    return sorted(found)


def contact_sheet(
    images: Sequence[BGRImage],
    *,
    columns: int = 4,
    tile: tuple[int, int] = (256, 256),
    padding: int = 8,
    background: tuple[int, int, int] = (32, 32, 32),
) -> BGRImage:
    """Tile images into a single grid, preserving each image's aspect ratio.

    This replaces the notebook's matplotlib subplot loops with something that
    writes a plain image file, so it works headless and in CI.

    Args:
        images: Images to tile. Must not be empty.
        columns: Number of grid columns.
        tile: ``(width, height)`` of each cell in pixels.
        padding: Gap between cells in pixels.
        background: BGR fill colour.

    Returns:
        The assembled grid image.

    Raises:
        ValueError: If ``images`` is empty or the layout is degenerate.
    """
    if not images:
        msg = "contact_sheet requires at least one image"
        raise ValueError(msg)
    if columns < 1:
        msg = f"columns must be positive, got {columns}"
        raise ValueError(msg)
    tile_w, tile_h = tile
    if tile_w < 1 or tile_h < 1:
        msg = f"tile size must be positive, got {tile}"
        raise ValueError(msg)

    columns = min(columns, len(images))
    rows = -(-len(images) // columns)  # ceiling division
    sheet_w = columns * tile_w + (columns + 1) * padding
    sheet_h = rows * tile_h + (rows + 1) * padding
    sheet: BGRImage = np.full((sheet_h, sheet_w, 3), background, dtype=np.uint8)

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        height, width = image.shape[:2]
        scale = min(tile_w / width, tile_h / height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        # INTER_AREA is the right filter for downscaling; most tiles shrink.
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        cell_x = padding + column * (tile_w + padding)
        cell_y = padding + row * (tile_h + padding)
        offset_x = cell_x + (tile_w - new_w) // 2
        offset_y = cell_y + (tile_h - new_h) // 2
        sheet[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return sheet
