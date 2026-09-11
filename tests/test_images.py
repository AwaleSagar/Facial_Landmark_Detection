"""Tests for image loading, saving and contact sheets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from facial_landmarks import contact_sheet, load_image, save_image, to_rgb
from facial_landmarks.images import iter_images


class TestLoadImage:
    def test_returns_three_channel_uint8(self, sample_path: Path):
        image = load_image(sample_path)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no image at"):
            load_image(tmp_path / "absent.jpg")

    def test_undecodable_file(self, tmp_path: Path):
        path = tmp_path / "not-an-image.jpg"
        path.write_text("plain text, not JPEG data")
        with pytest.raises(ValueError, match="could not be decoded"):
            load_image(path)

    def test_greyscale_source_is_widened_to_bgr(self, tmp_path: Path):
        path = tmp_path / "grey.png"
        cv2.imwrite(str(path), np.full((16, 16), 200, dtype=np.uint8))
        assert load_image(path).shape == (16, 16, 3)


class TestSaveImage:
    def test_roundtrip(self, tmp_path: Path, blank_image):
        path = save_image(blank_image, tmp_path / "nested" / "out.png")
        assert path.is_file()
        np.testing.assert_array_equal(load_image(path), blank_image)

    def test_creates_parent_directories(self, tmp_path: Path, blank_image):
        save_image(blank_image, tmp_path / "a" / "b" / "c.png")
        assert (tmp_path / "a" / "b" / "c.png").is_file()

    def test_unknown_extension_raises(self, tmp_path: Path, blank_image):
        with pytest.raises(OSError, match="failed to write"):
            save_image(blank_image, tmp_path / "out.unsupported")


class TestToRgb:
    def test_swaps_channels(self):
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0] = (10, 20, 30)
        assert to_rgb(bgr)[0, 0].tolist() == [30, 20, 10]


class TestIterImages:
    def test_expands_directory(self, data_dir: Path):
        paths = iter_images([data_dir])
        assert len(paths) > 100
        assert paths == sorted(paths)

    def test_accepts_individual_files(self, sample_path: Path):
        assert iter_images([sample_path]) == [sample_path]

    def test_deduplicates(self, sample_path: Path):
        assert iter_images([sample_path, sample_path]) == [sample_path]

    def test_ignores_non_images(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("hello")
        encoded = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1]
        (tmp_path / "pic.png").write_bytes(encoded.tobytes())
        assert [p.name for p in iter_images([tmp_path])] == ["pic.png"]

    def test_recursive_flag(self, tmp_path: Path):
        nested = tmp_path / "deep"
        nested.mkdir()
        cv2.imwrite(str(nested / "inner.png"), np.zeros((4, 4, 3), np.uint8))
        assert iter_images([tmp_path]) == []
        assert len(iter_images([tmp_path], recursive=True)) == 1

    def test_missing_path(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no such file or directory"):
            iter_images([tmp_path / "absent"])


class TestContactSheet:
    def test_layout_dimensions(self):
        images = [np.full((50, 50, 3), i, dtype=np.uint8) for i in range(5)]
        sheet = contact_sheet(images, columns=2, tile=(64, 64), padding=4)
        # 5 images over 2 columns needs 3 rows.
        assert sheet.shape == (3 * 64 + 4 * 4, 2 * 64 + 3 * 4, 3)

    def test_preserves_aspect_ratio(self):
        # A wide image must be letterboxed, not stretched to a square tile.
        wide = np.full((20, 100, 3), 255, dtype=np.uint8)
        sheet = contact_sheet([wide], columns=1, tile=(100, 100), padding=0, background=(0, 0, 0))
        filled_rows = np.where(sheet.any(axis=(1, 2)))[0]
        assert len(filled_rows) == 20

    def test_columns_clamped_to_image_count(self):
        sheet = contact_sheet(
            [np.zeros((10, 10, 3), np.uint8)], columns=8, tile=(20, 20), padding=0
        )
        assert sheet.shape[1] == 20

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="at least one image"):
            contact_sheet([])

    @pytest.mark.parametrize("columns", [0, -1])
    def test_rejects_bad_columns(self, columns):
        with pytest.raises(ValueError, match="columns must be positive"):
            contact_sheet([np.zeros((4, 4, 3), np.uint8)], columns=columns)

    def test_rejects_bad_tile(self):
        with pytest.raises(ValueError, match="tile size must be positive"):
            contact_sheet([np.zeros((4, 4, 3), np.uint8)], tile=(0, 10))
