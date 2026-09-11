"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from facial_landmarks.cli import EXIT_ERROR, EXIT_NO_FACES, EXIT_OK, build_parser, main


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_rejects_unknown_detector(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["detect", "x.jpg", "--detector", "bogus"])

    def test_defaults(self):
        args = build_parser().parse_args(["detect", "x.jpg"])
        assert args.detector == "yunet"
        assert args.landmarker == "lbf"
        assert args.offline is False

    def test_landmarker_none_is_accepted(self):
        args = build_parser().parse_args(["detect", "x.jpg", "--landmarker", "none"])
        assert args.landmarker == "none"

    def test_verbosity_counts(self):
        assert build_parser().parse_args(["-vv", "models"]).verbose == 2


class TestDetectCommand:
    @pytest.mark.slow
    def test_writes_annotated_output(self, tmp_path: Path, sample_path: Path, capsys):
        code = main(["detect", str(sample_path), "-o", str(tmp_path)])
        assert code == EXIT_OK
        assert (tmp_path / f"{sample_path.stem}_annotated.png").is_file()
        assert "1 face(s)" in capsys.readouterr().out

    @pytest.mark.slow
    def test_json_output_is_wellformed(self, tmp_path: Path, sample_path: Path):
        json_path = tmp_path / "nested" / "results.json"
        main(["detect", str(sample_path), "-o", str(tmp_path), "--json", str(json_path)])
        records = json.loads(json_path.read_text())
        assert len(records) == 1
        face = records[0]["faces"][0]
        assert records[0]["num_faces"] == 1
        assert set(face["box"]) == {"x", "y", "width", "height"}
        assert len(face["landmarks"]) == 68
        assert len(face["keypoints"]) == 5

    @pytest.mark.slow
    def test_no_images_flag_skips_writing(self, tmp_path: Path, sample_path: Path):
        main(["detect", str(sample_path), "-o", str(tmp_path), "--no-images"])
        assert list(tmp_path.glob("*.png")) == []

    def test_missing_input_exits_with_error(self, tmp_path: Path):
        assert main(["detect", str(tmp_path / "absent")]) == EXIT_ERROR

    def test_empty_directory_exits_with_error(self, tmp_path: Path):
        assert main(["detect", str(tmp_path)]) == EXIT_ERROR

    @pytest.mark.slow
    def test_no_faces_uses_a_distinct_exit_code(self, image_file: Path, tmp_path: Path):
        code = main(["detect", str(image_file), "-o", str(tmp_path), "--no-images"])
        assert code == EXIT_NO_FACES

    @pytest.mark.slow
    def test_landmarker_none_still_detects(self, tmp_path: Path, sample_path: Path, capsys):
        code = main(["detect", str(sample_path), "-o", str(tmp_path), "--landmarker", "none"])
        assert code == EXIT_OK
        assert "1 face(s)" in capsys.readouterr().out


class TestGridCommand:
    @pytest.mark.slow
    def test_writes_a_contact_sheet(self, tmp_path: Path, data_dir: Path):
        out = tmp_path / "sheet.png"
        code = main(["grid", str(data_dir), "--limit", "4", "--columns", "2", "-o", str(out)])
        assert code == EXIT_OK
        assert out.is_file()

    def test_missing_input_exits_with_error(self, tmp_path: Path):
        assert main(["grid", str(tmp_path)]) == EXIT_ERROR


class TestBenchmarkCommand:
    @pytest.mark.slow
    def test_reports_every_backend(self, tmp_path: Path, sample_path: Path, capsys):
        out = tmp_path / "bench.json"
        code = main(["benchmark", str(sample_path), "--json", str(out)])
        assert code == EXIT_OK
        rows = json.loads(out.read_text())
        assert {row["backend"] for row in rows} == {"haar", "yunet"}
        for row in rows:
            assert 0.0 <= row["recall"] <= 1.0
            assert row["ms_per_image"] > 0
        assert "recall" in capsys.readouterr().out

    def test_missing_input_exits_with_error(self, tmp_path: Path):
        assert main(["benchmark", str(tmp_path)]) == EXIT_ERROR


class TestModelsCommand:
    def test_lists_every_asset(self, capsys):
        assert main(["models"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "cache directory" in out
        for key in ("yunet", "haarcascade", "lbfmodel"):
            assert key in out

    def test_reports_missing_assets(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FACIAL_LANDMARKS_CACHE", str(tmp_path / "empty"))
        monkeypatch.setattr("facial_landmarks.assets._search_roots", list)
        assert main(["models"]) == EXIT_ERROR
        assert "MISSING" in capsys.readouterr().out


class TestErrorHandling:
    def test_offline_mode_reports_cleanly(self, tmp_path: Path, monkeypatch):
        """A missing weight file must not surface as a traceback."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FACIAL_LANDMARKS_CACHE", str(tmp_path / "empty"))
        monkeypatch.setattr("facial_landmarks.assets._search_roots", list)
        (tmp_path / "img.png").write_bytes(
            __import__("cv2").imencode(".png", __import__("numpy").zeros((8, 8, 3), dtype="uint8"))[
                1
            ]
        )
        assert main(["detect", str(tmp_path / "img.png"), "--offline"]) == EXIT_ERROR
