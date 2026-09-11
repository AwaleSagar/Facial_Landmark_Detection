"""Tests for model asset resolution, caching and download verification."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import pytest

from facial_landmarks.assets import (
    CACHE_ENV_VAR,
    REGISTRY,
    ModelAsset,
    _download,
    cache_dir,
    resolve_asset,
    sha256_of,
)


class TestCacheDir:
    def test_env_override_wins(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        assert cache_dir() == tmp_path

    def test_falls_back_to_xdg(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache_dir() == tmp_path / "facial-landmarks"

    def test_falls_back_to_home(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert cache_dir() == tmp_path / ".cache" / "facial-landmarks"


class TestRegistry:
    def test_every_asset_is_self_consistent(self):
        for key, asset in REGISTRY.items():
            assert asset.key == key
            assert asset.url.startswith("https://")
            assert len(asset.sha256) == 64
            assert asset.size > 0

    def test_env_var_name(self):
        assert REGISTRY["yunet"].env_var == "FACIAL_LANDMARKS_YUNET"

    def test_lfs_backed_url_uses_media_host(self):
        # raw.githubusercontent.com serves a 131-byte LFS pointer for this file,
        # not the model, so the registry must point at the media host.
        assert REGISTRY["yunet"].url.startswith("https://media.githubusercontent.com/")


class TestSha256:
    def test_matches_hashlib(self, tmp_path: Path):
        path = tmp_path / "blob.bin"
        payload = b"facial-landmarks" * 1000
        path.write_bytes(payload)
        assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


class TestResolveAsset:
    def test_unknown_key(self):
        with pytest.raises(KeyError, match="unknown model asset"):
            resolve_asset("not-a-model")

    def test_explicit_path_short_circuits(self, tmp_path: Path):
        path = tmp_path / "weights.onnx"
        path.write_bytes(b"x")
        assert resolve_asset("yunet", path=path) == path

    def test_explicit_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found at"):
            resolve_asset("yunet", path=tmp_path / "absent.onnx")

    def test_env_var_pin(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "pinned.onnx"
        path.write_bytes(b"x")
        monkeypatch.setenv(REGISTRY["yunet"].env_var, str(path))
        assert resolve_asset("yunet") == path

    def test_env_var_pointing_nowhere_raises(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(REGISTRY["yunet"].env_var, str(tmp_path / "gone.onnx"))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_asset("yunet")

    def test_finds_bundled_weights(self, tmp_path: Path, monkeypatch):
        bundled = tmp_path / "models"
        bundled.mkdir()
        target = bundled / REGISTRY["yunet"].filename
        target.write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(REGISTRY["yunet"].env_var, raising=False)
        assert resolve_asset("yunet") == target

    def test_offline_mode_refuses_to_download(self, tmp_path: Path, monkeypatch):
        # Point every search root and the cache somewhere empty.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
        monkeypatch.delenv(REGISTRY["yunet"].env_var, raising=False)
        monkeypatch.setattr("facial_landmarks.assets._search_roots", list)
        with pytest.raises(FileNotFoundError, match="downloads are disabled"):
            resolve_asset("yunet", allow_download=False)

    def test_cached_file_is_reused(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
        monkeypatch.delenv(REGISTRY["yunet"].env_var, raising=False)
        monkeypatch.setattr("facial_landmarks.assets._search_roots", list)
        cached = tmp_path / "cache" / REGISTRY["yunet"].filename
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"x")
        assert resolve_asset("yunet", allow_download=False) == cached


class TestDownloadVerification:
    def test_checksum_mismatch_is_rejected_and_not_cached(self, tmp_path: Path, monkeypatch):
        """A corrupt download must never be committed to the cache."""
        asset = ModelAsset(
            key="fake",
            filename="fake.bin",
            url="https://example.invalid/fake.bin",
            sha256="0" * 64,  # deliberately wrong
            size=4,
            description="test asset",
        )
        destination = tmp_path / asset.filename

        class FakeResponse:
            def read(self, *_args):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def fake_urlopen(*_args, **_kwargs):
            return FakeResponse()

        def fake_copyfileobj(_src, dst, *_args):
            dst.write(b"junk")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(shutil, "copyfileobj", fake_copyfileobj)

        with pytest.raises(OSError, match="checksum mismatch"):
            _download(asset, destination)

        assert not destination.exists()
        # No .partial debris is left behind either.
        assert list(tmp_path.glob("*.partial")) == []
