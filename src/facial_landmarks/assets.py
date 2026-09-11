"""Resolution, caching and verified download of model weight files.

Weights are large binaries that do not belong in source control, so this module
resolves them from (in order) an explicit path, an environment override, any
directory bundled with the checkout, the user cache, and finally an HTTPS
download that is checksum-verified before it is committed to the cache.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["REGISTRY", "ModelAsset", "cache_dir", "resolve_asset"]

logger = logging.getLogger(__name__)

#: Environment variable overriding where downloaded weights are cached.
CACHE_ENV_VAR: Final = "FACIAL_LANDMARKS_CACHE"
#: Environment variable prefix for pinning one asset to an explicit path,
#: e.g. ``FACIAL_LANDMARKS_YUNET=/models/yunet.onnx``.
ASSET_ENV_PREFIX: Final = "FACIAL_LANDMARKS_"

_DOWNLOAD_CHUNK: Final = 1 << 20
_DOWNLOAD_TIMEOUT: Final = 120


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """A downloadable model weight file.

    Attributes:
        key: Short registry name, e.g. ``"yunet"``.
        filename: Canonical file name used inside the cache directory.
        url: HTTPS source. GitHub LFS paths must point at ``media.``.
        sha256: Expected digest of the upstream file.
        size: Expected size in bytes.
        description: Human-readable summary for CLI output.
    """

    key: str
    filename: str
    url: str
    sha256: str
    size: int
    description: str

    @property
    def env_var(self) -> str:
        """Environment variable that pins this asset to an explicit path."""
        return f"{ASSET_ENV_PREFIX}{self.key.upper()}"


#: Every weight file the library knows how to fetch.
REGISTRY: Final[dict[str, ModelAsset]] = {
    "yunet": ModelAsset(
        key="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size=232589,
        description="YuNet CNN face detector (OpenCV Zoo, March 2023)",
    ),
    "haarcascade": ModelAsset(
        key="haarcascade",
        filename="haarcascade_frontalface_alt2.xml",
        url=(
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/"
            "haarcascades/haarcascade_frontalface_alt2.xml"
        ),
        sha256="7b0c967d9abbdfbde025eb9c786947d151b6426040d07a8f9562ed8fd90724b4",
        size=540616,
        description="Haar frontal-face cascade (legacy baseline)",
    ),
    "lbfmodel": ModelAsset(
        key="lbfmodel",
        filename="lbfmodel.yaml",
        url=("https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"),
        sha256="70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b",
        size=56375857,
        description="LBF 68-point facial landmark regressor",
    ),
}

#: Directories inside a checkout that may already carry weights.
BUNDLED_DIRS: Final = ("models", "haar_classifier")


def cache_dir() -> Path:
    """Return the directory used for downloaded weights.

    Honours :data:`CACHE_ENV_VAR`, then ``XDG_CACHE_HOME``, then ``~/.cache``.
    """
    if override := os.environ.get(CACHE_ENV_VAR):
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "facial-landmarks"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _search_roots() -> list[Path]:
    """Candidate checkout directories that may bundle weights."""
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    seen: list[Path] = []
    for root in roots:
        for name in BUNDLED_DIRS:
            candidate = root / name
            if candidate.is_dir() and candidate not in seen:
                seen.append(candidate)
    return seen


def _download(asset: ModelAsset, destination: Path) -> None:
    """Fetch ``asset`` to ``destination``, verifying the digest before commit.

    The file is written to a temporary path in the same directory and only
    moved into place once the checksum matches, so an interrupted download can
    never leave a corrupt file in the cache.

    Raises:
        OSError: If the download fails or the checksum does not match.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s (%.1f MB) from %s", asset.filename, asset.size / 1e6, asset.url)

    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=".partial")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            request = urllib.request.Request(  # noqa: S310 - registry URLs are https literals
                asset.url,
                headers={"User-Agent": "facial-landmarks/2.0"},
            )
            with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:  # noqa: S310
                shutil.copyfileobj(response, out, _DOWNLOAD_CHUNK)

        actual = sha256_of(tmp)
        if actual != asset.sha256:
            msg = (
                f"checksum mismatch for {asset.filename}: "
                f"expected {asset.sha256}, got {actual}. Refusing to cache it."
            )
            raise OSError(msg)
        tmp.replace(destination)
        logger.info("cached %s at %s", asset.filename, destination)
    finally:
        tmp.unlink(missing_ok=True)


def _resolve_locally(asset: ModelAsset, path: Path | str | None) -> Path | None:
    """Find ``asset`` without touching the network.

    Checks the explicit path, then the asset's environment variable, then any
    weights bundled in the checkout.

    Args:
        asset: The asset to locate.
        path: An explicit path that short-circuits the search.

    Returns:
        The located file, or ``None`` if only the cache or a download can help.

    Raises:
        FileNotFoundError: If an explicitly requested path does not exist.
    """
    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            msg = f"model file for {asset.key!r} not found at {explicit}"
            raise FileNotFoundError(msg)
        return explicit

    if pinned := os.environ.get(asset.env_var):
        candidate = Path(pinned).expanduser()
        if not candidate.is_file():
            msg = f"{asset.env_var} points at {candidate}, which does not exist"
            raise FileNotFoundError(msg)
        return candidate

    for root in _search_roots():
        candidate = root / asset.filename
        if candidate.is_file():
            logger.debug("using bundled %s from %s", asset.filename, root)
            return candidate
    return None


def resolve_asset(
    key: str,
    *,
    path: Path | str | None = None,
    allow_download: bool = True,
) -> Path:
    """Locate the weight file for ``key``, downloading it if necessary.

    Resolution order: explicit ``path`` argument, the asset's environment
    variable, directories bundled in the checkout, the user cache, download.

    Args:
        key: Registry key, e.g. ``"yunet"``.
        path: Explicit file path that short-circuits resolution.
        allow_download: If ``False``, raise instead of hitting the network.

    Returns:
        Path to an existing weight file.

    Raises:
        KeyError: If ``key`` is not in :data:`REGISTRY`.
        FileNotFoundError: If the file cannot be found or fetched.
    """
    if key not in REGISTRY:
        msg = f"unknown model asset {key!r}; known assets: {sorted(REGISTRY)}"
        raise KeyError(msg)
    asset = REGISTRY[key]

    if local := _resolve_locally(asset, path):
        return local

    cached = cache_dir() / asset.filename
    if cached.is_file():
        return cached

    if not allow_download:
        msg = (
            f"{asset.filename} is not available locally and downloads are disabled. "
            f"Fetch it from {asset.url} or set {asset.env_var}."
        )
        raise FileNotFoundError(msg)

    try:
        _download(asset, cached)
    except OSError as exc:
        msg = f"could not obtain {asset.filename}: {exc}"
        raise FileNotFoundError(msg) from exc
    return cached
