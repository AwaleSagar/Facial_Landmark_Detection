"""Helpers for quietening OpenCV's native (C++) output.

OpenCV writes some messages from C++ rather than through Python's logging, so
neither ``logging`` nor ``contextlib.redirect_stdout`` can intercept them:

* the contrib facemark module prints ``loading data from : ...`` to stdout
  whenever a model is loaded;
* the DNN backend emits ``[ WARN:0@...]`` lines through its own logger.

Both are noise in a CLI whose stdout is meant to be parseable, so this module
silences them at the file-descriptor level and via OpenCV's log level.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["quiet_opencv", "suppress_native_stdout"]


@contextlib.contextmanager
def suppress_native_stdout() -> Iterator[None]:
    """Temporarily redirect the process's stdout file descriptor to null.

    Yields control with fd 1 pointing at the null device, restoring it
    afterwards even if the body raises. Falls back to doing nothing when the
    descriptor cannot be duplicated, as under some test capture plugins.
    """
    try:
        sys.stdout.flush()
        saved = os.dup(1)
    except (OSError, ValueError):  # pragma: no cover - unusual stdio setups
        yield
        return

    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(null_fd)
        os.close(saved)


def quiet_opencv() -> None:
    """Raise OpenCV's native log level so informational warnings stay hidden.

    Safe to call repeatedly, and a no-op on builds without ``cv2.utils.logging``.
    """
    with contextlib.suppress(AttributeError):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
