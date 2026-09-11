"""Entry point for ``python -m facial_landmarks``."""

from __future__ import annotations

import sys

from facial_landmarks.cli import main

if __name__ == "__main__":
    sys.exit(main())
