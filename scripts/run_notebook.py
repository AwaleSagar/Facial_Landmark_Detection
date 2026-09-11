"""Execute the demo notebook, either to refresh its outputs or to check it.

The notebook's outputs are committed so that it is readable on GitHub without
being run. That only stays true if something actually runs it, so CI calls this
with ``--check`` and contributors call it without arguments to refresh.

    python scripts/run_notebook.py            # re-run and save outputs
    python scripts/run_notebook.py --check    # re-run, fail on error, save nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path(__file__).resolve().parents[1] / "facial_landmark.ipynb"
TIMEOUT_SECONDS = 600


def main(argv: list[str] | None = None) -> int:
    """Run the notebook and report the outcome.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 if every cell executed cleanly, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the notebook runs without writing its outputs back",
    )
    args = parser.parse_args(argv)

    # nbformat ships no annotations for read/write.
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"{NOTEBOOK.name} failed to execute:\n{exc}", file=sys.stderr)
        return 1

    if args.check:
        print(f"{NOTEBOOK.name}: all cells executed cleanly")
        return 0

    nbformat.write(notebook, NOTEBOOK)  # type: ignore[no-untyped-call]
    print(f"{NOTEBOOK.name}: outputs refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
