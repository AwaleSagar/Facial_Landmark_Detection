"""Command-line interface for face detection and landmark localisation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from facial_landmarks import __version__
from facial_landmarks._native import quiet_opencv
from facial_landmarks.assets import REGISTRY, cache_dir, resolve_asset
from facial_landmarks.detectors import DETECTOR_BACKENDS
from facial_landmarks.images import contact_sheet, iter_images, load_image, save_image
from facial_landmarks.landmarks import LANDMARKER_BACKENDS
from facial_landmarks.pipeline import FaceAnalyzer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from facial_landmarks.pipeline import FaceAnalysis

__all__ = ["build_parser", "main"]

logger = logging.getLogger("facial_landmarks")

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_NO_FACES: Final = 3


def _configure_logging(verbosity: int) -> None:
    """Set up log output; ``-v`` gives INFO, ``-vv`` DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if verbosity < 2:  # noqa: PLR2004 - -vv means "show me everything"
        quiet_opencv()


def _analysis_to_dict(path: Path, analysis: FaceAnalysis) -> dict[str, object]:
    """Serialise one image's analysis into JSON-safe primitives."""
    return {
        "image": str(path),
        "num_faces": analysis.num_faces,
        "faces": [
            {
                "box": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
                "score": box.score,
                "keypoints": {name: list(point) for name, point in box.keypoints},
                "landmarks": (
                    analysis.landmarks[i].points.tolist() if i < len(analysis.landmarks) else []
                ),
            }
            for i, box in enumerate(analysis.boxes)
        ],
    }


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the backend-selection flags shared by several sub-commands."""
    parser.add_argument(
        "--detector",
        choices=sorted(DETECTOR_BACKENDS),
        default="yunet",
        help="face detection backend (default: %(default)s)",
    )
    parser.add_argument(
        "--landmarker",
        choices=[*sorted(LANDMARKER_BACKENDS), "none"],
        default="lbf",
        help="landmark backend, or 'none' for boxes only (default: %(default)s)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never download weights; fail if they are not already available",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for every sub-command."""
    parser = argparse.ArgumentParser(
        prog="facial-landmarks",
        description="Detect faces and localise 68 facial landmarks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (repeatable)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="annotate images with boxes and landmarks")
    detect.add_argument("inputs", nargs="+", type=Path, help="image files or directories")
    detect.add_argument(
        "-o", "--output", type=Path, default=Path("out"), help="output directory (default: out)"
    )
    detect.add_argument("--json", type=Path, help="also write detections to this JSON file")
    detect.add_argument("--recursive", action="store_true", help="recurse into directories")
    detect.add_argument("--no-boxes", action="store_true", help="draw landmarks only")
    detect.add_argument("--no-scores", action="store_true", help="omit confidence labels")
    detect.add_argument(
        "--no-images", action="store_true", help="compute only; write no annotated images"
    )
    _add_backend_arguments(detect)

    grid = subparsers.add_parser("grid", help="tile annotated images into one contact sheet")
    grid.add_argument("inputs", nargs="+", type=Path, help="image files or directories")
    grid.add_argument(
        "-o", "--output", type=Path, default=Path("out/grid.png"), help="output image path"
    )
    grid.add_argument("--columns", type=int, default=4, help="grid columns (default: %(default)s)")
    grid.add_argument("--limit", type=int, default=8, help="max images (default: %(default)s)")
    grid.add_argument(
        "--tile", type=int, default=256, help="tile size in px (default: %(default)s)"
    )
    grid.add_argument("--recursive", action="store_true", help="recurse into directories")
    _add_backend_arguments(grid)

    bench = subparsers.add_parser(
        "benchmark", help="compare detector backends for recall and speed"
    )
    bench.add_argument("inputs", nargs="+", type=Path, help="image files or directories")
    bench.add_argument("--recursive", action="store_true", help="recurse into directories")
    bench.add_argument("--json", type=Path, help="write results to this JSON file")
    bench.add_argument("--offline", action="store_true", help="never download weights")

    models = subparsers.add_parser("models", help="show or prefetch model weights")
    models.add_argument(
        "--fetch", action="store_true", help="download any weights that are missing"
    )

    return parser


def _build_analyzer(args: argparse.Namespace) -> FaceAnalyzer:
    """Instantiate the analyzer described by the parsed arguments."""
    return FaceAnalyzer.from_backends(
        detector=args.detector,
        landmarker=None if args.landmarker == "none" else args.landmarker,
        allow_download=not args.offline,
    )


def _command_detect(args: argparse.Namespace) -> int:
    """Annotate every input image and report how many faces were found."""
    paths = iter_images(args.inputs, recursive=args.recursive)
    if not paths:
        logger.error("no images found in %s", ", ".join(str(p) for p in args.inputs))
        return EXIT_ERROR

    analyzer = _build_analyzer(args)
    logger.info("analysing %d image(s) with %s", len(paths), analyzer.description)

    records: list[dict[str, object]] = []
    total_faces = 0
    for path in paths:
        image, analysis = analyzer.analyze_path(path)
        total_faces += analysis.num_faces
        records.append(_analysis_to_dict(path, analysis))

        if not args.no_images:
            annotated = analysis.draw_on(
                image, show_boxes=not args.no_boxes, show_scores=not args.no_scores
            )
            destination = save_image(annotated, args.output / f"{path.stem}_annotated.png")
            logger.info("%s -> %s (%d face(s))", path.name, destination, analysis.num_faces)
        print(f"{path}: {analysis.num_faces} face(s)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    print(f"\n{total_faces} face(s) across {len(paths)} image(s)")
    return EXIT_OK if total_faces else EXIT_NO_FACES


def _command_grid(args: argparse.Namespace) -> int:
    """Build a contact sheet of annotated images."""
    paths = iter_images(args.inputs, recursive=args.recursive)[: args.limit]
    if not paths:
        logger.error("no images found")
        return EXIT_ERROR

    analyzer = _build_analyzer(args)
    tiles = []
    for path in paths:
        image, analysis = analyzer.analyze_path(path)
        tiles.append(analysis.draw_on(image))

    sheet = contact_sheet(tiles, columns=args.columns, tile=(args.tile, args.tile))
    destination = save_image(sheet, args.output)
    print(f"wrote {destination} ({len(tiles)} image(s))")
    return EXIT_OK


def _command_benchmark(args: argparse.Namespace) -> int:
    """Compare every detector backend on recall and throughput."""
    paths = iter_images(args.inputs, recursive=args.recursive)
    if not paths:
        logger.error("no images found")
        return EXIT_ERROR

    images = [load_image(path) for path in paths]
    results: list[dict[str, object]] = []

    for name in sorted(DETECTOR_BACKENDS):
        analyzer = FaceAnalyzer.from_backends(
            detector=name, landmarker=None, allow_download=not args.offline
        )
        hits = faces = 0
        start = time.perf_counter()
        for image in images:
            found = analyzer.detector.detect(image)
            faces += len(found)
            hits += bool(found)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "backend": name,
                "images": len(images),
                "images_with_a_face": hits,
                "recall": hits / len(images),
                "total_faces": faces,
                "seconds": round(elapsed, 3),
                "ms_per_image": round(1000 * elapsed / len(images), 1),
            }
        )

    header = f"{'backend':<10}{'found/total':>14}{'recall':>9}{'faces':>8}{'ms/image':>11}"
    print(header)
    print("-" * len(header))
    for row in results:
        ratio = f"{row['images_with_a_face']}/{row['images']}"
        print(
            f"{row['backend']:<10}"
            f"{ratio:>14}"
            f"{row['recall']:>8.1%}"
            f"{row['total_faces']:>8}"
            f"{row['ms_per_image']:>11}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return EXIT_OK


def _command_models(args: argparse.Namespace) -> int:
    """List model assets and optionally prefetch the missing ones."""
    print(f"cache directory: {cache_dir()}\n")
    status = EXIT_OK
    for key, asset in REGISTRY.items():
        try:
            path = resolve_asset(key, allow_download=args.fetch)
            state = f"ok   {path}"
        except FileNotFoundError as exc:
            state = f"MISSING ({exc})"
            status = EXIT_ERROR
        print(f"{key:<12} {asset.description}\n{'':<12} {state}\n")
    return status


_COMMANDS = {
    "detect": _command_detect,
    "grid": _command_grid,
    "benchmark": _command_benchmark,
    "models": _command_models,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 on success, 1 on error, 3 when no face was found.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.warning("interrupted")
        return EXIT_ERROR
    except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as exc:
        logger.error("%s", exc)  # noqa: TRY400 - a traceback helps nobody here
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
