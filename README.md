# Facial Landmark Detection

Face detection and 68-point facial landmark localisation, built on OpenCV with
pluggable backends.

```python
from facial_landmarks import FaceAnalyzer

analyzer = FaceAnalyzer.from_backends()  # YuNet detector + LBF landmarks
image, analysis = analyzer.analyze_path("data/image_0.jpg")

print(analysis.num_faces)  # 1
print(len(analysis.landmarks[0]))  # 68
print(analysis.landmarks[0].subset("left_eye"))  # (6, 2) array
```

---

## Install

Requires Python 3.11 or newer.

```bash
uv sync --group dev          # development, including test and lint tooling
pip install -e ".[plot]"     # or plain pip, with matplotlib for the notebook
```

Model weights resolve automatically. The Haar cascade and the LBF regressor
ship in `haar_classifier/`; the YuNet network is fetched once into
`~/.cache/facial-landmarks` and verified against a known SHA-256 before it is
cached. To check or prefetch everything:

```bash
facial-landmarks models --fetch
```

Set `FACIAL_LANDMARKS_CACHE` to move the cache, or `FACIAL_LANDMARKS_YUNET`
(and friends) to pin a specific file. `--offline` anywhere refuses to download.

## Command line

```bash
facial-landmarks detect data/ -o out/ --json out/faces.json
facial-landmarks grid data/ --limit 12 --columns 4 -o out/grid.png
facial-landmarks benchmark data/
facial-landmarks models
```

`detect` exits `0` when at least one face was found, `3` when none was, and `1`
on error, so it composes in a shell pipeline.

## Backends

Detection and landmark fitting are separate, protocol-based, and swappable:

```python
FaceAnalyzer.from_backends("yunet", "lbf")  # default
FaceAnalyzer.from_backends("haar", "lbf")  # the original notebook's pipeline
FaceAnalyzer.from_backends("yunet", None)  # boxes only, no landmark fitting
```

| backend | what it is | notes |
| --- | --- | --- |
| `yunet` | ~230 KB CNN from the OpenCV Zoo | default; returns a confidence score and five keypoints |
| `haar` | Viola-Jones cascade | the classic baseline, kept for comparison |
| `lbf` | Local Binary Features regressor | 68 points, iBUG 300-W scheme |

### Why YuNet is the default

Measured over the 113 portraits in `data/` with `facial-landmarks benchmark`,
against a fairly tuned cascade (`minNeighbours=3`, OpenCV's own default):

| backend | images with a face | recall | ms/image |
| --- | --- | --- | --- |
| `haar` | 86 / 113 | 76.1% | 47.9 |
| `yunet` | **108 / 113** | **95.6%** | **11.1** |

Nearly 20 points more recall at about a quarter of the runtime. Timings are
single-threaded CPU and will vary with hardware; the recall figures will not.
Reproduce both with `facial-landmarks benchmark data/`.

### Box conventions

Detectors frame a face differently: a Haar cascade returns a near-square box
around the whole head, while YuNet returns a tighter, taller crop. LBF was
trained on Haar-style rectangles, so feeding it a raw YuNet box puts it off its
operating point and drags the jaw contour downward.

Each detector therefore declares its `box_convention`, and the landmarker
remaps incoming boxes before fitting. The transform was fitted on the 77 sample
images where both detectors fire and agree (IoU > 0.3): Haar boxes are
consistently 1.22x wider and 0.93x as tall about a shared centre, with standard
deviations of 0.09 and 0.07.

Applying it halves the median disagreement between landmarks fitted from a
YuNet box and those fitted from the Haar box of the same face (normalised mean
error 0.054 to 0.022). That measures agreement with the regressor's native
operating point rather than accuracy against ground truth, which this dataset
does not carry — but that operating point is what the model was trained for.

`FaceAnalyzer.from_backends` wires this up for you; it only matters if you
construct a landmarker by hand.

## Library tour

| module | contents |
| --- | --- |
| `types` | `FaceBox`, `FaceLandmarks`, the 68-point region table |
| `detectors` | `YuNetDetector`, `HaarCascadeDetector`, the `FaceDetector` protocol |
| `landmarks` | `LBFLandmarker`, the `Landmarker` protocol |
| `pipeline` | `FaceAnalyzer`, `FaceAnalysis` |
| `drawing` | `annotate`, `draw_boxes`, `draw_landmarks` |
| `images` | `load_image`, `save_image`, `contact_sheet` |
| `assets` | weight resolution, caching, verified download |

Both value types are frozen, and landmark point arrays are read-only, so
results cannot be mutated behind another caller's back. Every drawing function
returns a new image rather than annotating the one it was given.

`facial_landmark.ipynb` walks through the same ground with plots.

## Development

```bash
uv sync --group dev
uv run pytest                  # 169 tests
uv run pytest -m "not slow"    # skip the model-backed ones
uv run ruff check . && uv run ruff format --check .
uv run mypy                    # strict
```

CI runs lint, strict type-checking, and the suite on Python 3.11, 3.12 and
3.13, then publishes the detector benchmark to the job summary.

## What changed in 2.0

This started as a single Python 3.7 notebook, which no longer ran at all on a
current OpenCV. The rewrite fixed these along the way, and
`tests/test_regressions.py` guards each one:

- **`cv2.imread(path, cv2.COLOR_BGR2RGB)`** passed a colour-conversion constant
  where an `IMREAD_*` flag belongs. The constant is `4`, which imread reads as
  `IMREAD_ANYCOLOR`, so a greyscale JPEG decoded to a 2D array and every later
  channel index fell over.
- **`if faces == ():`** raises `ValueError` as soon as `detectMultiScale`
  returns a populated ndarray, which is the ordinary case. Backends now return
  a plain list, empty when nothing is found.
- **`cv2.circle(img, (x, y), ...)`** with the float32 points the facemark API
  returns is rejected outright by OpenCV 5. All coordinates are rounded at the
  drawing boundary.
- **Both notebook functions returned an unbound local** when no face was found.
- **Annotation drew onto the input image**, so each figure silently accumulated
  the previous one's boxes.

Alongside those: a `src/` package with 97% test coverage, the YuNet default,
weights out of the checkout and into a verified cache, a CLI, and CI.

## Licence

GPL v3 — see [LICENSE](LICENSE).

Sample portraits in `data/` are used for demonstration. The Haar cascade comes
from OpenCV (BSD), the LBF model from the GSOC2017 facemark project, and YuNet
from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (MIT).
