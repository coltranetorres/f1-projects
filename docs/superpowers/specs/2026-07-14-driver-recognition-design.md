# Driver Recognition — Live Webcam Face Recognition

## Purpose

Recognize F1 drivers' faces shown to a webcam in real time. The user holds a photo (or poster/screen) of a driver up to the camera, and the app draws a bounding box with the driver's name and confidence on a live video window.

## Scope

Covers the 6 drivers currently in `driver_recognition/driver_images/` (alex, carlos, charles, lando, lewis, oscar — one reference photo each). Not scoped to the full 2026 grid; the design must make adding a driver later trivial (drop a photo in, rerun).

Out of scope: static image batch processing, video file processing, "which driver do you look like" novelty mode. These could be added later using the same gallery/matching core, but are not part of this project.

## Tech Stack

- **OpenCV** (`opencv-python`) — webcam capture and the live display window.
- **InsightFace** (`insightface` + `onnxruntime`) — face detection and 512-d face embeddings (`buffalo_l` model). Chosen over `face_recognition`/dlib because it installs cleanly via `uv add` on Windows/Python 3.13 with no C++ compiler required, and gives strong accuracy for few-shot gallery matching.

## Architecture

```
driver_recognition/
├── driver_images/          # existing — one reference photo per driver
│   ├── alex.jpg, carlos.jpg, charles.jpg, lando.jpg, lewis.jpg, oscar.jpg
├── gallery.py               # builds/loads the embedding gallery from driver_images/
├── recognize_webcam.py      # main app: opens webcam, detects + labels faces live
├── test_gallery.py          # smoke test: static-image self-match check
├── gallery_embeddings.npz   # generated cache: name -> embedding vector
└── models/                  # InsightFace model download cache
```

- `gallery.py` exposes `build_gallery()`, which runs InsightFace's `FaceAnalysis` over each photo in `driver_images/`, stores one embedding per driver (filename stem = driver name, e.g. `lewis.jpg` -> `"lewis"`), and caches the result to `gallery_embeddings.npz`. The cache is rebuilt automatically if any file in `driver_images/` is newer than the cache file. Adding a 7th driver later requires no code changes — just add a photo and rerun.
- `recognize_webcam.py` loads the gallery, opens the default webcam via OpenCV, and per frame:
  1. Runs `FaceAnalysis.get(frame)` to get detected faces (bounding box + embedding) for that frame.
  2. For each detected face, computes cosine similarity against all 6 gallery embeddings (embeddings are L2-normalized, so this is a dot product) and takes the best match.
  3. If `best_similarity >= THRESHOLD` (default `0.5`, a module-level constant), labels the face with the matched driver's name and similarity as a percentage; otherwise labels it `"Unknown"`.
  4. Draws a bounding box and label on the frame and shows it in an OpenCV window (`cv2.imshow`), updating every frame.
  5. Exits cleanly when the user presses `q`.
  - Multiple faces in frame are each matched independently, so holding up two driver photos at once labels both.

## Error Handling

- No webcam detected: print a clear error message and exit — do not let OpenCV fail with an opaque error.
- InsightFace models not yet cached locally: first run downloads them automatically (built into the library); requires internet access on first run only.
- No face detected in a given frame: display the raw frame with no box, no crash.
- Corrupt/unreadable file encountered in `driver_images/` during gallery build: skip it with a printed warning, continue building the gallery with the remaining photos.

## Testing

Live webcam behavior isn't unit-testable, so `test_gallery.py` provides a smoke test that doesn't require a camera:
1. Build the gallery from `driver_images/`.
2. Feed each of the 6 reference photos back through the same detect + match pipeline as static images.
3. Assert each photo matches its own name with high similarity (e.g. `>= 0.9`, since it's the exact same image used to build the gallery entry).

Run via `uv run python -m pytest driver_recognition/`. This catches regressions in the embedding pipeline or an accidentally-miscalibrated threshold, without needing a camera or manual verification each time.

## Dependencies to add

Add to root `pyproject.toml`: `opencv-python`, `insightface`, `onnxruntime`.
