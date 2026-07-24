# Driver Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live webcam app that recognizes F1 drivers' faces held up to the camera, matching against a small reference-photo gallery.

**Architecture:** A `gallery.py` module builds and caches face embeddings (via InsightFace) from the photos in `driver_images/`, keyed by filename stem (driver name). `recognize_webcam.py` opens the webcam via OpenCV, detects faces per frame, matches each against the gallery by cosine similarity, and overlays name + confidence (or "Unknown" below threshold) on a live window.

**Tech Stack:** OpenCV (`opencv-python`) for capture/display, InsightFace (`insightface` + `onnxruntime`, `buffalo_l` model) for face detection + embeddings, NumPy for similarity math.

## Global Constraints

- Dependencies to add to root `pyproject.toml`: `opencv-python`, `insightface`, `onnxruntime` (per spec's "Dependencies to add" section).
- Gallery source of truth: `driver_recognition/driver_images/*.jpg` — filename stem (e.g. `lewis.jpg` -> `"lewis"`) is the driver's canonical name used everywhere.
- Match threshold: `0.5` cosine similarity, as a module-level constant — below it, label `"Unknown"`.
- Embeddings are InsightFace's `normed_embedding` (L2-normalized), so cosine similarity is a plain dot product.
- InsightFace model cache lives at `driver_recognition/models/` (pass as `root=` to `FaceAnalysis`), not the library's default `~/.insightface`, so the project is self-contained.
- Self-match smoke test threshold: `>= 0.9` similarity (same photo used to build the gallery entry and to query it).
- Quit key for the live window: `q`.

---

### Task 1: Add CV dependencies and verify install

**Files:**
- Modify: `pyproject.toml` (root, currently has `fastf1`, `playwright`, `scikit-learn`, `scipy`, `shap`, `xgboost` under `dependencies`)

**Interfaces:**
- Produces: working `cv2`, `insightface`, `onnxruntime` imports available to all later tasks.

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml`'s `dependencies` list to add three entries (keep existing ones, keep alphabetical-ish ordering consistent with the file):

```toml
dependencies = [
    "fastf1>=3.8.3",
    "insightface>=0.7.3",
    "onnxruntime>=1.20.0",
    "opencv-python>=4.10.0",
    "playwright>=1.60.0",
    "scikit-learn>=1.8.0",
    "scipy>=1.14.0",
    "shap>=0.52.0",
    "xgboost>=3.2.0",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: completes without error, installs `opencv-python`, `insightface`, `onnxruntime` and their transitive deps (numpy, onnx, etc).

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "import cv2, insightface, onnxruntime; print('ok')"`
Expected output: `ok`

- [ ] **Step 4: Create the models cache directory**

Run: `mkdir -p driver_recognition/models`

This is where `gallery.py` (Task 2) will point InsightFace's model cache, keeping downloaded models inside the project instead of `~/.insightface`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add opencv-python, insightface, onnxruntime dependencies"
```

Note: `driver_recognition/models/` is created empty in this step and won't show up in `git add` until it has a file in it (Task 2 populates it) — that's expected, don't force-add an empty directory.

---

### Task 2: Gallery module — build, cache, and match embeddings

**Files:**
- Create: `driver_recognition/gallery.py`
- Create: `driver_recognition/test_gallery.py`

**Interfaces:**
- Consumes: `cv2`, `insightface.app.FaceAnalysis` (from Task 1's dependencies); reads `driver_recognition/driver_images/*.jpg`.
- Produces (used by Task 3):
  - `get_face_app() -> FaceAnalysis` — returns a lazily-initialized, module-cached `FaceAnalysis` instance configured with `root=MODELS_DIR`.
  - `build_gallery(force: bool = False) -> dict[str, np.ndarray]` — maps driver name -> 512-d `float32` normalized embedding. Uses the on-disk cache at `driver_recognition/gallery_embeddings.npz` unless it's missing/stale/`force=True`.
  - `best_match(query_embedding: np.ndarray, gallery: dict[str, np.ndarray]) -> tuple[str, float]` — returns `(driver_name, similarity)` for the closest gallery entry.
  - `DRIVER_IMAGES_DIR`, `THRESHOLD = 0.5` module-level constants.

- [ ] **Step 1: Write the failing test**

Create `driver_recognition/test_gallery.py`:

```python
import os

import cv2
import numpy as np

from gallery import DRIVER_IMAGES_DIR, best_match, build_gallery, get_face_app


def test_each_driver_photo_matches_itself():
    gallery = build_gallery(force=True)
    assert len(gallery) == 6

    app = get_face_app()

    for name in gallery:
        path = os.path.join(DRIVER_IMAGES_DIR, f"{name}.jpg")
        img = cv2.imread(path)
        assert img is not None, f"could not read {path}"

        faces = app.get(img)
        assert len(faces) >= 1, f"no face detected in {name}.jpg"

        query_embedding = faces[0].normed_embedding.astype(np.float32)
        matched_name, similarity = best_match(query_embedding, gallery)

        assert matched_name == name
        assert similarity >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd driver_recognition && uv run python -m pytest test_gallery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gallery'`

- [ ] **Step 3: Write the implementation**

Create `driver_recognition/gallery.py`:

```python
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_IMAGES_DIR = os.path.join(HERE, "driver_images")
MODELS_DIR = os.path.join(HERE, "models")
CACHE_PATH = os.path.join(HERE, "gallery_embeddings.npz")
THRESHOLD = 0.5

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

_app = None


def get_face_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_PATH):
        return False
    cache_mtime = os.path.getmtime(CACHE_PATH)
    for fname in os.listdir(DRIVER_IMAGES_DIR):
        if os.path.getmtime(os.path.join(DRIVER_IMAGES_DIR, fname)) > cache_mtime:
            return False
    return True


def build_gallery(force: bool = False) -> dict:
    if not force and _cache_is_fresh():
        data = np.load(CACHE_PATH)
        return {name: data[name] for name in data.files}

    app = get_face_app()
    gallery = {}
    for fname in sorted(os.listdir(DRIVER_IMAGES_DIR)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in _IMAGE_EXTENSIONS:
            continue

        path = os.path.join(DRIVER_IMAGES_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {fname}, skipping")
            continue

        faces = app.get(img)
        if not faces:
            print(f"Warning: no face detected in {fname}, skipping")
            continue

        gallery[name] = faces[0].normed_embedding.astype(np.float32)

    np.savez(CACHE_PATH, **gallery)
    return gallery


def best_match(query_embedding: np.ndarray, gallery: dict) -> tuple:
    names = list(gallery.keys())
    embeddings = np.stack([gallery[n] for n in names])
    similarities = embeddings @ query_embedding.astype(np.float32)
    best_idx = int(np.argmax(similarities))
    return names[best_idx], float(similarities[best_idx])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd driver_recognition && uv run python -m pytest test_gallery.py -v`
Expected: PASS (first run downloads the `buffalo_l` model into `driver_recognition/models/` — needs internet, may take a couple of minutes; subsequent runs are fast since the model and the `gallery_embeddings.npz` cache are both reused).

- [ ] **Step 5: Commit**

```bash
git add driver_recognition/gallery.py driver_recognition/test_gallery.py driver_recognition/models driver_recognition/gallery_embeddings.npz
git commit -m "Add gallery module for building and matching driver face embeddings"
```

Note: if `gallery_embeddings.npz` or files under `models/` are large, check `git status` output before committing — if they're large enough to be worth excluding, ask the user before adding a `.gitignore` entry rather than committing multi-hundred-MB model files.

---

### Task 3: Live webcam recognition app

**Files:**
- Create: `driver_recognition/recognize_webcam.py`

**Interfaces:**
- Consumes: `gallery.build_gallery()`, `gallery.get_face_app()`, `gallery.best_match()`, `gallery.THRESHOLD` (all from Task 2).
- Produces: a runnable script, no importable interface needed by other tasks.

- [ ] **Step 1: Write the implementation**

Create `driver_recognition/recognize_webcam.py`:

```python
import sys

import cv2
import numpy as np

from gallery import THRESHOLD, best_match, build_gallery, get_face_app


def main():
    gallery = build_gallery()
    if not gallery:
        print("Gallery is empty — no driver embeddings available. Check driver_images/.")
        sys.exit(1)

    app = get_face_app()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    print("Webcam recognition running. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read frame from webcam.")
                break

            faces = app.get(frame)
            for face in faces:
                query_embedding = face.normed_embedding.astype(np.float32)
                name, similarity = best_match(query_embedding, gallery)

                if similarity >= THRESHOLD:
                    label = f"{name} ({similarity * 100:.0f}%)"
                else:
                    label = "Unknown"

                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Driver Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manually verify the app runs end-to-end**

Run: `uv run python driver_recognition/recognize_webcam.py`

Expected:
- Terminal prints `Webcam recognition running. Press 'q' to quit.`
- A window titled "Driver Recognition" opens showing the live webcam feed.
- Holding up a printed/on-screen photo of one of the 6 drivers (or one of the files in `driver_images/` shown on a phone/second screen) to the camera draws a green box around the face with a label like `lewis (91%)`.
- Holding up a random other face shows `Unknown`.
- Pressing `q` closes the window and returns control to the terminal cleanly (no traceback).

This is a manual check, not an automated test — there is no camera-less way to verify live capture behavior, consistent with the spec's testing section.

- [ ] **Step 3: Commit**

```bash
git add driver_recognition/recognize_webcam.py
git commit -m "Add live webcam driver recognition app"
```

---

## Post-plan check

After Task 3, the app is complete per spec: `uv run python driver_recognition/recognize_webcam.py` is the full user-facing entry point. No further tasks needed.
