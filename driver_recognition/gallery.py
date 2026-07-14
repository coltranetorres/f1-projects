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
