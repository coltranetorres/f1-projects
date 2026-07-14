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
