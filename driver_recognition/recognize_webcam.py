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
