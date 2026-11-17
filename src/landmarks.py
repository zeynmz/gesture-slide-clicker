"""Camera + MediaPipe HandLandmarker wrapper.

Everything that touches the webcam or MediaPipe lives here, so record.py and
live.py see exactly the same preprocessing (most importantly: the mirror flip).

Run as a demo:  python src/landmarks.py   (press q or Esc to quit)
"""
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

# Native camera backend per OS. On Windows, DirectShow opens the webcam much
# faster than OpenCV's default (Media Foundation); elsewhere let OpenCV choose.
if sys.platform == "darwin":
    CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
    CAMERA_HELP = ("On macOS, allow camera access for your terminal app in "
                   "System Settings > Privacy & Security > Camera.")
elif sys.platform == "win32":
    CAMERA_BACKEND = cv2.CAP_DSHOW
    CAMERA_HELP = ("On Windows, check Settings > Privacy & security > Camera "
                   "(allow desktop apps), and close other apps using the camera.")
else:
    CAMERA_BACKEND = cv2.CAP_ANY
    CAMERA_HELP = "Check that a webcam is connected and not used by another app."


class Camera:
    """Webcam that always returns MIRRORED frames.

    Mirroring once, here, guarantees recording and live use see the same image,
    so "point_right" means the presenter's right in both (like a selfie view).
    """

    def __init__(self, index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(index, CAMERA_BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. {CAMERA_HELP} "
                               "Try --camera 1 if you have several cameras.")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)  # 1 = horizontal flip

    def release(self):
        self.cap.release()


class HandTracker:
    """Detects one hand and returns its 21 landmarks as a (21, 3) array in pixels.

    Coordinates are converted to pixels (x*width, y*height, z*width) because
    MediaPipe's normalized x and y are divided by different numbers (width vs
    height), which would distort angles and distances on a non-square image.
    MediaPipe's z uses roughly the same scale as x, hence z*width.
    """

    def __init__(self, model_path=MODEL_PATH, min_confidence=0.5):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing. Run: python setup_model.py")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode tracks the hand between frames instead of re-detecting
            # from scratch each time: faster and less jittery than IMAGE mode.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,  # one presenter, one gesturing hand
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        # "Left"/"Right" as MediaPipe sees it for the last detected hand.
        # Only stored as a sanity check against the --hand tag; not a feature.
        self.last_handedness = None

    def detect(self, frame_bgr):
        """Return a (21, 3) float array of landmarks in pixels, or None if no hand."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, MediaPipe wants RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps (milliseconds).
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            self.last_handedness = None
            return None
        self.last_handedness = result.handedness[0][0].category_name
        lm = result.hand_landmarks[0]
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)

    def close(self):
        self.landmarker.close()


def draw_landmarks(frame, points, color=(0, 255, 0)):
    """Draw the hand skeleton onto the frame (in place)."""
    if points is None:
        return
    xy = points[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 2)
    for i, (x, y) in enumerate(xy):
        # Wrist (0) and index fingertip (8) get a bigger red dot: handy for
        # checking at a glance that "pointing" looks the way we expect.
        big = i in (0, 8)
        cv2.circle(frame, (x, y), 6 if big else 4, (0, 0, 255) if big else (255, 255, 255), -1)


def demo(camera_index=0):
    cam, tracker = Camera(camera_index), HandTracker()
    prev = time.monotonic()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            points = tracker.detect(frame)
            draw_landmarks(frame, points)

            now = time.monotonic()
            fps, prev = 1 / max(now - prev, 1e-6), now
            status = "hand: yes" if points is not None else "hand: NO"
            cv2.putText(frame, f"{status}   {fps:4.1f} fps", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if points is not None:
                x, y, z = points[8]
                cv2.putText(frame, f"index tip px: ({x:.0f}, {y:.0f}, z={z:.0f})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("landmarks demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Show webcam with hand landmarks.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is wrong)")
    demo(parser.parse_args().camera)
