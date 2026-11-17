"""Download the MediaPipe HandLandmarker model file into models/.

The Tasks API needs a .task model bundle on disk (the legacy mp.solutions API
shipped it inside the package; the new one doesn't).
Run once:  python setup_model.py
"""
import urllib.request
from pathlib import Path

import mediapipe as mp

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")
DEST = Path(__file__).parent / "models" / "hand_landmarker.task"

print(f"mediapipe version: {mp.__version__}")
if DEST.exists():
    print(f"Model already present: {DEST}")
else:
    DEST.parent.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
