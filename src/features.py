"""Turn 21 raw landmarks into a feature vector the classifier can use.

Raw pixel coordinates depend on WHERE the hand is in the image and HOW BIG it
looks (distance to camera, hand size). Neither should change the gesture, so:
  1. translate: make every point relative to the wrist  -> position doesn't matter
  2. scale: divide by palm length (wrist -> middle knuckle) -> size/distance doesn't matter
  3. flatten to one row
We deliberately do NOT rotate the hand upright: the direction the hand points
is exactly what separates point_left from point_right.

Optional extra features (config "extra"):
  - finger extension, 5 values: (wrist->tip) / (wrist->knuckle) per finger.
    >1.5 = finger stretched out, <1.2 = curled. Barely affected by HaGRID's
    uncertain aspect ratio, which is why we add them.
  - index finger direction (cos, sin) and hand direction wrist->middle knuckle (cos, sin).

We use x and y only: HaGRID has no depth (z), and z from a single webcam is noisy.
Works on many hands at once: input shape (N, 21, 2+) or a single (21, 2+).
"""
import numpy as np

DEFAULT_CONFIG = {"extra": True}

WRIST, MIDDLE_MCP = 0, 9
FINGERS = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]  # (knuckle, tip): thumb..pinky


def _direction(points, a, b):
    """cos and sin of the bone a -> b, with y pointing up (so +sin = upwards)."""
    v = points[:, b] - points[:, a]
    angle = np.arctan2(-v[:, 1], v[:, 0])
    return np.column_stack([np.cos(angle), np.sin(angle)])


def make_features(landmarks, config=DEFAULT_CONFIG):
    """(N, 21, >=2) or (21, >=2) landmarks -> (N, n_features) float array."""
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim == 2:
        pts = pts[None]
    pts = pts[:, :, :2]                                    # x, y only

    rel = pts - pts[:, WRIST:WRIST + 1]                    # 1. relative to wrist
    palm = np.linalg.norm(rel[:, MIDDLE_MCP], axis=1)
    rel = rel / np.maximum(palm, 1e-6)[:, None, None]      # 2. scale by palm length
    feats = [rel.reshape(len(rel), -1)]                    # 3. flatten: 42 values

    if config.get("extra"):
        dist = np.linalg.norm(rel, axis=2)                 # distance of each point to wrist
        ext = np.column_stack([dist[:, t] / np.maximum(dist[:, k], 1e-6) for k, t in FINGERS])
        feats += [ext, _direction(pts, 5, 8), _direction(pts, WRIST, MIDDLE_MCP)]
    return np.hstack(feats)


def coords_from_df(df):
    """CSV rows (x0, y0, z0, ... z20) -> (N, 21, 3) array."""
    cols = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
    return df[cols].to_numpy(dtype=np.float64).reshape(-1, 21, 3)
