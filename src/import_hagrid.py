"""Convert HaGRIDv2 landmark annotations into our CSV format (data/hagrid.csv).

HaGRIDv2 (github.com/hukenovs/hagrid, CC BY-SA 4.0 variant) has ~1M photos of
tens of thousands of people, with MediaPipe hand landmarks already annotated.
We only use the annotation JSONs, never the images.

Run once (needs data/hagrid_raw/annotations.zip, 719 MB, see README):
    python src/import_hagrid.py

Mapping HaGRID classes -> our classes:
    palm, stop          -> open_palm
    no_gesture + others -> none   (thousands of people = varied negatives)

No pointing data comes from HaGRID: its "point" class is pointing AT the camera
(in 2D the index finger looks shorter than the other fingers; only ~1% of those
hands have just the index extended), not a presenter pointing sideways. It
still makes a good "none" example (pointing at the screen is not a command).

Instead we make SYNTHETIC pointing hands by rotating upright hands by ~90 deg
in the image plane, in two variants (note column), compared in Phase 3:
    rot_one   "one" (index finger up)  -> index finger pointing sideways
    rot_palm  "palm"/"stop" (flat hand) -> flat hand pointing sideways
Caveat: a 2D rotation of a hand facing the camera is not the same as a real
sideways point (that usually shows the side of the hand). Our own recordings
are the ground truth that tells us whether this helps.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from record import COORD_COLUMNS, TAG_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "hagrid_raw" / "annotations.zip"
OUT_PATH = ROOT / "data" / "hagrid.csv"

# HaGRID stores x and y as fractions of image width and height, but not the
# image size, and sizes vary. We assume portrait Full HD (1080 x 1920), the most
# common case. This is UNCERTAIN: some hands will be stretched. Features that
# only ask "is this finger extended?" barely change with the aspect ratio, which
# we checked (palm/fist look the same at 9:16 and 16:9), so Phase 3 should lean on those.
IMG_W, IMG_H = 1080, 1920

OPEN_PALM = {"palm", "stop"}
# Index finger extended sideways ("finger gun", L-shape): too close to a real
# point to call "none", but not a clean point either. Leave them out.
EXCLUDE = {"three_gun", "thumb_index", "thumb_index2"}

# Caps keep the CSV manageable and the classes roughly balanced.
MAX_PER_CLASS = {"open_palm": 20000}
MAX_NONE_PER_SOURCE = 1500      # per HaGRID class mapped to "none"...
MAX_NONE_NO_GESTURE = 12000     # ...except no_gesture, the most natural postures
SYNTH_PER_VARIANT = 20000       # rotated hands per variant (~half right, half left)
SYNTH_JITTER_DEG = 20           # real points aren't perfectly horizontal
# Which bone defines "the direction the hand points" for each variant.
SYNTH_SOURCES = {"rot_one": ({"one"}, (5, 8)),           # index finger
                 "rot_palm": (OPEN_PALM, (0, 9))}         # wrist -> middle knuckle
SEED = 0


def direction(points, a=5, b=8):
    """Angle (degrees) of the bone a -> b; default the index finger. 0 = right, 90 = up."""
    v = points[b] - points[a]
    return np.degrees(np.arctan2(-v[1], v[0]))  # minus: image y grows downwards


def rotate_to(points, bone, target_deg):
    """Rotate the hand around the wrist so that `bone` points at target_deg."""
    phi = np.radians(target_deg - direction(points, *bone))
    # Work with y pointing up (math convention), then convert back to image y.
    u, v = points[:, 0] - points[0, 0], -(points[:, 1] - points[0, 1])
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([points[0, 0] + c * u - s * v, points[0, 1] - (s * u + c * v)])


def to_coords(pts):
    """(21, 2) pixel points -> 63 values in our x0,y0,z0,... order. HaGRID has no depth."""
    return np.column_stack([pts, np.full(21, np.nan)]).round(2).flatten().tolist()


def map_label(source, points):
    """Return our class name for one HaGRID hand, or None to skip it."""
    if source in EXCLUDE:
        return None
    angle = direction(points)
    sideways_right = abs(angle) <= 45
    sideways_left = abs(angle) >= 135
    if source in OPEN_PALM:
        return "open_palm"
    if source == "one" and (sideways_right or sideways_left):
        return None  # a tilted "index finger up" looks exactly like a point
    return "none"


def main():
    rng = np.random.default_rng(SEED)
    candidates = defaultdict(list)  # (our_class, source_class) -> rows
    synth_pool = defaultdict(list)  # variant -> (image_id, person, points) of upright hands
    skipped_no_user = 0
    with zipfile.ZipFile(ZIP_PATH) as z:
        files = sorted(n for n in z.namelist()
                       if n.endswith(".json") and "checkpoint" not in n)
        for name in files:
            print(f"reading {name}")
            for image_id, ann in json.loads(z.read(name)).items():
                if not ann.get("user_id"):
                    # Without a person ID we can't keep this person out of the
                    # training folds when testing on them, so skip it.
                    skipped_no_user += 1
                    continue
                for source, lm in zip(ann["labels"], ann["hand_landmarks"]):
                    if len(lm) != 21:  # auto-annotation failed for this hand
                        continue
                    # To pixels so x and y have the same unit, like landmarks.py.
                    # Mirroring isn't needed: point labels come from the finger
                    # direction in these coordinates, and a mirrored right hand
                    # is just a valid left hand.
                    pts = np.array(lm) * [IMG_W, IMG_H]
                    person = "hagrid_" + ann["user_id"][:12]
                    # Upright hands (pointing up +-45 deg) are sources for rotation.
                    for variant, (sources, bone) in SYNTH_SOURCES.items():
                        if source in sources and 45 <= direction(pts, *bone) <= 135:
                            synth_pool[variant].append((image_id, person, pts))
                    label = map_label(source, pts)
                    if label is None:
                        continue
                    tags = [image_id, "", person, label, "unknown", "unknown", 0, source, ""]
                    candidates[(label, source)].append(tags + to_coords(pts))

    print(f"skipped {skipped_no_user} images without user_id")
    rows = []
    for our_class in ["open_palm", "none"]:
        groups = {s: r for (c, s), r in candidates.items() if c == our_class}
        for source, group in sorted(groups.items()):
            if our_class == "none":
                cap = MAX_NONE_NO_GESTURE if source == "no_gesture" else MAX_NONE_PER_SOURCE
            else:  # split the class cap proportionally over its source classes
                total = sum(len(g) for g in groups.values())
                cap = round(MAX_PER_CLASS[our_class] * len(group) / total)
            take = rng.choice(len(group), size=min(cap, len(group)), replace=False)
            rows += [group[i] for i in sorted(take)]
            print(f"{our_class:12s} <- {source:16s} {len(take):6d} of {len(group)}")

    # Synthetic points: each sampled upright hand is rotated to ONE direction
    # (random right/left), so no hand appears as both point_right and point_left.
    for variant, (_, bone) in SYNTH_SOURCES.items():
        pool = synth_pool[variant]
        for i in rng.choice(len(pool), size=min(SYNTH_PER_VARIANT, len(pool)), replace=False):
            image_id, person, pts = pool[i]
            label = "point_right" if rng.random() < 0.5 else "point_left"
            target = (0 if label == "point_right" else 180) + rng.uniform(-1, 1) * SYNTH_JITTER_DEG
            tags = [image_id + "_" + variant, "", person, label, "unknown", "unknown", 0, variant, ""]
            rows.append(tags + to_coords(rotate_to(pts, bone, target)))
        print(f"point_*      <- {variant:16s} {min(SYNTH_PER_VARIANT, len(pool)):6d} of {len(pool)} upright hands")

    df = pd.DataFrame(rows, columns=TAG_COLUMNS + COORD_COLUMNS)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(df)} hands from {df['person'].nunique()} people")
    print(df["gesture"].value_counts().to_string())


if __name__ == "__main__":
    main()
