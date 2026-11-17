"""Data collection tool: records hand landmarks (never images) to data/gestures.csv.

Example:
    python src/record.py --person zeynep --gesture point_right --hand right --distance near --session 1

Controls in the camera window:
    SPACE  start a take (countdown, then records --seconds)
    u      undo the last take of this run (deletes its rows from the CSV)
    q/Esc  quit and print the coverage summary

Each take gets its own recording_id. Frames inside one take are almost
identical, so later we treat a take (and a person) as a unit, not single frames.

Only print the coverage table:
    python src/record.py --summary
"""
import argparse
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from classifier import GESTURES
from landmarks import Camera, HandTracker, draw_landmarks

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "gestures.csv"
TAG_COLUMNS = ["recording_id", "timestamp", "person", "gesture", "hand", "distance",
               "session", "note", "mp_handedness"]
COORD_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]  # x0,y0,z0,...,z20

# The "none" class has to cover everything that is NOT a command, so it needs
# more (and more varied) data than each gesture. These tips are cycled on screen.
NONE_TIPS = [
    "talk with your hands", "touch / scratch your face", "hold a cup or pen",
    "point at the screen", "rest hand on desk / chin", "flat hand sideways (near-miss!)",
    "half-open hand, thumbs up", "fist, wave, adjust hair",
]
GESTURE_TIPS = ["vary it: tilt, rotate a bit", "move around the frame", "closer / further a little"]


def put_text(frame, text, row, color=(0, 255, 255), scale=0.8):
    cv2.putText(frame, text, (20, 40 + 38 * row), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def append_rows(rows):
    df = pd.DataFrame(rows, columns=TAG_COLUMNS + COORD_COLUMNS)
    CSV_PATH.parent.mkdir(exist_ok=True)
    # Write after every take, so a crash never loses more than one take.
    df.to_csv(CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)


def remove_recording(recording_id):
    df = pd.read_csv(CSV_PATH)
    df[df["recording_id"] != recording_id].to_csv(CSV_PATH, index=False)


def print_summary():
    if not CSV_PATH.exists():
        print("No data yet.")
        return
    df = pd.read_csv(CSV_PATH)
    print(f"\n=== {CSV_PATH.name}: {len(df)} frames, {df['recording_id'].nunique()} takes ===")

    # Frames per person x gesture, split by hand and distance: empty cells = coverage gaps.
    table = pd.pivot_table(df, index=["person", "gesture"], columns=["hand", "distance"],
                           values="x0", aggfunc="count", fill_value=0)
    # Show every hand x distance column, even ones nobody recorded yet (those are the gaps).
    all_cols = pd.MultiIndex.from_product([["left", "right"], ["near", "mid", "far"]],
                                          names=["hand", "distance"])
    table = table.reindex(columns=all_cols, fill_value=0)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\nFrames per person x gesture x (hand, distance):")
        print(table)
        print("\nTakes (recordings) per person x gesture:")
        print(df.groupby(["person", "gesture"])["recording_id"].nunique().unstack(fill_value=0))
        print("\nSessions per person:")
        print(df.groupby("person")["session"].nunique())

    # The "none" class should be the biggest class (target: ~2x the largest gesture),
    # because it has to cover every non-command hand pose.
    counts = df["gesture"].value_counts()
    biggest_gesture = counts.drop("none", errors="ignore").max()
    n_none = counts.get("none", 0)
    print("\nFrames per class:")
    print(counts.to_string())
    if biggest_gesture and n_none < 2 * biggest_gesture:
        print(f"-> 'none' has {n_none} frames; aim for ~{2 * biggest_gesture} "
              f"(2x the largest gesture class).")

    # Sanity check: does MediaPipe agree with the --hand tag? A low rate for a
    # take usually means a typo in --hand. Can't test the mirror logic here;
    # if it's ~0% everywhere, MediaPipe's labels are just flipped (harmless).
    agree = (df["mp_handedness"].str.lower() == df["hand"]).groupby(df["recording_id"]).mean()
    # Our mirror flip reverses MediaPipe's Left/Right labels. Whichever convention
    # most takes follow is "normal"; takes that go against it are suspicious.
    flipped = (agree < 0.5).mean() > 0.5
    suspicious = agree[agree >= 0.5] if flipped else agree[agree < 0.5]
    print(f"\nHand check: {len(agree) - len(suspicious)}/{len(agree)} takes consistent with "
          f"their --hand tag (MediaPipe labels {'reversed by mirror' if flipped else 'as-is'}).")
    if len(suspicious):
        print("Check these takes (possible wrong --hand):", ", ".join(suspicious.index))


def record(args):
    cam, tracker = Camera(args.camera), HandTracker()
    state, t_state = "ready", 0.0     # ready -> countdown -> recording -> ready
    rows, takes_this_run = [], []
    tips = NONE_TIPS if args.gesture == "none" else GESTURE_TIPS
    tag_line = (f"{args.person} | {args.gesture} | {args.hand} hand | "
                f"{args.distance} | session {args.session}")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            points = tracker.detect(frame)
            draw_landmarks(frame, points)
            now = time.monotonic()

            if state == "countdown":
                remaining = args.countdown - (now - t_state)
                put_text(frame, f"GET READY: {remaining:.0f}", 2, (0, 165, 255), 1.5)
                if remaining <= 0:
                    state, t_state, rows = "recording", now, []
                    recording_id = uuid.uuid4().hex[:8]

            elif state == "recording":
                elapsed = now - t_state
                if points is not None:  # skip frames without a hand
                    tags = [recording_id, datetime.now().isoformat(timespec="milliseconds"),
                            args.person, args.gesture, args.hand, args.distance,
                            args.session, args.note, tracker.last_handedness]
                    rows.append(tags + points.flatten().round(2).tolist())
                put_text(frame, f"REC {args.seconds - elapsed:4.1f}s   frames: {len(rows)}",
                         2, (0, 0, 255), 1.0)
                put_text(frame, f"tip: {tips[int(elapsed / 3) % len(tips)]}", 3)
                if elapsed >= args.seconds:
                    if rows:
                        append_rows(rows)
                        takes_this_run.append(recording_id)
                    print(f"Take {len(takes_this_run)}: saved {len(rows)} frames "
                          f"(recording_id={recording_id})")
                    state = "ready"

            else:  # ready
                put_text(frame, "SPACE = record   u = undo last take   q = quit", 2)
                put_text(frame, f"takes saved this run: {len(takes_this_run)}", 3)

            put_text(frame, tag_line, 0, scale=0.7)
            put_text(frame, "hand: yes" if points is not None else "hand: NO", 1,
                     (0, 255, 0) if points is not None else (0, 0, 255), 0.7)
            cv2.imshow("record (q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) and state != "recording":
                break
            if key == ord(" ") and state == "ready":
                state, t_state = "countdown", now
            if key == ord("u") and state == "ready" and takes_this_run:
                remove_recording(takes_this_run.pop())
                print("Last take removed.")
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Record hand landmarks for one gesture.")
    p.add_argument("--person", help="first name, lowercase (used for leave-one-person-out)")
    p.add_argument("--gesture", choices=GESTURES)
    p.add_argument("--hand", choices=["left", "right"])
    p.add_argument("--distance", choices=["near", "mid", "far"])
    p.add_argument("--session", type=int, help="free number; new number per sitting/day/room")
    p.add_argument("--note", default="", help='optional, e.g. "standing, dim light"')
    p.add_argument("--seconds", type=float, default=15, help="length of one take")
    p.add_argument("--countdown", type=float, default=3)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--summary", action="store_true", help="only print the coverage summary")
    args = p.parse_args()

    if not args.summary:
        missing = [n for n in ("person", "gesture", "hand", "distance", "session")
                   if getattr(args, n) is None]
        if missing:
            p.error("missing: " + ", ".join("--" + m for m in missing))
        args.person = args.person.strip().lower()  # "Zeynep" and "zeynep" = same person
        record(args)
    print_summary()


if __name__ == "__main__":
    main()
