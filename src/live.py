"""Real-time gesture slide clicker.

    python src/live.py --dry-run          # prints actions, presses nothing (try this first)
    python src/live.py                    # really presses the arrow keys
    python src/live.py --threshold 0.9 --frames 20 --cooldown 1.5

After starting, click on your presentation window so it receives the key
presses. Quit with q (with the camera window focused) or Ctrl+C in the terminal.

Smoothing, because single frames are noisy (Phase 3: ~12% of "none" frames
look like a point on their own):
  - a gesture only counts when predicted with confidence >= --threshold
    for --frames frames IN A ROW (15 frames ~ 0.5 s at 30 fps)
  - it fires ONCE per hold: you must lower/change your hand before the same
    gesture can fire again (otherwise holding a point would skip many slides)
  - after any action, nothing fires for --cooldown seconds
open_palm toggles pause; while paused, only open_palm (to resume) is acted on.

Live test mode (people NOT in the training data), see live_test.py:
    python src/live.py --test ayse --kind gestures --dry-run
    python src/live.py --test ayse --kind talking --dry-run
    python src/live.py --test ayse --kind presentation --keynote
The operator marks with c (correct) / f (false trigger) / m (missed) on the
camera window. --keynote drives Keynote directly (AppleScript), so the camera
window can keep the keyboard focus for marking while the slides still move.
    python src/live.py --summary          # print all live test results
"""
import argparse
import subprocess
import sys
import time

import cv2

from classifier import DummyClassifier, load_classifier
from landmarks import Camera, HandTracker, draw_landmarks
from live_test import LiveTest, print_summary

KEYS = {"point_right": "right", "point_left": "left"}
KEYNOTE = {"point_right": "show next", "point_left": "show previous"}


class Smoother:
    """Turns a stream of per-frame (gesture, confidence) into rare, deliberate actions."""

    def __init__(self, threshold=0.8, frames=15, cooldown=1.0):
        self.threshold, self.frames, self.cooldown = threshold, frames, cooldown
        self.candidate, self.streak = None, 0
        self.last_fired, self.last_fire_time = None, -1e9
        self.absent = 0  # frames since the last fired gesture was last seen
        self.paused = False

    def update(self, gesture, confidence, now):
        """Feed one frame. Returns the gesture to act on, or None."""
        sure = gesture if (gesture != "none" and confidence >= self.threshold) else None
        # Re-arm: the last gesture may fire again only after the hand showed something
        # else for half of --frames. One flickering frame mid-hold must not count.
        self.absent = 0 if sure == self.last_fired else self.absent + 1
        if self.absent >= self.frames // 2:
            self.last_fired = None
        # Count consecutive confident frames of the same gesture.
        self.streak = self.streak + 1 if (sure is not None and sure == self.candidate) else int(sure is not None)
        self.candidate = sure

        if (sure is None or self.streak < self.frames or sure == self.last_fired
                or now - self.last_fire_time < self.cooldown):
            return None
        if self.paused and sure != "open_palm":
            return None  # paused: ignore everything except the resume gesture
        self.last_fired, self.last_fire_time, self.streak = sure, now, 0
        if sure == "open_palm":
            self.paused = not self.paused
        return sure

    def progress(self):
        """0..1, how close the current candidate is to firing (for the overlay bar)."""
        return min(self.streak / self.frames, 1.0)


def send(action, args):
    """Turn a point gesture into a slide change."""
    if args.dry_run:
        return
    if args.keynote:
        # Keynote's own AppleScript commands work without Keynote having keyboard focus.
        subprocess.run(["osascript", "-e", f'tell application "Keynote" to {KEYNOTE[action]}'],
                       check=False)
    else:
        import pyautogui  # macOS needs Accessibility permission for this
        pyautogui.press(KEYS[action])


def draw_overlay(frame, gesture, confidence, smoother, last_action, test=None):
    h, w = frame.shape[:2]
    state, color = ("PAUSED", (0, 0, 255)) if smoother.paused else ("ACTIVE", (0, 200, 0))
    cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)  # dark band so text is readable
    if test:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, test.status_line(), (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 0), 2)
    cv2.putText(frame, state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"{gesture}  {confidence:.2f}", (260, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"last action: {last_action}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)
    # Progress bar: fills while the same confident gesture is held; full = fires.
    bar_w = int((w - 40) * smoother.progress())
    cv2.rectangle(frame, (20, 118), (20 + bar_w, 126), (0, 255, 255), -1)


def main():
    p = argparse.ArgumentParser(description="Control slides with hand gestures.")
    p.add_argument("--threshold", type=float, default=0.8, help="min confidence per frame")
    p.add_argument("--frames", type=int, default=15, help="consecutive confident frames to fire")
    p.add_argument("--cooldown", type=float, default=1.0, help="seconds after an action")
    p.add_argument("--dry-run", action="store_true", help="print actions, don't press keys")
    p.add_argument("--dummy", action="store_true", help="use DummyClassifier (no model needed)")
    p.add_argument("--scale", type=float, default=0.5, help="preview window size (fraction)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--keynote", action="store_true",
                   help="macOS only: control Keynote via AppleScript instead of pressing arrow keys")
    p.add_argument("--test", metavar="NAME", help="live test mode for this (unseen) person")
    p.add_argument("--kind", default="gestures", choices=["gestures", "talking", "presentation"],
                   help="gestures: called-out gestures; talking: no gestures, count false "
                        "triggers; presentation: a real talk")
    p.add_argument("--summary", action="store_true", help="only print live test results")
    args = p.parse_args()
    if args.keynote and sys.platform != "darwin":
        p.error("--keynote only works on macOS (AppleScript). Use plain arrow keys instead.")
    if args.summary:
        print_summary()
        return

    clf = DummyClassifier() if args.dummy else load_classifier()
    print("Model:", getattr(clf, "description", "dummy"))
    smoother = Smoother(args.threshold, args.frames, args.cooldown)
    test = None
    if args.test:
        test = LiveTest(args.test, args.kind, {"threshold": args.threshold, "frames": args.frames,
                                               "cooldown": args.cooldown})
        print("TEST MODE: operator keys on the camera window: c = correct, "
              "f = false trigger, m = missed gesture, q = finish and save")
    cam, tracker = Camera(args.camera), HandTracker()
    last_action = "-"
    window = "gesture clicker (q to quit)"
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            points = tracker.detect(frame)
            gesture, conf = clf.predict(points) if points is not None else ("none", 1.0)
            now = time.monotonic()

            action = smoother.update(gesture, conf, now)
            if action:
                if action == "open_palm":
                    last_action = "PAUSE" if smoother.paused else "RESUME"
                else:
                    last_action = f"{KEYS[action]} arrow"
                    send(action, args)
                if test:
                    test.fired(last_action)
                print(f"{time.strftime('%H:%M:%S')}  {last_action}"
                      f"{'  (dry run)' if args.dry_run else ''}")

            draw_landmarks(frame, points)
            draw_overlay(frame, gesture, conf, smoother, last_action, test)
            small = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            cv2.imshow(window, small)
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)  # stay above the slides
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if test and key != 255:
                msg = test.key(chr(key))
                if msg:
                    print(f"{time.strftime('%H:%M:%S')}  {msg}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()
        if test:
            test.save()


if __name__ == "__main__":
    main()
