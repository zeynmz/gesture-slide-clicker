"""Bookkeeping for live tests with people who are NOT in the training data.

Used by live.py --test NAME. An operator watches and marks with keys on the
camera window:
    c  last fired action was correct
    f  last fired action was a FALSE trigger (fired, but nobody meant it,
       or the wrong direction)
    m  a gesture was MISSED (made deliberately, but nothing fired)
Fired actions the operator doesn't mark count as correct (so during a real
presentation the operator only has to press f and m); the summary shows how
many were unmarked so this assumption stays visible.

Results are appended to results/live_events.csv (every event) and
results/live_summary.csv (one row per test session).
"""
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
EVENTS_CSV = RESULTS / "live_events.csv"
SUMMARY_CSV = RESULTS / "live_summary.csv"


class LiveTest:
    def __init__(self, person, kind, settings):
        self.person, self.kind, self.settings = person.strip().lower(), kind, settings
        self.session = uuid.uuid4().hex[:8]
        self.start = time.monotonic()
        self.events = []  # dicts: time, seconds, event, mark

    def _add(self, event, mark):
        self.events.append({"session": self.session, "person": self.person, "kind": self.kind,
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "seconds": round(time.monotonic() - self.start, 1),
                            "event": event, "mark": mark})

    def fired(self, action):
        self._add(action, "unmarked")

    def key(self, ch):
        """Handle an operator key. Returns a short message for the terminal, or None."""
        if ch == "m":
            self._add("missed_gesture", "missed")
            return "marked: MISSED gesture"
        if ch in ("c", "f"):
            unmarked = [e for e in self.events if e["mark"] == "unmarked"]
            if not unmarked:
                return "nothing to mark (no unmarked action)"
            unmarked[-1]["mark"] = "correct" if ch == "c" else "false"
            return f"marked: {unmarked[-1]['event']} -> {unmarked[-1]['mark'].upper()}"
        return None

    def counts(self):
        marks = pd.Series([e["mark"] for e in self.events], dtype=object)
        n = {m: int((marks == m).sum()) for m in ("correct", "false", "missed", "unmarked")}
        n["recognized"] = n["correct"] + n["unmarked"]   # unmarked fired actions count as correct
        n["tried"] = n["recognized"] + n["missed"]
        n["minutes"] = (time.monotonic() - self.start) / 60
        return n

    def status_line(self):
        n = self.counts()
        return (f"TEST {self.person} ({self.kind}) {n['minutes']:.1f} min | ok {n['recognized']} "
                f"false {n['false']} missed {n['missed']} | c / f / m")

    def save(self):
        RESULTS.mkdir(exist_ok=True)
        n = self.counts()
        row = {"session": self.session, "date": datetime.now().isoformat(timespec="minutes"),
               "person": self.person, "kind": self.kind, "minutes": round(n["minutes"], 2),
               "tried": n["tried"], "recognized": n["recognized"], "missed": n["missed"],
               "false_triggers": n["false"], "unmarked": n["unmarked"], **self.settings}
        for path, df in [(EVENTS_CSV, pd.DataFrame(self.events)), (SUMMARY_CSV, pd.DataFrame([row]))]:
            if len(df):
                df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print_summary()


def print_summary():
    if not SUMMARY_CSV.exists():
        print("No live tests yet.")
        return
    s = pd.read_csv(SUMMARY_CSV)
    g = s.groupby(["person", "kind"])[["minutes", "tried", "recognized", "missed",
                                        "false_triggers", "unmarked"]].sum()
    g["recognized %"] = (100 * g["recognized"] / g["tried"].where(g["tried"] > 0)).round(0)
    g["false / min"] = (g["false_triggers"] / g["minutes"].where(g["minutes"] > 0)).round(2)
    total = g.sum(numeric_only=True)
    total["recognized %"] = round(100 * total["recognized"] / max(total["tried"], 1))
    total["false / min"] = round(total["false_triggers"] / max(total["minutes"], 1e-9), 2)
    g.loc[("ALL", ""), :] = total
    print("\n=== Live test summary (all sessions so far) ===")
    with pd.option_context("display.width", 200):
        print(g.round(2).to_string())
    print(f"\nFiles: {SUMMARY_CSV}, {EVENTS_CSV}")
