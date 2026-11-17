"""Offline evaluation of the chosen setup with leave-one-person-out.

    python src/evaluate.py                     # setup of the saved model (kNN, own + HaGRID palm/none)
    python src/evaluate.py --points rot_palm   # same, with the synthetic flat points added

Writes:
    models/confusion_matrix.png   all 3 held-out people + each person separately
    models/evaluation.md          confusion matrix, false-trigger analysis, threshold table

Each person's frames are predicted by a model that never saw that person, so the
matrix shows what to expect for a NEW presenter (in our room / camera, though).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window needed, we only save a PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from classifier import GESTURES
from features import coords_from_df, make_features
from train import HAGRID_POINTS, MODELS, hagrid_subset, load_data

ROOT = Path(__file__).resolve().parent.parent
PNG_PATH = ROOT / "models" / "confusion_matrix.png"
MD_PATH = ROOT / "models" / "evaluation.md"
# Single-hue sequential ramp (light = near 0, dark = high), same scale in every panel.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#1f2328", "#5f6670"


def lopo_predictions(model_name, points, extra):
    """Predict every own frame with a model trained without that frame's person."""
    own, hagrid = load_data()
    hg = hagrid_subset(hagrid, points)
    cfg = {"extra": extra}
    X_own, y_own = make_features(coords_from_df(own), cfg), own["gesture"].to_numpy()
    X_hg, y_hg = make_features(coords_from_df(hg), cfg), hg["gesture"].to_numpy()
    pred, conf = np.empty(len(own), dtype=object), np.zeros(len(own))
    for person in sorted(own["person"].unique()):
        test = (own["person"] == person).to_numpy()
        model = MODELS[model_name]().fit(np.vstack([X_own[~test], X_hg]),
                                         np.concatenate([y_own[~test], y_hg]))
        proba = model.predict_proba(X_own[test])
        pred[test] = model.classes_[proba.argmax(1)]
        conf[test] = proba.max(1)
    own["pred"], own["conf"] = pred, conf
    return own


def confusion(df):
    """Counts, rows = true class, columns = predicted class, fixed class order."""
    return pd.crosstab(df["gesture"], df["pred"]).reindex(index=GESTURES, columns=GESTURES,
                                                          fill_value=0)


def draw_matrix(ax, counts, title):
    pct = counts.div(counts.sum(axis=1), axis=0) * 100  # row % = recall per true class
    ax.imshow(pct.to_numpy(), cmap=BLUES, vmin=0, vmax=100)
    for i in range(len(GESTURES)):
        for j in range(len(GESTURES)):
            v = pct.iat[i, j]
            ax.text(j, i, f"{v:.0f}%\n{counts.iat[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else INK)  # light text only on dark cells
    ax.set_xticks(range(len(GESTURES)), [g.replace("_", "\n") for g in GESTURES], fontsize=8, color=MUTED)
    ax.set_yticks(range(len(GESTURES)), [g.replace("_", "\n") for g in GESTURES], fontsize=8, color=MUTED)
    ax.set_xlabel("predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("true", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def main():
    p = argparse.ArgumentParser(description="Leave-one-person-out confusion matrix.")
    p.add_argument("--model", default="kNN", choices=list(MODELS))
    p.add_argument("--points", default="none", choices=list(HAGRID_POINTS),
                   help="which synthetic HaGRID pointing hands to add to training")
    p.add_argument("--no-extra", action="store_true", help="coordinates only, no extra features")
    args = p.parse_args()

    df = lopo_predictions(args.model, args.points, not args.no_extra)
    people = sorted(df["person"].unique())
    setup = f"{args.model}, own + HaGRID (points={args.points}), extra features={not args.no_extra}"

    fig, axes = plt.subplots(1, 1 + len(people), figsize=(4.2 * (1 + len(people)), 4.4))
    draw_matrix(axes[0], confusion(df), "all 3 held-out people")
    for ax, person in zip(axes[1:], people):
        draw_matrix(ax, confusion(df[df["person"] == person]), f"held out: {person}")
    fig.suptitle(f"Leave-one-person-out confusion matrices (row % and frame counts)\n{setup}",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=130, facecolor="white", bbox_inches="tight")

    is_none = df["gesture"] == "none"
    is_point = df["gesture"].str.startswith("point")
    pred_point = df["pred"].str.startswith("point")
    lines = [f"# Evaluation ({setup})\n",
             "Leave-one-person-out: every frame is predicted by a model that never saw that person.\n",
             "## Confusion matrix (all people, frame counts; rows = true, columns = predicted)\n",
             "```\n" + confusion(df).to_string() + "\n```\n",
             "## Where do the 'none' frames go? (% of each activity's frames)\n",
             "A 'none' frame predicted as a point is a potential false trigger.\n",
             "```\n" + (pd.crosstab(df.loc[is_none, "note"], df.loc[is_none, "pred"],
                                   normalize="index") * 100).round(0).to_string() + "\n```\n",
             "## Confidence threshold trade-off (per frame, before the K-frames smoothing)\n",
             "```\nthreshold  none->point (false-trigger rate)  points recognized"]
    for th in [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fire = pred_point & (df["conf"] >= th)
        lines.append(f"  {th:.1f}       {fire[is_none].mean():6.1%}                     "
                     f"{(fire & (df['pred'] == df['gesture']))[is_point].mean():6.1%}")
    lines.append("```\n")
    lines.append("Precision of point_left/right looks high (~0.97) only because our test data has\n"
                 "about twice as many pointing frames as 'none' frames. In a real talk almost every\n"
                 "frame is 'none', so the none->point rate is the number that predicts false triggers.\n")
    MD_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved {PNG_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
