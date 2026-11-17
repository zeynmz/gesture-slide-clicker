"""Train and compare gesture classifiers, then save the best one.

The real question is "does it work for someone it has never seen?", so every
number here comes from LEAVE-ONE-PERSON-OUT on our own 3 people: train without
person P, test on P, repeat for each P. HaGRID people are only ever used for
training, never for testing (they are not in our camera setup).

With 3 people there are only 3 test folds. We print every fold plus the mean
and the min-max range, and round to 2 decimals: the differences between
experiments are only meaningful if they are larger than the fold-to-fold spread.

Experiments (results also written to models/results.md):
  1. models:   kNN vs Random Forest vs MLP          (own + HaGRID palm/none)
  2. data:     own only / HaGRID only / own + HaGRID, with and without the
               synthetic rotated pointing hands (rot_one, rot_palm)
  3. features: normalized coords only vs coords + extra features
  4. splits:   random frames vs random takes vs persons (why random splits lie)
Then the best setup is retrained on all data and saved to models/gesture_model.joblib.

Run:  python src/train.py          (takes a few minutes)
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifier import GESTURES
from features import DEFAULT_CONFIG, coords_from_df, make_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "gesture_model.joblib"
RESULTS_PATH = ROOT / "models" / "results.md"

MODELS = {
    # kNN and MLP are distance/gradient based, so they need standardized features.
    "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                   n_jobs=-1, random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=0)),
}
# HaGRID subsets. "none" = no synthetic pointing hands, only palm + none.
HAGRID_POINTS = {"none": [], "rot_one": ["rot_one"], "rot_palm": ["rot_palm"]}

report_lines = []


def log(text=""):
    """Print and keep for models/results.md."""
    print(text)
    report_lines.append(text)


def log_table(df):
    log("```\n" + df.to_string() + "\n```")


def load_data():
    own = pd.read_csv(ROOT / "data" / "gestures.csv")
    own["note"] = own["note"].fillna("")
    hagrid = pd.read_csv(ROOT / "data" / "hagrid.csv")
    return own, hagrid


def hagrid_subset(hagrid, points):
    synthetic = hagrid["note"].str.startswith("rot_")
    return hagrid[~synthetic | hagrid["note"].isin(HAGRID_POINTS[points])]


def metrics(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=GESTURES, output_dict=True,
                                zero_division=0)
    m = {"acc": accuracy_score(y_true, y_pred), "macroF1": rep["macro avg"]["f1-score"]}
    for c in GESTURES:
        m[f"{c} P"] = rep[c]["precision"]
        m[f"{c} R"] = rep[c]["recall"]
    return m


def leave_one_person_out(model_name, own, hagrid=None, use_own=True, config=DEFAULT_CONFIG):
    """Train on (other own people + optional HaGRID), test on each own person in turn."""
    X_own, y_own = make_features(coords_from_df(own), config), own["gesture"].to_numpy()
    X_hg = make_features(coords_from_df(hagrid), config) if hagrid is not None else None
    rows, model = {}, None
    for person in sorted(own["person"].unique()):
        test = (own["person"] == person).to_numpy()
        parts_X, parts_y = [], []
        if use_own:
            parts_X.append(X_own[~test]); parts_y.append(y_own[~test])
        if X_hg is not None:
            parts_X.append(X_hg); parts_y.append(hagrid["gesture"].to_numpy())
        # Without own data the training set is the same in every fold: fit once.
        if model is None or use_own:
            model = MODELS[model_name]().fit(np.vstack(parts_X), np.concatenate(parts_y))
        pred = model.predict(X_own[test])
        rows[person] = metrics(y_own[test], pred)
        # Pointing recall split by pointing style (our --note tag).
        for style in ["index", "flat"]:
            sel = (own["note"].to_numpy()[test] == style)
            rows[person][f"point R ({style})"] = np.mean(pred[sel] == y_own[test][sel])
    return summarize(pd.DataFrame(rows).T)


def summarize(per_fold):
    """Per-fold rows + mean + range (max - min) so the spread is always visible."""
    out = per_fold.copy()
    out.loc["MEAN"] = per_fold.mean()
    out.loc["range"] = per_fold.max() - per_fold.min()
    return out.round(2)


def short(table, cols=("acc", "macroF1", "point_right P", "point_left P",
                       "point R (index)", "point R (flat)")):
    return table[list(cols)]


def experiment_models(own, hagrid):
    log("## 1. Model comparison (train: own + HaGRID palm/none, test: held-out person)\n")
    results = {}
    for name in MODELS:
        t = time.time()
        table = leave_one_person_out(name, own, hagrid_subset(hagrid, "none"))
        results[name] = table
        log(f"### {name}  ({time.time() - t:.0f} s)")
        log_table(table)
    best = max(results, key=lambda n: results[n].loc["MEAN", "macroF1"])
    log(f"\nBest model by mean macro F1: **{best}**\n")
    return best


def experiment_data(own, hagrid, model):
    log(f"## 2. Training data comparison ({model})\n")
    setups = {"own only": (True, None)}
    for pts in HAGRID_POINTS:
        setups[f"HaGRID only, points={pts}"] = (False, pts)
        setups[f"own + HaGRID, points={pts}"] = (True, pts)
    summary = {}
    for name, (use_own, pts) in setups.items():
        if not use_own and pts == "none":
            continue  # no pointing data at all -> can't learn point_left/right
        hg = hagrid_subset(hagrid, pts) if pts is not None else None
        table = leave_one_person_out(model, own, hg, use_own=use_own)
        log(f"### {name}")
        log_table(short(table))
        summary[name] = (use_own, pts, table.loc["MEAN"])
    overview = pd.DataFrame({k: v[2] for k, v in summary.items()}).T
    log("### Overview (means over the 3 people)")
    log_table(short(overview).round(2))
    best = max(summary, key=lambda k: summary[k][2]["macroF1"])
    log(f"\nBest data setup by mean macro F1: **{best}**\n")
    return summary[best][0], summary[best][1]


def experiment_features(own, hagrid, model, use_own, pts):
    log("## 3. Feature comparison\n")
    hg = hagrid_subset(hagrid, pts) if pts is not None else None
    best_cfg, best_f1 = None, -1
    for cfg in [{"extra": False}, {"extra": True}]:
        table = leave_one_person_out(model, own, hg, use_own=use_own, config=cfg)
        log(f"### extra features = {cfg['extra']}")
        log_table(short(table))
        if table.loc["MEAN", "macroF1"] > best_f1:
            best_cfg, best_f1 = cfg, table.loc["MEAN", "macroF1"]
    return best_cfg


def experiment_splits(own, model):
    log(f"## 4. Why a random split overestimates ({model}, own data only)\n")
    X, y = make_features(coords_from_df(own)), own["gesture"].to_numpy()
    rows = {}
    # a) random frames: neighbouring frames of the same take land in train AND test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    rows["random frames (80/20)"] = metrics(yte, MODELS[model]().fit(Xtr, ytr).predict(Xte))
    # b) whole takes held out: the person was still seen (in other takes)
    fold = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=own["recording_id"]):
        fold.append(metrics(y[te], MODELS[model]().fit(X[tr], y[tr]).predict(X[te])))
    rows["random takes (5-fold)"] = pd.DataFrame(fold).mean().to_dict()
    # c) whole person held out: the honest one
    rows["held-out person (3-fold)"] = leave_one_person_out(model, own).loc["MEAN"].to_dict()
    log_table(pd.DataFrame(rows).T[["acc", "macroF1", "point_right P", "point_left P"]].round(3))
    log("\nFrames of one take are nearly identical, so with a random frame split the test\n"
        "set contains near-copies of training frames: the model is tested on things it\n"
        "has (almost) memorized. Holding out whole people is what the live clicker faces.\n")


def main():
    own, hagrid = load_data()
    log("# Training results\n")
    log(f"Own data: {len(own)} frames, people: {', '.join(sorted(own['person'].unique()))}. "
        f"HaGRID: {len(hagrid)} hands from {hagrid['person'].nunique()} people.\n")
    log("Every number: leave-one-person-out over our 3 people (3 folds). "
        "P = precision, R = recall. 'range' = max - min over the folds.\n")

    model = experiment_models(own, hagrid)
    use_own, pts = experiment_data(own, hagrid, model)
    config = experiment_features(own, hagrid, model, use_own, pts)
    experiment_splits(own, model)

    # Final model: the best setup trained on ALL people (no one held out).
    hg = hagrid_subset(hagrid, pts) if pts is not None else None
    parts = ([own] if use_own else []) + ([hg] if hg is not None else [])
    train = pd.concat(parts)
    final = MODELS[model]().fit(make_features(coords_from_df(train), config), train["gesture"])
    description = f"{model}, own={use_own}, HaGRID points={pts}, features={config}"
    joblib.dump({"model": final, "feature_config": config, "description": description}, MODEL_PATH)
    log(f"## Saved model\n\n{MODEL_PATH.name}: {description}, trained on {len(train)} rows.")
    RESULTS_PATH.write_text("\n".join(report_lines) + "\n")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
