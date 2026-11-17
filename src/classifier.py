"""The interface between the data/model side and the live side.

    clf = load_classifier()                      # or DummyClassifier() before a model exists
    name, confidence = clf.predict(landmarks)    # landmarks: (21, 3) array from landmarks.py

live.py only ever calls predict(), so it doesn't need to know which model,
features or training data are behind it.
"""
from pathlib import Path

GESTURES = ["point_right", "point_left", "open_palm", "none"]
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_model.joblib"


class DummyClassifier:
    """Placeholder for testing the live side without a model: always says 'none'."""

    def predict(self, landmarks):
        return "none", 1.0


class GestureClassifier:
    def __init__(self, bundle):
        self.model = bundle["model"]
        # Saved with the model, so live use computes exactly the features it was trained on.
        self.feature_config = bundle["feature_config"]
        self.description = bundle["description"]

    def predict(self, landmarks):
        from features import make_features
        proba = self.model.predict_proba(make_features(landmarks, self.feature_config))[0]
        best = proba.argmax()
        return str(self.model.classes_[best]), float(proba[best])


def load_classifier(path=MODEL_PATH):
    import joblib
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing. Run: python src/train.py")
    return GestureClassifier(joblib.load(path))
