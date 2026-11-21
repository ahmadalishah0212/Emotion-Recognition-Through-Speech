"""Load the selected saved pipeline and predict from real audio."""
import json
from pathlib import Path
import joblib
import numpy as np
from .feature_extraction import extract_features, FEATURE_NAMES

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_selected(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    selection = model_dir / "selected.json"
    if not selection.exists():
        raise FileNotFoundError("Trained model is missing. Download RAVDESS and run python -m src.train first.")
    info = json.loads(selection.read_text(encoding="utf-8"))
    model_path = model_dir / info["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Selected model file is missing: {model_path}. Run python -m src.train.")
    model = joblib.load(model_path)
    if model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("Saved model expects a different feature version; retrain the models.")
    return model, info


def predict_audio(audio_bytes, model_dir=MODEL_DIR):
    """Return one prediction and its chart-ready score vector from current bytes."""
    model, info = load_selected(model_dir)
    features = extract_features(audio_bytes).reshape(1, -1)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Selected model does not provide class scores")
    scores = np.asarray(model.predict_proba(features)[0], dtype=np.float64)
    if len(scores) != len(model.classes_) or not np.isfinite(scores).all() or abs(float(scores.sum()) - 1) > 1e-4:
        raise ValueError("Selected model returned invalid class scores")
    probabilities = {str(label): float(score) for label, score in zip(model.classes_, scores)}
    label = str(model.predict(features)[0])
    if label != max(probabilities, key=probabilities.get):
        raise ValueError("Model class and score order disagree; result was not displayed")
    return {"emotion": label, "probabilities": probabilities, "model": info["name"]}
