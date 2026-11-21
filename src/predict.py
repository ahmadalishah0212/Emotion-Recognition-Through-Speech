"""Load the selected saved pipeline and predict from real audio."""
import json
from pathlib import Path
import joblib
from .feature_extraction import extract_features, FEATURE_NAMES

MODEL_DIR = Path("models")


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
    model, info = load_selected(model_dir)
    features = extract_features(audio_bytes).reshape(1, -1)
    label = str(model.predict(features)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = {str(k): float(v) for k, v in zip(model.classes_, model.predict_proba(features)[0])}
    return {"emotion": label, "probabilities": probabilities, "model": info["name"]}
