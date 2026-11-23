"""Opt-in trace of current audio bytes and their displayed model scores."""
import json
from hashlib import sha256
from pathlib import Path
from .feature_extraction import extract_features, extract_features_v2, load_audio
from .predict import EXTRACTORS

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def trace_audio(payload, result, model_dir=MODEL_DIR):
    signal, sr = load_audio(payload)
    model_dir = Path(model_dir)
    selection = json.loads((model_dir / "selected.json").read_text(encoding="utf-8"))
    extractor = EXTRACTORS[selection.get("feature_version", "v1")][0]
    features = extractor(payload)
    model_file = model_dir / selection["file"]
    ranked = sorted(result["probabilities"].values(), reverse=True)
    return {"audio_sha256": sha256(payload).hexdigest(), "audio_bytes": len(payload),
            "duration_seconds": len(signal) / sr, "processed_sample_rate": sr,
            "signal_shape": list(signal.shape), "signal_min": float(signal.min()),
            "signal_max": float(signal.max()), "feature_version": selection.get("feature_version", "v1"),
            "feature_shape": list(features.shape),
            "feature_first_five": [float(value) for value in features[:5]],
            "model_file": str(model_file.resolve()), "model_sha256": sha256(model_file.read_bytes()).hexdigest(),
            "predicted_class": result["emotion"], "top_probability": ranked[0],
            "second_probability": ranked[1], "margin": ranked[0] - ranked[1],
            "probabilities": result["probabilities"]}
