"""Inspect exact uploaded audio files from the command line (no app state)."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf

from src.feature_extraction import extract_features, load_audio
from src.predict import predict_audio


def inspect(path):
    raw = path.read_bytes()
    try:
        original = sf.info(path)
        original_rate, original_seconds = original.samplerate, original.duration
    except RuntimeError:
        original_rate, original_seconds = None, None
    signal, sample_rate = load_audio(raw)
    features = extract_features(raw)
    prediction = predict_audio(raw)
    return {
        "file": str(path), "bytes": len(raw), "sha256": sha256(raw).hexdigest(),
        "original_rate": original_rate, "original_seconds": original_seconds,
        "processed_rate": sample_rate, "processed_seconds": round(len(signal) / sample_rate, 3),
        "samples": int(len(signal)), "signal_min": float(signal.min()), "signal_max": float(signal.max()),
        "feature_shape": list(features.shape), "feature_first_five": [round(float(v), 3) for v in features[:5]],
        "prediction": prediction,
    }, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    previous = None
    for path in args.files:
        info, features = inspect(path)
        if previous is not None:
            info["feature_distance_from_previous"] = float(np.linalg.norm(features - previous))
        previous = features
        print(info)


if __name__ == "__main__":
    main()
