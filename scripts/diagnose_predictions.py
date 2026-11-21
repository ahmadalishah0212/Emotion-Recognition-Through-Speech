"""Audit real audio inputs and saved-model predictions without changing models."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.dataset_loader import scan_dataset
from src.feature_extraction import extract_features, load_audio, FEATURE_NAMES
from src.predict import predict_audio, load_selected


def main():
    rows, _ = scan_dataset()
    model, info = load_selected()
    actors = ("01", "05", "10")
    emotions = ("happy", "sad", "angry", "neutral", "fearful", "calm", "disgust", "surprised")
    selected = []
    for actor in actors:
        for emotion in emotions:
            selected.append(next(row for row in rows if row["actor"] == actor and row["emotion"] == emotion))

    records, vectors = [], []
    for row in selected:
        path = Path(row["path"])
        raw = path.read_bytes()
        signal, sr = load_audio(raw)
        vector = extract_features(raw)
        # Training calls extract_features(path); the app calls extract_features(bytes).
        path_vector = extract_features(path)
        result = predict_audio(raw)
        assert vector.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(vector).all()
        assert np.allclose(vector, path_vector, rtol=1e-5, atol=1e-4)
        assert result["emotion"] == max(result["probabilities"], key=result["probabilities"].get)
        records.append({
            "file": str(path), "actor": row["actor"], "actual": row["emotion"],
            "sha256": sha256(raw).hexdigest(), "duration_seconds": round(len(signal) / sr, 3),
            "sample_rate": sr, "signal_shape": list(signal.shape),
            "signal_min": round(float(signal.min()), 4), "signal_max": round(float(signal.max()), 4),
            "feature_shape": list(vector.shape),
            "feature_first_five": [round(float(x), 3) for x in vector[:5]],
            "predicted": result["emotion"],
            "probabilities": {k: round(v, 6) for k, v in result["probabilities"].items()},
        })
        vectors.append(vector)

    distances = [float(np.linalg.norm(vectors[0] - vector)) for vector in vectors[1:]]
    pairwise_distances = [float(np.linalg.norm(vectors[i] - vectors[j]))
                          for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    score_labels = sorted(records[0]["probabilities"])
    scores = [np.array([record["probabilities"][label] for label in score_labels]) for record in records]
    pairwise_score_distances = [float(np.linalg.norm(scores[i] - scores[j]))
                                for i in range(len(scores)) for j in range(i + 1, len(scores))]
    assert min(pairwise_distances) > 0 and min(pairwise_score_distances) > 0
    confusion = json.loads(Path("results/metrics/svm.json").read_text(encoding="utf-8"))
    matrix = np.array(confusion["confusion_matrix"])
    test_predictions = dict(zip(confusion["labels"], matrix.sum(axis=0).astype(int).tolist()))
    report = {
        "model": info["name"], "model_feature_count": int(model.n_features_in_),
        "selected_files": len(records), "selected_prediction_counts": dict(Counter(r["predicted"] for r in records)),
        "unique_audio_hashes": len({r["sha256"] for r in records}),
        "held_out_test_prediction_counts": test_predictions,
        "minimum_feature_distance_from_first": round(min(distances), 4),
        "maximum_feature_distance_from_first": round(max(distances), 4),
        "minimum_pairwise_feature_distance": round(min(pairwise_distances), 4),
        "minimum_pairwise_score_distance": round(min(pairwise_score_distances), 6),
        "records": records,
    }
    output = Path("results/metrics/prediction_diagnostics.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    for record in records:
        print(Path(record["file"]).name, "actual:", record["actual"], "predicted:", record["predicted"])


if __name__ == "__main__":
    main()
