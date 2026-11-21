"""Audit saved models on exact uploads and actor-held-out RAVDESS speech.

Run: python scripts/deep_audio_diagnostic.py [external audio files ...]
The same 46-feature vectors are passed to all three saved models. Existing
training metrics and model files are never overwritten.
"""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from io import BytesIO
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.dataset_loader import scan_dataset
from src.feature_extraction import extract_features, load_audio, FEATURE_NAMES
from src.train import TEST_ACTORS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("svm", "logistic_regression", "random_forest")
OUT = ROOT / "results" / "diagnostics"


def model_scores(model, features):
    x = features.reshape(1, -1)
    labels = [str(label) for label in model.classes_]
    scores = [float(value) for value in model.predict_proba(x)[0]]
    ranked = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    predicted = str(model.predict(x)[0])
    if predicted != ranked[0][0]:
        raise AssertionError(f"{predicted} differs from top probability {ranked[0][0]}")
    return {"prediction": predicted, "probabilities": dict(zip(labels, scores)),
            "highest": ranked[0][1], "second": ranked[1][1],
            "margin": ranked[0][1] - ranked[1][1], "score_sum": sum(scores)}


def inspect_payload(payload, filename, truth, models):
    signal, sr = load_audio(payload)
    features = extract_features(payload)
    if features.shape != (len(FEATURE_NAMES),):
        raise AssertionError("Feature format mismatch")
    scores = {name: model_scores(model, features) for name, model in models.items()}
    repeat = model_scores(models["svm"], extract_features(payload))
    if repeat["probabilities"] != scores["svm"]["probabilities"]:
        raise AssertionError(f"Nonreproducible SVM result for {filename}")
    svm = models["svm"]
    scaled = svm.named_steps["standardscaler"].transform(features.reshape(1, -1))[0]
    folds = svm.named_steps["calibratedclassifiercv"].calibrated_classifiers_
    max_similarity = []
    for fold in folds:
        estimator = fold.estimator
        squared = np.sum((estimator.support_vectors_ - scaled) ** 2, axis=1)
        max_similarity.append(float(np.exp(-estimator._gamma * squared.min())))
    return {"file": str(filename), "truth": truth, "sha256": sha256(payload).hexdigest(),
            "bytes": len(payload), "duration_seconds": len(signal) / sr, "sample_rate": sr,
            "signal_shape": list(signal.shape), "signal_min": float(signal.min()),
            "signal_max": float(signal.max()), "feature_shape": list(features.shape),
            "feature_first_five": [float(value) for value in features[:5]],
            "models": scores, "repeat_identical": True,
            "svm_max_rbf_similarity_by_calibration_fold": max_similarity}, features


def inspect_audio(path, truth, models):
    path = Path(path)
    try:
        label = path.resolve().relative_to(ROOT)
    except ValueError:
        label = path.name
    return inspect_payload(path.read_bytes(), label, truth, models)


def synthetic_tone(frequency):
    samples = (0.3 * np.sin(2 * np.pi * frequency * np.arange(16000) / 16000) * 32767).astype("<i2")
    stream = BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external", nargs="*", type=Path)
    args = parser.parse_args()
    models = {name: joblib.load(ROOT / "models" / f"{name}.joblib") for name in MODELS}
    for name, model in models.items():
        if model.n_features_in_ != len(FEATURE_NAMES):
            raise AssertionError(f"{name}: saved feature count differs")
        print(name, "classes", list(map(str, model.classes_)))
    selected = json.loads((ROOT / "models/selected.json").read_text())
    if selected["name"] != "svm" or selected["file"] != "svm.joblib":
        raise AssertionError("Unexpected selected model")

    rows, errors = scan_dataset(ROOT / "data/raw")
    test_rows = [row for row in rows if row["actor"] in TEST_ACTORS]
    if len(test_rows) != 360 or any(row["actor"] not in TEST_ACTORS for row in test_rows):
        raise AssertionError("Held-out actor set differs from recorded baseline")
    # One of each emotion from three actors, including two held-out actors.
    sample_rows = []
    for actor in ("01", "05", "10"):
        for emotion in ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"):
            sample_rows.append(next(row for row in rows if row["actor"] == actor and row["emotion"] == emotion))
    detailed = []
    vectors = []
    for row in sample_rows:
        info, vector = inspect_audio(Path(row["path"]), row["emotion"], models)
        info["actor"] = row["actor"]
        detailed.append(info)
        vectors.append(vector)
    for path in args.external:
        info, vector = inspect_audio(path, None, models)
        info["actor"] = None
        detailed.append(info)
        vectors.append(vector)
    for frequency in (220, 660):
        info, vector = inspect_payload(synthetic_tone(frequency), f"synthetic_tone_{frequency}Hz.wav", None, models)
        info["actor"] = None
        detailed.append(info)
        vectors.append(vector)
    for index, item in enumerate(detailed):
        item["feature_differs_from_all_previous"] = all(not np.array_equal(vectors[index], earlier) for earlier in vectors[:index])
    if not all(item["feature_differs_from_all_previous"] for item in detailed):
        raise AssertionError("Two different selected files produced identical features")

    # Recompute independent predictions on every held-out actor clip, without fitting.
    x_test = np.stack([extract_features(Path(row["path"])) for row in test_rows])
    y_test = np.array([row["emotion"] for row in test_rows])
    labels = sorted(set(y_test))
    metrics = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        predicted = model.predict(x_test)
        probability = model.predict_proba(x_test)
        if not np.array_equal(predicted, model.classes_[np.argmax(probability, axis=1)]):
            raise AssertionError(f"{name}: labels differ from highest probabilities")
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="macro", zero_division=0)
        matrix = confusion_matrix(y_test, predicted, labels=labels)
        metrics[name] = {"accuracy": float(accuracy_score(y_test, predicted)), "macro_precision": float(precision),
                         "macro_recall": float(recall), "macro_f1": float(f1), "labels": labels,
                         "confusion_matrix": matrix.tolist(), "prediction_counts": dict(Counter(map(str, predicted)))}
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(8), yticks=range(8), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Actual", title=f"{name}: held-out actor confusion")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(8):
            for j in range(8):
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_confusion.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.25
    for index, name in enumerate(MODELS):
        counts = metrics[name]["prediction_counts"]
        ax.bar(np.arange(8) + index * width, [counts.get(label, 0) for label in labels], width, label=name)
    ax.set(xticks=np.arange(8) + width, xticklabels=labels, ylabel="Predictions on 360 held-out clips",
           title="Saved-model prediction distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "prediction_distribution.png", dpi=150)
    plt.close(fig)

    sanitized_errors = [{"path": str(Path(error["path"]).resolve().relative_to(ROOT)), "error": error["error"]} for error in errors]
    report = {"selected_model": selected, "held_out_actors": sorted(TEST_ACTORS),
              "held_out_count": len(test_rows), "dataset_scan_errors": sanitized_errors,
              "selected_inputs": detailed, "held_out_metrics": metrics}
    (OUT / "deep_audio_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "selected_inputs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "truth", "actor", "sha256", "duration_seconds", "sample_rate", "feature_shape",
                         "model", "prediction", "highest", "second", "margin", *labels])
        for item in detailed:
            for name in MODELS:
                result = item["models"][name]
                writer.writerow([item["file"], item["truth"], item["actor"], item["sha256"], item["duration_seconds"],
                                 item["sample_rate"], item["feature_shape"], name, result["prediction"],
                                 result["highest"], result["second"], result["margin"],
                                 *[result["probabilities"][label] for label in labels]])
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"])
        for name in MODELS:
            row = metrics[name]
            writer.writerow([name, row["accuracy"], row["macro_precision"], row["macro_recall"], row["macro_f1"]])
    print(json.dumps({"held_out_metrics": metrics, "selected_inputs": len(detailed)}, indent=2))


if __name__ == "__main__":
    main()
