"""Compare modest SVM settings with actor-group CV, then one held-out test.

The saved baseline is never replaced here. This is an audit experiment, not
automatic model selection for the app.
"""
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.spatial.distance import cdist

from src.dataset_loader import scan_dataset
from src.feature_extraction import extract_features, FEATURE_NAMES
from src.train import TEST_ACTORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/audio_tuning.json"
CANDIDATES = ((1, "scale"), (3, "scale"), (10, "scale"), (30, "scale"), (10, 0.01))


def build(c, gamma):
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(
        SVC(C=c, gamma=gamma, random_state=42), method="sigmoid", cv=3))


def measure(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_f1": float(f1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external", nargs="*", type=Path)
    args = parser.parse_args()
    rows, errors = scan_dataset(ROOT / "data/raw")
    cache = ROOT / "data/cache/ravdess_features.npz"
    if cache.exists():
        saved = np.load(cache)
        paths = [str(row["path"]) for row in rows]
        if list(saved["paths"]) != paths or saved["features"].shape != (len(rows), len(FEATURE_NAMES)):
            raise ValueError("Cached feature matrix does not match current dataset")
        x = saved["features"]
    else:
        x = np.stack([extract_features(row["path"]) for row in rows])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, paths=[row["path"] for row in rows], features=x)
    y = np.array([row["emotion"] for row in rows])
    actors = np.array([row["actor"] for row in rows])
    test = np.isin(actors, list(TEST_ACTORS))
    if test.sum() != 360 or set(actors[test]) & set(actors[~test]):
        raise AssertionError("Actor-independent test split changed")
    x_train, y_train, groups = x[~test], y[~test], actors[~test]
    folds = list(GroupKFold(n_splits=3).split(x_train, y_train, groups))
    candidates = []
    for c, gamma in CANDIDATES:
        fold_metrics = []
        for train_idx, val_idx in folds:
            if set(groups[train_idx]) & set(groups[val_idx]):
                raise AssertionError("Group CV actor overlap")
            model = build(c, gamma)
            model.fit(x_train[train_idx], y_train[train_idx])
            fold_metrics.append(measure(y_train[val_idx], model.predict(x_train[val_idx])))
        candidates.append({"c": c, "gamma": gamma, "folds": fold_metrics,
                           "mean_macro_f1": float(np.mean([fold["macro_f1"] for fold in fold_metrics]))})
        print("candidate", c, gamma, "group_cv_macro_f1", candidates[-1]["mean_macro_f1"], flush=True)
    winner = max(candidates, key=lambda item: item["mean_macro_f1"])
    final = build(winner["c"], winner["gamma"])
    final.fit(x_train, y_train)
    prediction = final.predict(x[test])
    baseline = joblib.load(ROOT / "models/svm.joblib")
    baseline_prediction = baseline.predict(x[test])
    labels = sorted(set(y))
    output = {"method": "3-fold actor GroupKFold on 18 training actors; untouched 6-actor test set",
              "feature_count": len(FEATURE_NAMES), "train_count": int((~test).sum()), "test_count": int(test.sum()),
              "candidates": candidates, "chosen_by_cv": {"c": winner["c"], "gamma": winner["gamma"]},
              "candidate_test": measure(y[test], prediction),
              "candidate_classification_report": classification_report(y[test], prediction, labels=labels, output_dict=True, zero_division=0),
              "candidate_confusion_matrix": confusion_matrix(y[test], prediction, labels=labels).tolist(),
              "saved_baseline_test": measure(y[test], baseline_prediction),
              "saved_baseline_matches_recorded": np.array_equal(baseline_prediction, baseline.predict(x[test])),
              "dataset_scan_errors": [{"path": str(Path(error["path"]).resolve().relative_to(ROOT)),
                                       "error": error["error"]} for error in errors]}
    test_actors = actors[test]
    actor_ids = sorted(set(test_actors))
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        sampled_actors = rng.choice(actor_ids, size=len(actor_ids), replace=True)
        indices = np.concatenate([np.flatnonzero(test_actors == actor) for actor in sampled_actors])
        candidate_f1 = measure(y[test][indices], prediction[indices])["macro_f1"]
        baseline_f1 = measure(y[test][indices], baseline_prediction[indices])["macro_f1"]
        differences.append(candidate_f1 - baseline_f1)
    output["macro_f1_difference"] = output["candidate_test"]["macro_f1"] - output["saved_baseline_test"]["macro_f1"]
    output["actor_bootstrap_95pct_interval_for_difference"] = list(map(float, np.percentile(differences, [2.5, 97.5])))
    scaler = StandardScaler().fit(x_train)
    scaled_train = scaler.transform(x_train)
    heldout_nearest = cdist(scaler.transform(x[test]), scaled_train).min(axis=1)
    output["feature_distance_audit"] = {
        "method": "Euclidean nearest-training-clip distance after train-only StandardScaler; descriptive, not a calibrated OOD detector",
        "heldout_percentiles_50_90_95_99_100": list(map(float, np.percentile(heldout_nearest, [50, 90, 95, 99, 100]))),
        "external": []}
    for path in args.external:
        vector = extract_features(path.read_bytes()).reshape(1, -1)
        nearest = float(cdist(scaler.transform(vector), scaled_train).min())
        output["feature_distance_audit"]["external"].append({"file": path.name, "nearest_train_distance": nearest})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("selected", output["chosen_by_cv"], "candidate_test", output["candidate_test"],
          "saved_baseline_test", output["saved_baseline_test"], flush=True)


if __name__ == "__main__":
    main()
