"""Train-only candidate selection and untouched RAVDESS/TESS/EmoDB tests."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.dataset_sources import scan_ravdess, scan_tess, scan_emodb, validate_rows, summary
from src.feature_extraction import extract_features, extract_features_v2, FEATURE_NAMES, FEATURE_NAMES_V2
from src.train import TEST_ACTORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/expanded"
CACHE = ROOT / "data/cache"
LABELS = ("angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised")


def metrics(y_true, prediction, source_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, labels=source_labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, prediction)),
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
            "per_class": classification_report(y_true, prediction, labels=source_labels,
                                               output_dict=True, zero_division=0),
            "confusion_labels": list(LABELS),
            "confusion_matrix": confusion_matrix(y_true, prediction, labels=LABELS).tolist(),
            "prediction_counts": dict(Counter(map(str, prediction)))}


def features_for(rows, version):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"expanded_features_{version}.npz"
    hashes = [row["sha256"] for row in rows]
    if cache.exists():
        saved = np.load(cache)
        if list(saved["hashes"]) == hashes:
            print("using cached", version, flush=True)
            return saved["features"]
    extractor = extract_features if version == "v1" else extract_features_v2
    vectors = []
    for index, row in enumerate(rows, 1):
        vectors.append(extractor(row["path"]))
        if index % 250 == 0:
            print("features", version, index, "/", len(rows), flush=True)
    result = np.stack(vectors)
    np.savez_compressed(cache, hashes=hashes, features=result)
    return result


def build(name):
    if name == "baseline_v1" or name.startswith("svm"):
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(C=10, gamma="scale", random_state=42), method="sigmoid", cv=3))
    if name.startswith("logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42))
    if name == "forest_v1":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    raise ValueError(name)


def main():
    ravdess, ravdess_errors = scan_ravdess(ROOT / "data/raw")
    tess, tess_errors = scan_tess(ROOT / "data/external/tess")
    emodb, emodb_errors = scan_emodb(ROOT / "data/external/emodb")
    if len(ravdess) != 1439 or len(tess) != 2800 or len(emodb) != 454:
        raise ValueError("Expected complete validated RAVDESS, TESS and EmoDB datasets")
    rows, validation_errors = validate_rows(ravdess + tess + emodb)
    if validation_errors:
        raise ValueError(f"Invalid or duplicate audio: {validation_errors[:3]}")
    if {row["speaker"] for row in rows if row["dataset"] == "ravdess" and row["speaker"].split(":")[1] in TEST_ACTORS} & {
       row["speaker"] for row in rows if row["dataset"] == "ravdess" and row["speaker"].split(":")[1] not in TEST_ACTORS}:
        raise AssertionError("RAVDESS speaker leakage")
    labels = np.array([row["emotion"] for row in rows])
    speakers = np.array([row["speaker"] for row in rows])
    datasets = np.array([row["dataset"] for row in rows])
    rav_train = (datasets == "ravdess") & ~np.isin([speaker.split(":")[-1] for speaker in speakers], list(TEST_ACTORS))
    rav_test = (datasets == "ravdess") & ~rav_train
    tess_oaf = np.flatnonzero(speakers == "tess:OAF")
    tess_yaf = np.flatnonzero(speakers == "tess:YAF")
    emodb_test = np.flatnonzero(datasets == "emodb")
    rng = np.random.default_rng(42)
    tess_train = np.concatenate([rng.choice([i for i in tess_oaf if labels[i] == label], size=70, replace=False)
                                 for label in TESS_CLASSES])
    train_base = np.flatnonzero(rav_train)
    test_sets = {"ravdess_unseen_actors": np.flatnonzero(rav_test),
                 "tess_unseen_actor": tess_yaf, "emodb_unseen_dataset": emodb_test}
    if len(train_base) != 1079 or len(test_sets["ravdess_unseen_actors"]) != 360 or len(tess_train) != 490:
        raise AssertionError("Unexpected split counts")
    if set(speakers[np.concatenate([train_base, tess_train])]) & set(speakers[np.concatenate(list(test_sets.values()))]):
        raise AssertionError("A speaker appears in training and test")
    x_v1 = features_for(rows, "v1")
    x_v2 = features_for(rows, "v2")
    candidates = ["baseline_v1", "svm_expanded_v1", "logistic_expanded_v1", "forest_v1",
                  "svm_expanded_v2", "logistic_expanded_v2"]
    cv = []
    folds = list(GroupKFold(n_splits=3).split(train_base, groups=speakers[train_base]))
    for name in candidates:
        x = x_v2 if name.endswith("v2") else x_v1
        fold_scores = []
        for tr, val in folds:
            fit_indices = train_base[tr]
            if name != "baseline_v1":
                fit_indices = np.concatenate([fit_indices, tess_train])
            val_indices = train_base[val]
            if set(speakers[fit_indices]) & set(speakers[val_indices]):
                raise AssertionError("Actor overlap in CV")
            model = build(name)
            model.fit(x[fit_indices], labels[fit_indices])
            fold_scores.append(metrics(labels[val_indices], model.predict(x[val_indices]), LABELS)["macro_f1"])
        cv.append({"candidate": name, "fold_macro_f1": fold_scores, "mean_macro_f1": float(np.mean(fold_scores))})
        print("CV", name, cv[-1]["mean_macro_f1"], flush=True)
    winner = max(cv, key=lambda item: item["mean_macro_f1"])["candidate"]
    print("chosen from training CV", winner, flush=True)
    x_win = x_v2 if winner.endswith("v2") else x_v1
    fit_indices = train_base if winner == "baseline_v1" else np.concatenate([train_base, tess_train])
    final = build(winner)
    final.fit(x_win[fit_indices], labels[fit_indices])
    baseline = joblib.load(ROOT / "models/svm.joblib")
    evaluation = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for test_name, indices in test_sets.items():
        source_labels = sorted(set(labels[indices]))
        truth = labels[indices]
        evaluation[test_name] = {"clips": len(indices), "speakers": len(set(speakers[indices])),
                                 "source_labels": source_labels,
                                 "saved_baseline": metrics(truth, baseline.predict(x_v1[indices]), source_labels),
                                 "selected_candidate": metrics(truth, final.predict(x_win[indices]), source_labels)}
        print("TEST", test_name, "baseline", evaluation[test_name]["saved_baseline"]["macro_f1"],
              "candidate", evaluation[test_name]["selected_candidate"]["macro_f1"], flush=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, key, title in zip(axes, ("saved_baseline", "selected_candidate"), ("Saved baseline", winner)):
            matrix = np.array(evaluation[test_name][key]["confusion_matrix"])
            ax.imshow(matrix, cmap="Blues")
            ax.set(xticks=range(8), yticks=range(8), xticklabels=LABELS, yticklabels=LABELS,
                   xlabel="Predicted", ylabel="Actual", title=title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.suptitle(test_name)
        fig.tight_layout()
        fig.savefig(OUT / f"{test_name}_comparison.png", dpi=140)
        plt.close(fig)
    report = {"protocol": "RAVDESS six actors untouched; TESS OAF 70/class train, YAF test; EmoDB all unseen",
              "dataset_counts": {source: summary([row for row in rows if row["dataset"] == source])
                                 for source in ("ravdess", "tess", "emodb")},
              "dataset_scan_errors": {source: [{**error, "path": str(Path(error["path"]).relative_to(ROOT))}
                                               for error in errors]
                                      for source, errors in (("ravdess", ravdess_errors),
                                                             ("tess", tess_errors), ("emodb", emodb_errors))},
              "feature_count": {"v1": len(FEATURE_NAMES), "v2": len(FEATURE_NAMES_V2)},
              "tess_training_clips": len(tess_train), "train_speakers": sorted(set(speakers[fit_indices])),
              "test_speakers": {name: sorted(set(speakers[indices])) for name, indices in test_sets.items()},
              "training_cv": cv, "selected_candidate": winner, "evaluation": evaluation}
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(final, CACHE / "expanded_candidate.joblib")
    print("saved candidate only to ignored data/cache; selected deployed model unchanged", flush=True)


TESS_CLASSES = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")

if __name__ == "__main__":
    main()
