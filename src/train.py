"""Train three CPU-friendly classifiers with a held-out actor test set."""
from collections import Counter
import json
from pathlib import Path
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .dataset_loader import scan_dataset, dataset_summary
from .feature_extraction import extract_features, FEATURE_NAMES

SEED = 42
TEST_ACTORS = {"05", "10", "15", "20", "23", "24"}
MODELS = {
    "svm": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=SEED)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
}


def prepare_features(rows):
    features, valid_rows, errors = [], [], []
    for row in rows:
        try:
            features.append(extract_features(row["path"]))
            valid_rows.append(row)
        except (ValueError, OSError) as exc:
            errors.append({"path": row["path"], "error": str(exc)})
    if not features:
        raise ValueError("No usable audio after feature extraction")
    return np.stack(features), valid_rows, errors


def run_training(data_dir="data/raw"):
    rows, scan_errors = scan_dataset(data_dir)
    X, rows, feature_errors = prepare_features(rows)
    y = np.array([row["emotion"] for row in rows])
    test = np.array([row["actor"] in TEST_ACTORS for row in rows])
    if not test.any() or test.all():
        raise ValueError("Need recordings from both training and test actors; see TEST_ACTORS in src/train.py")
    labels = sorted(set(y))
    if set(y[test]) != set(y[~test]):
        raise ValueError("Every emotion must occur in both actor groups")
    for folder in ("models", "results/metrics", "results/figures", "results/model_comparison"):
        Path(folder).mkdir(parents=True, exist_ok=True)
    summary = {"dataset": dataset_summary(rows), "scan_errors": scan_errors,
               "feature_errors": feature_errors, "test_actors": sorted(TEST_ACTORS),
               "training_count": int((~test).sum()), "test_count": int(test.sum()),
               "training_class_counts": dict(Counter(y[~test])), "test_class_counts": dict(Counter(y[test])),
               "feature_names": FEATURE_NAMES, "random_seed": SEED}
    Path("results/metrics/data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results = []
    for name, build in MODELS.items():
        model = build()
        started = time.perf_counter()
        model.fit(X[~test], y[~test])
        elapsed = time.perf_counter() - started
        predictions = model.predict(X[test])
        precision, recall, macro_f1, _ = precision_recall_fscore_support(y[test], predictions, average="macro", zero_division=0)
        metrics = {"name": name, "accuracy": float(accuracy_score(y[test], predictions)),
                   "macro_precision": float(precision), "macro_recall": float(recall),
                   "macro_f1": float(macro_f1), "train_seconds": round(elapsed, 3),
                   "classification_report": classification_report(y[test], predictions, labels=labels, output_dict=True, zero_division=0),
                   "confusion_matrix": confusion_matrix(y[test], predictions, labels=labels).tolist(),
                   "labels": labels, "test_count": int(test.sum())}
        joblib.dump(model, f"models/{name}.joblib")
        Path(f"results/metrics/{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{name.replace('_', ' ').title()} confusion matrix")
        fig.tight_layout()
        fig.savefig(f"results/figures/{name}_confusion.png", dpi=160)
        plt.close(fig)
        results.append({k: metrics[k] for k in ("name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "train_seconds")})
    comparison = pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=False)
    comparison.to_csv("results/model_comparison/comparison.csv", index=False)
    best = str(comparison.iloc[0]["name"])
    Path("models/selected.json").write_text(json.dumps({"name": best, "file": f"{best}.joblib", "selection_metric": "macro_f1", "feature_count": len(FEATURE_NAMES)}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=comparison, x="name", y="macro_f1", color="#38bdf8", ax=ax)
    ax.set(xlabel="Model", ylabel="Macro F1", ylim=(0, 1), title="Held-out actor comparison")
    fig.tight_layout()
    fig.savefig("results/figures/model_comparison.png", dpi=160)
    plt.close(fig)
    return comparison


if __name__ == "__main__":
    print(run_training().to_string(index=False))
