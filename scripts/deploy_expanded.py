"""Deploy the training-CV winner after checking the saved experiment artifact."""
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from src.feature_extraction import FEATURE_NAMES, FEATURE_NAMES_V2

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "results/expanded/evaluation.json").read_text(encoding="utf-8"))
    winner = max(report["training_cv"], key=lambda row: row["mean_macro_f1"])["candidate"]
    if winner != report["selected_candidate"]:
        raise ValueError("Stored candidate does not match training-only selection")
    version = "v2" if winner.endswith("v2") else "v1"
    feature_count = len(FEATURE_NAMES_V2) if version == "v2" else len(FEATURE_NAMES)
    source = ROOT / "data/cache/expanded_candidate.joblib"
    model = joblib.load(source)
    if model.n_features_in_ != feature_count:
        raise ValueError("Candidate feature count is inconsistent")
    destination = ROOT / "models" / f"{winner}.joblib"
    shutil.copyfile(source, destination)
    selection = {"name": winner, "file": destination.name, "selection_metric": "training_actor_cv_macro_f1",
                 "feature_version": version, "feature_count": feature_count,
                 "evaluation": "results/expanded/evaluation.json"}
    (ROOT / "models/selected.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Deployed {winner}: {feature_count} features; original models retained")


if __name__ == "__main__":
    main()
