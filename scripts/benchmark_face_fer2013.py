"""Optional labeled FER2013 crop benchmark; requires user-obtained official CSV."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.face import FACE_LABELS, load_face_model

FER2013_LABELS = ("angry", "disgust", "fearful", "happy", "sad", "surprised", "neutral")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="FER2013 fer2013.csv obtained under official terms")
    parser.add_argument("--usage", default="PrivateTest", choices=("PublicTest", "PrivateTest"))
    parser.add_argument("--output", type=Path, default=Path("results/face/fer2013_benchmark.json"))
    arguments = parser.parse_args()
    session = load_face_model()
    input_name = session.get_inputs()[0].name
    truth, predicted = [], []
    with arguments.csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["Usage"] != arguments.usage:
                continue
            label_id = int(row["emotion"])
            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            if label_id not in range(7) or pixels.size != 48 * 48:
                raise ValueError("Invalid FER2013 label or 48x48 pixel row")
            image = cv2.resize(pixels.reshape(48, 48), (64, 64), interpolation=cv2.INTER_LINEAR)
            logits = session.run(None, {input_name: image.astype(np.float32).reshape(1, 1, 64, 64)})[0]
            if np.asarray(logits).shape != (1, 8) or not np.isfinite(logits).all():
                raise ValueError("Face model returned invalid logits")
            truth.append(FER2013_LABELS[label_id])
            predicted.append(FACE_LABELS[int(np.argmax(logits))])
    if not truth:
        raise ValueError(f"No {arguments.usage} rows found")
    result = {"source": str(arguments.csv_path), "split": arguments.usage, "clips": len(truth),
              "scope": "FER2013 cropped classifier only; face detector and video pipeline excluded",
              "accuracy": float(accuracy_score(truth, predicted)),
              "per_class": classification_report(truth, predicted, labels=FER2013_LABELS,
                                                 output_dict=True, zero_division=0),
              "confusion_labels": list(FER2013_LABELS) + ["contempt"],
              "confusion_matrix": confusion_matrix(truth, predicted,
                                                   labels=list(FER2013_LABELS) + ["contempt"]).tolist()}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Evaluated {len(truth)} crops; accuracy {result['accuracy']:.4f}; saved {arguments.output}")


if __name__ == "__main__":
    main()
