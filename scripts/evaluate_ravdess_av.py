"""Evaluate actual face frames and fusion on an official RAVDESS actor ZIP."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.fusion import experimental_fusion
from src.predict import predict_audio
from src.train import TEST_ACTORS
from src.video import analyze_video_payload

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry",
            "06": "fearful", "07": "disgust", "08": "surprised"}
SHARED = ("neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised")


def measure(rows, key):
    valid = [row for row in rows if row["truth"] in SHARED and row[key] is not None]
    truth = [row["truth"] for row in valid]
    guesses = [row[key] for row in valid]
    return {"clips": len(valid), "accuracy": float(accuracy_score(truth, guesses)),
            "macro_f1": classification_report(truth, guesses, labels=SHARED,
                                               output_dict=True, zero_division=0)["macro avg"]["f1-score"],
            "per_class": classification_report(truth, guesses, labels=SHARED,
                                               output_dict=True, zero_division=0),
            "confusion_labels": list(SHARED) + ["contempt", "calm"],
            "confusion_matrix": confusion_matrix(truth, guesses,
                                                 labels=list(SHARED) + ["contempt", "calm"]).tolist(),
            "prediction_counts": dict(Counter(guesses))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/face/ravdess_av_evaluation.json"))
    parser.add_argument("--limit", type=int, default=0, help="Optional diagnostic limit; omit for full actor")
    parser.add_argument("--model", choices=("ferplus", "opencv_zoo"), default="ferplus")
    args = parser.parse_args()
    if args.model == "ferplus":
        from src.face import aggregate_frames, load_face_model
        session = load_face_model()
    else:
        from src.face_opencv import aggregate_frames, load_model
        session = load_model()
    rows = []
    with zipfile.ZipFile(args.archive) as package:
        names = []
        for name in package.namelist():
            stem = Path(name).stem
            parts = stem.split("-")
            if len(parts) == 7 and parts[0:2] == ["01", "01"] and parts[2] in EMOTIONS and name.lower().endswith(".mp4"):
                names.append(name)
        names.sort()
        if args.limit:
            names = names[:args.limit]
        if not names:
            raise ValueError("Archive contains no full audiovisual speech MP4s")
        actors = {Path(name).stem.split("-")[-1] for name in names}
        if len(actors) != 1:
            raise ValueError("Use a single-actor official RAVDESS video archive")
        actor = next(iter(actors))
        for index, name in enumerate(names, 1):
            video = analyze_video_payload(package.read(name), ".mp4")
            full = aggregate_frames(video["frames"], session)
            central = aggregate_frames(video["frames"][3:9], session)
            voice = predict_audio(video["audio"]) if actor in TEST_ACTORS and video["audio"] else None
            fusion = experimental_fusion(voice, central)
            truth = EMOTIONS[Path(name).stem.split("-")[2]]
            rows.append({"file": Path(name).name, "truth": truth, "face_full": full["emotion"] if full else None,
                         "face_central": central["emotion"] if central else None,
                         "face_full_scores": full["probabilities"] if full else None,
                         "face_central_scores": central["probabilities"] if central else None,
                         "voice": voice["emotion"] if voice else None,
                         "fusion_central": fusion["emotion"] if fusion else None,
                         "frames_sampled": video["sampled"], "full_frames_with_face": full["frames_with_face"] if full else 0,
                         "central_frames_with_face": central["frames_with_face"] if central else 0,
                         "audio_bytes": len(video["audio"]) if video["audio"] else 0})
            if index % 10 == 0:
                print(f"Analyzed {index}/{len(names)} videos", flush=True)
    result = {"source_archive": args.archive.name, "actor": actor, "face_model": args.model,
              "independent_voice_test": actor in TEST_ACTORS,
              "protocol_note": "Face results use intended acted clip labels. Central-frame comparison is exploratory if chosen on this archive. Voice/fusion are independent only for original held-out RAVDESS actors.",
              "clips": len(rows), "face_full": measure(rows, "face_full"),
              "face_central": measure(rows, "face_central"),
              "voice": measure(rows, "voice") if actor in TEST_ACTORS else None,
              "fusion_central": measure(rows, "fusion_central") if actor in TEST_ACTORS else None,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; face macro F1 full={result['face_full']['macro_f1']:.3f}, "
          f"central={result['face_central']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
