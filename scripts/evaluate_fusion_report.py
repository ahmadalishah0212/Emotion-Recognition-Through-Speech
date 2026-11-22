"""Add full-frame voice/face fusion metrics to a RAVDESS AV evaluation JSON."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_ravdess_av import measure
from src.fusion import experimental_fusion
from src.predict import predict_audio
from src.video import analyze_video_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not report["independent_voice_test"] or report["source_archive"] != args.archive.name:
        raise ValueError("A matching held-out actor report and archive are required")
    rows = {row["file"]: row for row in report["rows"]}
    with zipfile.ZipFile(args.archive) as package:
        for name in package.namelist():
            row = rows.get(Path(name).name)
            if row is None:
                continue
            audio = analyze_video_payload(package.read(name), ".mp4")["audio"]
            voice = predict_audio(audio) if audio else None
            row["voice_scores"] = voice["probabilities"] if voice else None
            face = ({"probabilities": row["face_full_scores"]}
                    if row["face_full_scores"] else None)
            fused = experimental_fusion(voice, face)
            row["fusion_full"] = fused["emotion"] if fused else None
            if voice and row["voice"] != voice["emotion"]:
                raise ValueError(f"Voice prediction changed for {name}")
    report["fusion_full"] = measure(report["rows"], "fusion_full")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Full-frame fusion macro F1: {report['fusion_full']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
