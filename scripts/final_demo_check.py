"""Run actual local demo inference and record pipeline behavior, not accuracy."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio_quality import assess_audio
from src.face_opencv import analyze_photo, aggregate_frames, load_model
from src.fusion import experimental_fusion
from src.predict import predict_audio
from src.video import analyze_video_payload

root = Path(__file__).resolve().parents[1]
assets = root / "demo/assets"
session = load_model()
rows = []
for label in ("happy", "sad", "angry", "neutral"):
    payload = (assets / f"{label}.wav").read_bytes()
    quality = assess_audio(payload)
    voice = predict_audio(payload)
    photo = analyze_photo((assets / f"{label}.jpg").read_bytes(), session)
    clip = analyze_video_payload((assets / f"{label}.mp4").read_bytes())
    face = aggregate_frames(clip["frames"], session)
    fusion = experimental_fusion(voice, face)
    rows.append({"acted_label": label, "voice": voice, "photo_faces": [
        {k: f[k] for k in ("number", "emotion", "probabilities", "warnings")} for f in photo["faces"]],
        "video_face": face, "fusion": fusion, "audio_quality": quality,
        "sampled_frames": clip["sampled"]})
frame = cv2.imread(str(assets / "happy.jpg"))
probes = {"dim": (frame * .15).astype(np.uint8), "blur": cv2.GaussianBlur(frame,(31,31),0),
          "two_faces": np.concatenate([frame,frame],axis=1), "blank": np.zeros_like(frame)}
checks = {}
for name, image in probes.items():
    _, encoded = cv2.imencode(".png", image)
    result = analyze_photo(encoded.tobytes(), session)
    checks[name] = {"faces":len(result["faces"]), "warnings":[f["warnings"] for f in result["faces"]]}
report = {"note":"Pipeline probes only; selected first clips, not an accuracy benchmark.",
          "samples": rows, "image_probes":checks}
destination = root / "results/final_demo_check.json"
destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(destination)
print(checks)
