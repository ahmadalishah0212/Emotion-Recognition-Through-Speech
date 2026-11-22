"""Prepare local, attributed RAVDESS demo inputs; do not select by model score."""
from pathlib import Path
import sys
import zipfile
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video import analyze_video_payload

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "demo/assets"
out.mkdir(parents=True, exist_ok=True)
archive = ROOT / "data/cache/Video_Speech_Actor_05.zip"
if not archive.exists():
    raise SystemExit("Run python scripts/download_ravdess_av.py 05 first")
with zipfile.ZipFile(archive) as package:
    for code, label in (("03", "happy"), ("04", "sad"), ("05", "angry"), ("01", "neutral")):
        names = sorted(n for n in package.namelist() if Path(n).name.startswith(f"01-01-{code}-") and n.endswith(".mp4"))
        payload = package.read(names[0])
        (out / f"{label}.mp4").write_bytes(payload)
        clip = analyze_video_payload(payload)
        if clip["audio"]:
            (out / f"{label}.wav").write_bytes(clip["audio"])
        cv2.imwrite(str(out / f"{label}.jpg"), clip["frames"][len(clip["frames"]) // 2])
        print(f"{label}: {names[0]}")
print("Local assets ready. Source RAVDESS, Livingstone & Russo, CC BY-NC-SA 4.0.")
