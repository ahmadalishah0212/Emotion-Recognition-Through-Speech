"""Check whether the conservative input guard rejects known speech clips."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio_quality import assess_audio

root = Path(__file__).resolve().parents[1] / "data/raw"
paths = sorted(root.glob("Actor_*/*.wav"))
rejected, warned = [], []
for path in paths:
    try:
        report = assess_audio(path)
        if report["warnings"]:
            warned.append((str(path.relative_to(root)), report["warnings"]))
    except ValueError as error:
        rejected.append((str(path.relative_to(root)), str(error)))
print({"scanned": len(paths), "rejected": len(rejected), "warned": len(warned),
       "rejected_examples": rejected[:5], "warning_examples": warned[:5]})
