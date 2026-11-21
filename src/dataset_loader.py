"""RAVDESS speech-only filename parsing and dataset checks."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import wave

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
DATA_DIR = Path("data/raw")


def parse_filename(path):
    parts = Path(path).stem.split("-")
    if len(parts) != 7 or any(len(p) != 2 or not p.isdigit() for p in parts):
        raise ValueError(f"Invalid RAVDESS filename: {Path(path).name}")
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if (modality, channel) != ("03", "01") or emotion not in EMOTIONS:
        raise ValueError(f"Expected RAVDESS audio-only speech: {Path(path).name}")
    if intensity not in ("01", "02") or statement not in ("01", "02") or repetition not in ("01", "02") or not 1 <= int(actor) <= 24:
        raise ValueError(f"Invalid RAVDESS metadata: {Path(path).name}")
    return {"path": str(path), "emotion": EMOTIONS[emotion], "actor": actor,
            "statement": statement, "repetition": repetition, "intensity": intensity}


def scan_dataset(root=DATA_DIR, check_audio=True):
    root = Path(root)
    files = sorted(root.rglob("*.wav")) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No WAV files in {root}. Run python scripts/download_dataset.py or see data/README.md")
    rows, errors, seen_hashes = [], [], set()
    for path in files:
        try:
            row = parse_filename(path)
            if check_audio:
                with wave.open(str(path), "rb") as wav:
                    if wav.getnframes() == 0 or wav.getframerate() <= 0:
                        raise ValueError("empty audio")
                digest = sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    raise ValueError("duplicate audio bytes")
                seen_hashes.add(digest)
            rows.append(row)
        except (ValueError, wave.Error, EOFError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if not rows:
        raise ValueError(f"No valid RAVDESS speech WAV files in {root}")
    return rows, errors


def dataset_summary(rows):
    return {"files": len(rows), "actors": len({r["actor"] for r in rows}),
            "emotions": dict(sorted(Counter(r["emotion"] for r in rows).items()))}
