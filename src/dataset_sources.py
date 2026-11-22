"""Source-specific speech labels with explicit dataset and speaker provenance."""
from collections import Counter
import csv
from hashlib import sha256
from pathlib import Path
import soundfile as sf

from .dataset_loader import scan_dataset

TESS_LABELS = {"angry": "angry", "disgust": "disgust", "fear": "fearful",
               "happy": "happy", "neutral": "neutral", "ps": "surprised", "sad": "sad"}
EMODB_LABELS = {"anger": "angry", "disgust": "disgust", "fear": "fearful",
                "happiness": "happy", "neutral": "neutral", "sadness": "sad"}
CREMAD_LABELS = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
                 "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def scan_ravdess(root):
    rows, errors = scan_dataset(root)
    return [{"path": row["path"], "emotion": row["emotion"],
             "speaker": f"ravdess:{row['actor']}", "dataset": "ravdess", "language": "en"}
            for row in rows], errors


def scan_tess(root):
    paths = sorted(Path(root).rglob("*.wav"))
    if not paths:
        return [], []
    rows, errors = [], []
    for path in paths:
        parts = path.stem.split("_")
        if len(parts) < 3 or parts[0] not in ("OAF", "YAF") or parts[-1].lower() not in TESS_LABELS:
            errors.append({"path": str(path), "error": "invalid TESS speaker or label"})
            continue
        rows.append({"path": str(path), "emotion": TESS_LABELS[parts[-1].lower()],
                     "speaker": f"tess:{parts[0]}", "dataset": "tess", "language": "en"})
    return rows, errors


def scan_emodb(root):
    root = Path(root)
    metadata = root / "db.emotion.csv"
    if not metadata.exists():
        return [], []
    rows, errors = [], []
    with metadata.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            source = item["emotion"]
            if source == "boredom":  # No equivalent in the application's eight labels.
                continue
            path = root / item["file"]
            if source not in EMODB_LABELS or not path.exists():
                errors.append({"path": str(path), "error": "missing file or incompatible EmoDB label"})
                continue
            speaker = path.stem[:2]
            rows.append({"path": str(path), "emotion": EMODB_LABELS[source],
                         "speaker": f"emodb:{speaker}", "dataset": "emodb", "language": "de"})
    return rows, errors


def scan_cremad(root):
    paths = sorted(Path(root).rglob("*.wav"))
    rows, errors = [], []
    for path in paths:
        parts = path.stem.split("_")
        if len(parts) != 4 or not parts[0].isdigit() or parts[2] not in CREMAD_LABELS:
            errors.append({"path": str(path), "error": "invalid CREMA-D filename or label"})
            continue
        rows.append({"path": str(path), "emotion": CREMAD_LABELS[parts[2]],
                     "speaker": f"cremad:{parts[0]}", "dataset": "cremad", "language": "en"})
    return rows, errors


def validate_rows(rows):
    """Reject corrupt audio and duplicate bytes; keep cross-corpus provenance."""
    valid, errors, seen = [], [], set()
    for row in rows:
        path = Path(row["path"])
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0 or info.duration < 0.4:
                raise ValueError("empty or too short")
            digest = sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError("duplicate audio bytes")
            seen.add(digest)
            valid.append({**row, "seconds": info.duration, "sample_rate": info.samplerate,
                          "sha256": digest})
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return valid, errors


def summary(rows):
    return {"clips": len(rows), "speakers": len(set(row["speaker"] for row in rows)),
            "labels": dict(Counter(row["emotion"] for row in rows))}
