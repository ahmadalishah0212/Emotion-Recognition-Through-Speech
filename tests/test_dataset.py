from pathlib import Path
import wave
import pytest
from src.dataset_loader import parse_filename, scan_dataset


def test_label_and_actor_mapping():
    row = parse_filename("03-01-05-02-01-01-24.wav")
    assert row["emotion"] == "angry" and row["actor"] == "24"


def test_reject_non_speech():
    with pytest.raises(ValueError):
        parse_filename("03-02-05-02-01-01-24.wav")


def test_scan_and_duplicate(tmp_path):
    for actor in ("01", "02"):
        path = tmp_path / f"03-01-05-02-01-01-{actor}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)
    rows, errors = scan_dataset(tmp_path)
    assert len(rows) == 1 and len(errors) == 1
    assert "duplicate" in errors[0]["error"]
