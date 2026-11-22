from pathlib import Path
import wave
import pytest
from src.dataset_loader import parse_filename, scan_dataset
from src.dataset_sources import scan_tess, scan_emodb, validate_rows


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


def test_external_label_mapping_and_speaker_provenance(tmp_path):
    tess = tmp_path / "tess"
    tess.mkdir()
    sample = tess / "YAF_word_ps.wav"
    with wave.open(str(sample), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * 16000)
    rows, errors = scan_tess(tess)
    assert not errors and rows[0]["emotion"] == "surprised"
    assert rows[0]["speaker"] == "tess:YAF"
    valid, rejected = validate_rows(rows)
    assert len(valid) == 1 and not rejected


def test_emodb_boredom_is_excluded(tmp_path):
    (tmp_path / "wav").mkdir()
    (tmp_path / "wav/03a01Bc.wav").write_bytes(b"unused")
    (tmp_path / "db.emotion.csv").write_text("file,emotion\nwav/03a01Bc.wav,boredom\n")
    rows, errors = scan_emodb(tmp_path)
    assert rows == [] and errors == []
