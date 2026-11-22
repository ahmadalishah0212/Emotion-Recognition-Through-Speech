"""Behavioral checks for missing cues and transparent class alignment."""
import numpy as np
import pytest
from pathlib import Path
from src.face import aggregate_frames, load_face_model
from src.fusion import SHARED, experimental_fusion
from src.video import analyze_video_payload


def test_no_face_does_not_produce_emotion():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    assert aggregate_frames([frame], session=None) is None


def test_fusion_requires_both_cues():
    voice = {"probabilities": {label: 1 / 8 for label in (*SHARED, "calm")}}
    assert experimental_fusion(voice, None) is None


def test_fusion_excludes_unmatched_classes_and_normalizes():
    voice = {"probabilities": {label: (0.6 if label == "happy" else 0.01) for label in SHARED} | {"calm": 0.34}}
    face = {"probabilities": {label: (0.7 if label == "sad" else 0.01) for label in SHARED} | {"contempt": 0.24}}
    result = experimental_fusion(voice, face)
    assert set(result["probabilities"]) == set(SHARED)
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["excluded_voice_mass"] == pytest.approx(0.34)
    assert result["excluded_face_mass"] == pytest.approx(0.24)


def test_corrupt_video_rejected():
    with pytest.raises(ValueError, match="decode"):
        analyze_video_payload(b"not a video", ".mp4")


def test_real_face_model_has_expected_outputs_if_downloaded():
    from src.face import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("Download FER+ to run the real model check")
    session = load_face_model()
    assert session.get_inputs()[0].shape == [1, 1, 64, 64]
    assert session.get_outputs()[0].shape == [1, 8]


def test_local_video_fixtures_extract_real_cues_if_available():
    cache = Path(__file__).resolve().parents[1] / "data/cache"
    with_audio = cache / "with_audio.mp4"
    blank = cache / "blank.mp4"
    silent = cache / "silent.mp4"
    if not all(path.exists() for path in (with_audio, blank, silent)):
        pytest.skip("Optional uncommitted video fixtures absent")
    clip = analyze_video_payload(with_audio.read_bytes(), ".mp4")
    assert clip["sampled"] == 12 and clip["audio"]
    assert max(clip["frames"][0].shape[:2]) <= 960
    face = aggregate_frames(clip["frames"], load_face_model())
    assert face and face["frames_with_face"] > 0
    empty_face = analyze_video_payload(blank.read_bytes(), ".mp4")
    assert aggregate_frames(empty_face["frames"], load_face_model()) is None
    empty_audio = analyze_video_payload(silent.read_bytes(), ".mp4")
    assert empty_audio["audio"] is None
