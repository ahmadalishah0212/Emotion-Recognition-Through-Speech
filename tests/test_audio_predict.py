from io import BytesIO
import json
from hashlib import sha256
from pathlib import Path
import wave
import numpy as np
import pytest
from src.feature_extraction import extract_features, FEATURE_NAMES, FEATURE_NAMES_V2
from src.predict import load_selected, predict_audio
from src.audio_debug import trace_audio
from src.audio_quality import assess_audio


def sample_wav(seconds=1, frequency=440):
    t = np.arange(16000 * seconds) / 16000
    samples = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def test_feature_dimensions_and_finite_values():
    features = extract_features(sample_wav())
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_different_audio_changes_features_and_bytes_match_training_path(tmp_path):
    first_audio = sample_wav(frequency=220)
    second_audio = sample_wav(frequency=660)
    first = extract_features(first_audio)
    second = extract_features(second_audio)
    assert np.linalg.norm(first - second) > 1
    path = tmp_path / "sample.wav"
    path.write_bytes(first_audio)
    np.testing.assert_allclose(first, extract_features(path), rtol=1e-5, atol=1e-4)


def test_invalid_audio():
    with pytest.raises(ValueError):
        extract_features(b"")
    with pytest.raises(ValueError, match="could not be decoded"):
        extract_features(b"not audio")


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_selected(tmp_path)


def test_audio_prediction_through_saved_pipeline(tmp_path):
    from sklearn.dummy import DummyClassifier
    import joblib
    model = DummyClassifier(strategy="constant", constant="happy")
    model.fit(np.ones((2, len(FEATURE_NAMES))), ["happy", "sad"])
    joblib.dump(model, tmp_path / "dummy.joblib")
    (tmp_path / "selected.json").write_text(json.dumps({"name": "dummy", "file": "dummy.joblib"}))
    result = predict_audio(sample_wav(), tmp_path)
    assert result["emotion"] == "happy" and result["model"] == "dummy"
    assert result["emotion"] == max(result["probabilities"], key=result["probabilities"].get)


def test_svm_label_matches_its_score_distribution():
    """Regression check for a label/score mismatch seen in the live app."""
    from src.train import MODELS
    rng = np.random.default_rng(42)
    x = rng.normal(size=(36, len(FEATURE_NAMES)))
    y = np.array(["calm", "angry"] * 18)
    model = MODELS["svm"]()
    model.fit(x, y)
    probes = rng.normal(size=(5, len(FEATURE_NAMES)))
    labels = model.predict(probes)
    score_labels = model.classes_[model.predict_proba(probes).argmax(axis=1)]
    assert np.array_equal(labels, score_labels)


def test_current_bytes_drive_card_chart_and_debug_trace():
    first_audio = sample_wav(frequency=220)
    second_audio = sample_wav(frequency=660)
    first = predict_audio(first_audio)
    second = predict_audio(second_audio)
    repeat = predict_audio(first_audio)
    first_trace = trace_audio(first_audio, first)
    second_trace = trace_audio(second_audio, second)
    assert first_trace["audio_sha256"] == sha256(first_audio).hexdigest()
    assert second_trace["audio_sha256"] == sha256(second_audio).hexdigest()
    model, selection = load_selected()
    assert first_trace["feature_shape"] == [model.n_features_in_]
    assert first_trace["feature_version"] == selection.get("feature_version", "v1")
    assert model.n_features_in_ in (len(FEATURE_NAMES), len(FEATURE_NAMES_V2))
    assert first_trace["model_file"].endswith(selection["file"])
    assert first["probabilities"] == first_trace["probabilities"]
    assert first["emotion"] == max(first["probabilities"], key=first["probabilities"].get)
    assert first["probabilities"] == repeat["probabilities"]
    # The low-level predictor remains genuine inference, even for unsuitable tones.
    assert not np.array_equal(extract_features(first_audio), extract_features(second_audio))


def test_real_heldout_speech_changes_saved_model_scores():
    root = Path(__file__).resolve().parents[1] / "data/raw"
    angry = root / "Actor_05/03-01-05-01-01-01-05.wav"
    happy = root / "Actor_10/03-01-03-01-01-01-10.wav"
    if not angry.exists() or not happy.exists():
        pytest.skip("Download RAVDESS for the real speech integration check")
    first = predict_audio(angry.read_bytes())
    second = predict_audio(happy.read_bytes())
    assert first["emotion"] == max(first["probabilities"], key=first["probabilities"].get)
    assert second["emotion"] == max(second["probabilities"], key=second["probabilities"].get)
    assert first["probabilities"] != second["probabilities"]


def test_pure_tone_is_rejected_before_app_prediction():
    with pytest.raises(ValueError, match="pure tone"):
        assess_audio(sample_wav())
