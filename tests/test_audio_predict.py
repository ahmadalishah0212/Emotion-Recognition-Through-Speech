from io import BytesIO
import json
import wave
import numpy as np
import pytest
from src.feature_extraction import extract_features, FEATURE_NAMES
from src.predict import load_selected, predict_audio


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
