from io import BytesIO
import json
import wave
import numpy as np
import pytest
from src.feature_extraction import extract_features, FEATURE_NAMES
from src.predict import load_selected, predict_audio


def sample_wav(seconds=1):
    t = np.arange(16000 * seconds) / 16000
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def test_feature_dimensions_and_finite_values():
    features = extract_features(sample_wav())
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_invalid_audio():
    with pytest.raises(ValueError):
        extract_features(b"")


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
