from io import BytesIO
import numpy as np
import soundfile as sf
import pytest
from src.prosody import extract_features_v3,PROSODY_NAMES
from src.feature_extraction import extract_features_v2


def test_prosody_preserves_existing_features_and_short_audio_shape():
    sr=16000
    t=np.arange(int(sr*.45))/sr
    audio=(.3*np.sin(2*np.pi*180*t)*(1+.3*np.sin(2*np.pi*4*t))).astype('float32')
    buffer=BytesIO();sf.write(buffer,audio,sr,format='WAV')
    payload=buffer.getvalue();features=extract_features_v3(payload)
    assert len(PROSODY_NAMES)==8 and features.shape==(138,)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:130],extract_features_v2(payload))
    assert features[133]==pytest.approx(.45)


def test_prosody_rejects_silence():
    buffer=BytesIO();sf.write(buffer,np.zeros(16000),16000,format='WAV')
    with pytest.raises(ValueError,match='silent'):extract_features_v3(buffer.getvalue())
