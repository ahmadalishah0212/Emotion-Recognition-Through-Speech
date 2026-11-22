"""One audio-to-feature path shared by training and inference."""
from io import BytesIO
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
MIN_SECONDS = 0.4
MAX_SECONDS = 12.0
N_MFCC = 20
FEATURE_NAMES = ([f"mfcc_{i+1}_{stat}" for stat in ("mean", "std") for i in range(N_MFCC)]
                 + ["rms_mean", "rms_std", "zcr_mean", "zcr_std", "centroid_mean", "centroid_std"])
FEATURE_NAMES_V2 = (FEATURE_NAMES[:40]
                    + [f"mfcc_delta{order}_{i+1}_{stat}" for order in (1, 2)
                       for stat in ("mean", "std") for i in range(N_MFCC)]
                    + FEATURE_NAMES[40:]
                    + ["bandwidth_mean", "bandwidth_std", "rolloff_mean", "rolloff_std"])


def load_audio(source):
    try:
        if isinstance(source, bytes):
            if not source:
                raise ValueError("Audio file is empty")
            # soundfile handles WAV/FLAC/OGG; librosa/audioread can handle MP3 if installed.
            try:
                signal, sr = sf.read(BytesIO(source), dtype="float32", always_2d=False)
            except Exception:
                if not (source.startswith(b"ID3") or (len(source) >= 2 and source[0] == 0xFF and source[1] & 0xE0 == 0xE0)):
                    raise ValueError("Audio could not be decoded. Try a valid WAV or MP3 file.")
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "upload.mp3"
                    path.write_bytes(source)
                    signal, sr = librosa.load(path, sr=None, mono=True)
        else:
            signal, sr = librosa.load(source, sr=None, mono=True)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Audio could not be decoded. Try a valid WAV file.") from exc
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.size == 0 or sr <= 0 or not np.isfinite(signal).all():
        raise ValueError("Audio is empty or invalid")
    if len(signal) / sr < MIN_SECONDS:
        raise ValueError(f"Audio must be at least {MIN_SECONDS} seconds long")
    if len(signal) / sr > MAX_SECONDS:
        signal = signal[:int(MAX_SECONDS * sr)]
    if sr != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SAMPLE_RATE)
    # Fixed amplitude normalization reduces microphone-volume variation.
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-6:
        raise ValueError("Audio is silent")
    signal = signal / peak
    return signal, SAMPLE_RATE


def extract_features(source):
    signal, sr = load_audio(source)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=signal)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                               [rms.mean(), rms.std(), zcr.mean(), zcr.std(),
                                centroid.mean(), centroid.std()]]).astype(np.float32)
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Feature extraction produced invalid values")
    return features


def extract_features_v2(source):
    """Experimental richer summary; the saved baseline still uses v1."""
    signal, sr = load_audio(source)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    components = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    for order in (1, 2):
        delta = librosa.feature.delta(mfcc, order=order)
        components.extend((delta.mean(axis=1), delta.std(axis=1)))
    rms = librosa.feature.rms(y=signal)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    components.append(np.array([rms.mean(), rms.std(), zcr.mean(), zcr.std(),
                                centroid.mean(), centroid.std(), bandwidth.mean(), bandwidth.std(),
                                rolloff.mean(), rolloff.std()]))
    features = np.concatenate(components).astype(np.float32)
    if features.shape != (len(FEATURE_NAMES_V2),) or not np.isfinite(features).all():
        raise ValueError("Experimental feature extraction produced invalid values")
    return features
