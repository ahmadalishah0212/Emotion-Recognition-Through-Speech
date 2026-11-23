"""Small deterministic prosodic summary shared by training and inference."""
import librosa
import numpy as np
from .feature_extraction import load_audio,extract_features_v2

PROSODY_NAMES=('f0_median','f0_std','f0_p90_p10','duration','rms_median','rms_p90_p10','rms_delta_std','active_frame_fraction')

def prosody_summary(signal,sr):
    f0=librosa.yin(signal,fmin=65,fmax=500,sr=sr,frame_length=1024,hop_length=512)
    rms=librosa.feature.rms(y=signal)[0]
    voiced=f0[rms[:len(f0)]>max(.01,float(rms.max())*.15)]
    return np.array([np.median(voiced) if len(voiced) else 0,np.std(voiced) if len(voiced) else 0,
                     np.percentile(voiced,90)-np.percentile(voiced,10) if len(voiced) else 0,
                     len(signal)/sr,np.median(rms),np.percentile(rms,90)-np.percentile(rms,10),
                     np.std(np.diff(rms)),len(voiced)/len(f0)],dtype=np.float32)

def extract_features_v3(source):
    signal,sr=load_audio(source)
    features=np.concatenate([extract_features_v2(source),prosody_summary(signal,sr)])
    if features.shape!=(138,) or not np.isfinite(features).all():
        raise ValueError('Prosodic features are invalid')
    return features
