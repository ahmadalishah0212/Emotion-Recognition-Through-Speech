"""Conservative input checks; not a speech detector or OOD accuracy claim."""
import numpy as np

from .feature_extraction import load_audio


def assess_audio(source):
    signal, sr = load_audio(source)
    duration = len(signal) / sr
    if duration < 0.65:
        return {"duration": duration, "sample_rate": sr,
                "warnings": ["Very short recordings may not contain enough speech to assess expression."]}
    # A nearly single-frequency signal with the same pitch throughout is not speech.
    slices = [signal[start:start + sr] for start in range(0, len(signal) - sr + 1, max(1, sr // 2))]
    slices = slices[:12]
    dominance = []
    for chunk in slices:
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) ** 2
        if spectrum.sum() > 0:
            dominance.append(float(spectrum.max() / spectrum.sum()))
    if dominance and float(np.median(dominance)) > 0.60:
        raise ValueError("This recording resembles a pure tone, not speech. Upload a spoken clip.")
    peak_fraction = float(np.mean(np.abs(signal) >= 0.999))
    warnings = []
    if peak_fraction > 0.08:
        warnings.append("The recording appears heavily clipped; results may be unreliable.")
    if duration < 1.0:
        warnings.append("Short recordings may not contain enough speech to assess expression.")
    return {"duration": duration, "sample_rate": sr, "warnings": warnings}
