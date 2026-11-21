"""Create descriptive graphs from the actual downloaded recordings."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display
import seaborn as sns
from .dataset_loader import scan_dataset, dataset_summary
from .feature_extraction import load_audio


def run_eda(data_dir="data/raw"):
    rows, errors = scan_dataset(data_dir)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(data=frame, x="emotion", order=sorted(frame.emotion.unique()), color="#38bdf8", ax=ax)
    ax.set(title="Actual RAVDESS emotion distribution", xlabel="Emotion", ylabel="Recordings")
    fig.tight_layout(); fig.savefig("results/figures/emotion_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=frame, x="actor", color="#34d399", ax=ax)
    ax.set(title="Recordings per actor", xlabel="Actor", ylabel="Recordings")
    fig.tight_layout(); fig.savefig("results/figures/actor_distribution.png", dpi=160); plt.close(fig)
    signal, sr = load_audio(rows[0]["path"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    librosa.display.waveshow(signal, sr=sr, ax=axes[0]); axes[0].set_title(f"Waveform: {Path(rows[0]['path']).name}")
    spectrum = librosa.amplitude_to_db(abs(librosa.stft(signal)), ref=np.max)
    librosa.display.specshow(spectrum, sr=sr, x_axis="time", y_axis="hz", ax=axes[1]); axes[1].set_title("Spectrogram")
    fig.tight_layout(); fig.savefig("results/figures/sample_audio.png", dpi=160); plt.close(fig)
    return {"summary": dataset_summary(rows), "invalid_files": len(errors), "sample": rows[0]["path"]}


if __name__ == "__main__":
    print(run_eda())
