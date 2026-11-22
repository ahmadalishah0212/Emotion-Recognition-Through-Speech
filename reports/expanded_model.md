# Expanded speech model experiment

## Reproducible protocol

The original RAVDESS six-actor test set (Actors 05, 10, 15, 20, 23, 24; 360 clips) stayed untouched. Of the 1,439 usable RAVDESS speech clips, 1,079 from 18 other actors were eligible for training. TESS contributed 490 randomly selected OAF clips (seed 42, 70 per seven mapped classes) to expanded candidates. All 1,400 YAF TESS clips and all 454 compatible German EmoDB clips from ten speakers remained external tests. The other 910 OAF clips were unused. Corpus-prefixed speaker IDs prevent accidental overlap. One byte-identical RAVDESS duplicate was excluded; all 2,800 TESS and 454 compatible EmoDB files passed decoding and hash validation. EmoDB boredom (81 files) has no eight-class equivalent and was excluded.

The candidate comparison used **three-fold GroupKFold over the 18 RAVDESS training actors**. Each expanded model also received the fixed OAF subset in its training fold; validation used only the held-out RAVDESS fold. TESS YAF, EmoDB and the original RAVDESS test actors were not consulted for hyperparameter choice. The winner was selected by mean validation macro F1, fitted once to the allowed training clips, then tested. The baseline test column comes from the original saved calibrated SVM. Because RAVDESS has a unique `calm` class and the external corpora do not, external macro metrics use only each source's true label set while preserving wrong-class predictions (including `calm`) as errors. This avoids treating an absent class as a zero-support class in macro averaging.

## Candidates and selection

| Candidate | Features | Training clips | Training-only CV macro F1 |
|---|---:|---:|---:|
| Calibrated SVM, baseline protocol | 46 | RAVDESS 1,079 | 0.3839 |
| Calibrated SVM, expanded | 46 | RAVDESS + TESS 1,569 | 0.3707 |
| Balanced logistic regression, expanded | 46 | RAVDESS + TESS 1,569 | 0.3676 |
| Balanced random forest, expanded | 46 | RAVDESS + TESS 1,569 | 0.3331 |
| Calibrated SVM, expanded | 130 | RAVDESS + TESS 1,569 | 0.4053 |
| **Balanced logistic regression, expanded** | **130** | **RAVDESS + TESS 1,569** | **0.4093** |

The 130-feature extractor retains 20 MFCC means and standard deviations, six original RMS/zero-crossing/centroid summaries, adds first and second MFCC delta means and standard deviations, and adds spectral bandwidth and rolloff means and standard deviations. Training and inference use the same 16 kHz mono, peak-normalized, maximum-12-second preprocessing. A standard scaler is fitted inside the logistic pipeline. The original models remain saved in `models/`; `models/selected.json` points to `logistic_expanded_v2.joblib`.

## Untouched tests

| Test | Baseline accuracy | New accuracy | Baseline macro precision / recall / F1 | New macro precision / recall / F1 |
|---|---:|---:|---:|---:|
| RAVDESS six actors, n=360 | 0.4417 | **0.4583** | 0.4391 / 0.4219 / 0.3975 | 0.4535 / 0.4661 / **0.4512** |
| TESS YAF, n=1,400 | 0.3386 | **0.3779** | 0.3558 / 0.3386 / 0.2408 | 0.4547 / 0.3779 / **0.3337** |
| EmoDB all compatible, n=454 | 0.1101 | **0.3194** | 0.2696 / 0.1228 / 0.1072 | 0.4612 / 0.2703 / **0.3004** |

The exact precision, recall, per-class reports, prediction counts, eight-class confusion arrays, speaker lists and corpus class distributions are in [evaluation.json](../results/expanded/evaluation.json). Three side-by-side confusion figures are in `results/expanded/`. The improvement is observed on these specific acted test sets; no uncertainty interval or natural-speech test supports a general population claim. TESS is especially narrow (two female speakers; same vocabulary across speakers). EmoDB is a German acted corpus and still yields weak absolute performance. The selected model is practical on CPU and small enough to commit. The FER+ face model and fusion are separate and unbenchmarked on labeled external videos.

## Limits and safeguards

The old SVM's fixed-looking Fearful distributions on pure tones and stock non-speech files came from extreme RBF feature distance and calibration saturation; the [diagnostic log](../TESTING.md) shows exact input hashes and scores. The new logistic model avoids that particular saturation mechanism but can still be wrong or overconfident. The app rejects invalid, silent, too-short and highly tonal files and warns on clipping and short recordings. It has no validated speech detector, background-noise robustness measurement, probability calibration study, or out-of-distribution threshold. Speech with music, crying, accents and phone compression remain unvalidated.

## Reproduction

Run the download, validation, training and deployment commands in [README](../README.md). The candidate remains in ignored `data/cache/` until `scripts/deploy_expanded.py` verifies that its 130-feature count matches the recorded training-CV winner. The reported numbers were read from the saved evaluation JSON, not typed into training code or used for selection.
