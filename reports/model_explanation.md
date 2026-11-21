# Model and feature explanation

Audio is decoded to mono, resampled to 16 kHz, capped at 12 seconds and peak-normalized. Files shorter than 0.4 seconds, silent or invalid are rejected. We summarize 20 MFCC coefficients by mean and standard deviation (40 values). MFCCs describe the spectral shape of speech, a rough representation of vocal timbre. Mean and standard deviation of RMS energy describe loudness and variation. Zero-crossing rate approximates rapid sign changes, which can reflect noisy or unvoiced speech. Spectral centroid describes where high-frequency energy is concentrated. The six summary values plus MFCCs give 46 features.

SVM finds boundaries between emotion classes in scaled feature space. Random forest averages decisions of many trees and can model nonlinear relationships without feature scaling. Logistic regression is a simple linear baseline. StandardScaler is saved inside the SVM and logistic regression pipelines and fit on training actors only. It is applied automatically to test audio and app input. The forest uses raw features.

Actors 05, 10, 15, 20, 23 and 24 are held out completely. A recording from any of those actors cannot appear in training. All three models use the same split. Macro F1 averages class F1 scores equally and determines the selected model, with accuracy as a tie breaker. The app reads `models/selected.json` and loads that exact saved pipeline. Scores from `predict_proba` are estimates, not guarantees or calibrated measures of a person's emotional state.

The actual winner, class errors and weaknesses must be filled in from `results/model_comparison/comparison.csv` and confusion matrices after training. No model can be honestly named best before that run.
