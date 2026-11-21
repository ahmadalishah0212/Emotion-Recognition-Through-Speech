# Emotion Recognition Through Speech

**Team:** Ahmad Ali Shah, Saboor Khan, Sardar Bilal  
**Status:** Implementation prepared; empirical sections pending dataset download and dependency installation.

## Abstract and problem statement

This project classifies the *acted vocal expression* in a short speech recording into one of eight RAVDESS labels. It compares three CPU-friendly machine learning models using a speaker-independent test set and serves the selected saved model through a local Streamlit app. It cannot determine a person's genuine feelings from voice alone.

## Objectives and dataset

The objectives are reproducible dataset checks, audio feature extraction, fair model comparison, and a working upload/record/predict interface. The input is audio-only speech from [RAVDESS](https://zenodo.org/records/1188976), published by Livingstone and Russo (2018). The official speech ZIP contains 1,440 WAV files from 24 actors according to its record; the **observed local count is pending validation**. The target labels are neutral, calm, happy, sad, angry, fearful, disgust and surprised. Recording conditions are controlled and acted. The dataset uses CC BY-NC-SA 4.0; the audio stays outside Git.

## Preprocessing and input features

`src.dataset_loader` checks filename fields and WAV integrity and removes byte-identical duplicates. Missing files are flagged through the observed count; unreadable or empty files are reported and excluded. There are no tabular null values to impute. Target names are used as categorical classes; scikit-learn encodes them internally. `src.feature_extraction` converts audio to mono 16 kHz, rejects short or silent audio, limits processing to 12 seconds, normalizes amplitude and extracts 46 summary features: 20 MFCC means, 20 MFCC standard deviations, and mean/standard deviation of RMS, zero-crossing rate and spectral centroid. Scaling is contained within the SVM and logistic regression pipelines, fitted only on training data. The random forest does not need scaling.

## EDA

`notebooks/exploratory_analysis.ipynb` and `python -m src.eda` count classes and actors and generate waveform and spectrogram plots from actual WAV files. **Observed distributions and patterns are pending the dataset run.** Graphs will be saved in `results/figures/`.

## Methodology

Actors 05, 10, 15, 20, 23 and 24 form the held-out test group; the other 18 form training. The fixed actor split reduces speaker leakage and is shared by all models. SVM with an RBF kernel, random forest (200 trees), and class-weighted logistic regression are trained on the same 46 features. The fixed seed is 42. Training outputs model files, a data audit, class reports, confusion matrices and a comparison chart. Accuracy, macro precision, macro recall and macro F1 are computed. Macro F1 is the main selection metric because it treats each emotion equally; accuracy can mask weak minority-class performance.

## Results and selected model

**Pending: no RAVDESS audio or ML dependencies were available in the execution environment.** No genuine training, confusion matrix, comparison score or best model can be reported yet. Run the commands in the README, then copy the actual table from `results/model_comparison/comparison.csv` here. Review every confusion matrix and describe specific weak classes. The training script selects the highest macro F1 (accuracy breaks a tie) and writes `models/selected.json`; the app loads this selected model. Explain its observed strengths and errors after the run.

## Application and demonstration

`app.py` accepts a WAV/MP3/FLAC/OGG upload or a browser microphone recording through Streamlit's `st.audio_input`. It plays the input, extracts the same features, loads the saved selected model and displays the predicted expression and model score distribution. Missing model and bad audio produce actionable errors. The microphone requires browser permission and Streamlit support. Demo: show dataset validation, EDA figures, the comparison CSV, then analyze two real speech clips in the app.

## Limitations and conclusion

RAVDESS uses acted North American English speech and a small number of speakers; real-world language, accents, noise, recording devices and spontaneous emotion may differ. File-level probabilities are not calibrated certainty. The fixed single split is reproducible but gives one estimate; a future extension could use group cross-validation. The implementation is ready for empirical completion, but its accuracy and selected model remain unknown until actual training.

## References

- Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLoS ONE 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
- RAVDESS archive and license: https://zenodo.org/records/1188976
- scikit-learn documentation: https://scikit-learn.org/stable/
- librosa documentation: https://librosa.org/doc/latest/
- Streamlit documentation: https://docs.streamlit.io/
