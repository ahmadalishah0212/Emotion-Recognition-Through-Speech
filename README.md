# VocalLens: Multimodal Emotion Recognition Through Speech and Facial Expressions

**Hear the voice. See the emotion.**

**Team:** Ahmad Ali Shah, Saboor Khan, Sardar Bilal. VocalLens is a local Python and Streamlit course project with voice, face, and experimental combined expression analysis. The voice models were trained and evaluated by this project; the face model is pretrained. See [testing status](TESTING.md).

## Features

- Official RAVDESS speech ZIP download, checksum check and dataset audit
- Shared audio preprocessing and 46-dimensional feature extraction
- Speaker-independent evaluation of SVM, random forest and logistic regression
- Saved models, class reports, confusion matrices and macro F1 comparison
- Streamlit upload and browser microphone input with actual saved-model prediction
- Explicit input-source selection, so a previous microphone recording cannot override a new upload
- Camera still capture and video face analysis using pretrained FER+ and OpenCV face detection
- Video audio extraction and separate voice/face results, with labeled experimental late fusion
- File fingerprint and processing details to verify which input was analyzed

## Windows setup (PowerShell)

Install Python **3.12** from [python.org](https://www.python.org/downloads/) and select **Add Python to PATH**. The saved model was trained with Python 3.12, scikit-learn 1.9.1, NumPy 2.5.5, SciPy 1.18.1 and joblib 1.6.0; `requirements.txt` pins those model-loading dependencies. In PowerShell, from this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_face_model.py
python scripts/download_dataset.py
python scripts/validate_dataset.py
python -m src.eda
python -m src.train
python -m pytest -q
python -m streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell session. No GPU or paid API is required. Browser microphone and camera capture require permission. WAV is the safest audio format. MP3 decoding worked on this laptop; decoder support can vary elsewhere. Video audio extraction uses the FFmpeg binary bundled by `imageio-ffmpeg`.

The face model download is about 35 MB. The script verifies SHA-256 before saving it to `models/face/`; the binary is excluded from Git. Its source is the [MIT-licensed ONNX Model Zoo FER+ model](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md). Face mode requires this file. Video uploads are limited to 50 MB and two minutes; up to 12 frames are analyzed. Camera mode captures one still frame. For both cues from the same recording, upload a video with sound.

From **Command Prompt** in this workspace, use the project Python directly (no activation required):

```bat
cd /d "C:\Users\ahmad\Desktop\Emotion Recognition"
.venv\Scripts\python.exe -m streamlit run app.py
```

Running `python -m streamlit` from `C:\Users\ahmad` uses the system Python instead of this project's environment, which produces `No module named streamlit`.

The official dataset is already extracted in this workspace. For a new clone, if the dataset script cannot connect, download **Audio_Speech_Actors_01-24.zip** from the [official RAVDESS record](https://zenodo.org/records/1188976), place it in `data/raw/`, then rerun `python scripts/download_dataset.py`. The script checks the archive's official MD5 and extracts it. See [data setup](data/README.md). Do not commit the dataset ZIP or extracted audio.

## Output and interpretation

The instructor's required 2–4 model range is met by three trained models. Training created `models/{svm,random_forest,logistic_regression}.joblib` and `models/selected.json`; the app never retrains. Results appear in `results/metrics/`, `results/figures/` and `results/model_comparison/comparison.csv`. On 360 recordings from six held-out actors, the calibrated SVM was selected by macro F1: **0.3975** (accuracy **0.4417**), versus logistic regression **0.3927** and random forest **0.3738** macro F1. The margin over logistic regression is small, and neutral, happy and sad are weak classes. Read the per-class report and confusion matrices before presenting the result.

Use `python -m src.eda` to regenerate graphs and `python -m src.train` to regenerate model results. To launch only the app after training, use `python -m streamlit run app.py`. The app explains missing model files and asks you to train first.

## Demo for the instructor

1. Open the [dataset source](https://zenodo.org/records/1188976) and run `python scripts/validate_dataset.py`.
2. Show `results/figures/emotion_distribution.png`, `actor_distribution.png` and `sample_audio.png` after EDA.
3. Show the actual comparison CSV and three confusion matrices after training; explain macro F1 and the selected model.
4. Start Streamlit. In **Voice only**, upload two different RAVDESS WAVs and press **Analyze voice** for each. The file hash and chart should update. A stock crying MP3 can still be labeled Fearful because the model was trained on spoken sentences, not non-speech crying.
5. In **Face only**, capture a camera still or upload a video. In **Voice + Face**, upload a video with sound to show separate predictions and the experimental combined cue.
6. Explain that outputs describe expression cues, not a person's true feelings; the combined cue has no measured accuracy.

## Repository and submission

`REQUIREMENTS_CHECKLIST.md`, `reports/project_report.md`, `reports/model_explanation.md`, `reports/team_roles.md`, `reports/viva_preparation.md`, and `TESTING.md` contain the report, method, roles and audit. The generated models, metrics and graphs can be committed; the large dataset remains excluded. The project repository is [Emotion-Recognition-Through-Speech](https://github.com/ahmadalishah0212/Emotion-Recognition-Through-Speech). To push later local changes from this checkout, run:

```powershell
git add app.py src scripts tests README.md requirements.txt reports TESTING.md REQUIREMENTS_CHECKLIST.md
git commit -m "Describe your update"
git push origin main
```

Submit the GitHub link, the official dataset link, the report, and a live app demo. The baseline, dataset/EDA and model milestones were pushed; check `git status` and `git log` for later milestones.

## Limitations

RAVDESS is acted studio speech. The selected SVM scored 44.17% accuracy and 39.75% macro F1 on one held-out actor split, so stock crying or laughter can share an incorrect label despite different feature vectors and charts. FER+ scores visible expressions but was not trained or independently evaluated by this group. Face detection can miss nonfrontal faces; with multiple faces only the largest per frame is scored, which may switch people across frames. Combined scores use shared classes only and are unvalidated. Model bars are estimates, not diagnostic certainty. See [report](reports/project_report.md).
