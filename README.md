# Emotion Recognition Through Speech

**Team:** Ahmad Ali Shah, Saboor Khan, Sardar Bilal. A Python and Streamlit course project that compares three speech emotion classifiers on RAVDESS audio. The project is implemented, but genuine training results require the official dataset and dependencies; see [testing status](TESTING.md).

## Features

- Official RAVDESS speech ZIP download, checksum check and dataset audit
- Shared audio preprocessing and 46-dimensional feature extraction
- Speaker-independent evaluation of SVM, random forest and logistic regression
- Saved models, class reports, confusion matrices and macro F1 comparison
- Streamlit upload and browser microphone input with actual saved-model prediction

## Windows setup (PowerShell)

Install Python **3.11 or 3.12** from [python.org](https://www.python.org/downloads/) and select **Add Python to PATH**. In PowerShell, from this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/validate_dataset.py
python -m src.eda
python -m src.train
python -m pytest -q
python -m streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell session. On a Python 3.11 installation, change `py -3.12` to `py -3.11`. No GPU or paid API is required. `st.audio_input` needs microphone access in the browser. WAV is the safest upload format; MP3 support depends on the audio decoder available to librosa. No separate FFmpeg installation is required for WAV.

If the dataset script cannot connect, use a browser to download **Audio_Speech_Actors_01-24.zip** from the [official RAVDESS record](https://zenodo.org/records/1188976), place it in `data/raw/`, then rerun `python scripts/download_dataset.py`. The script checks the archive's official MD5 and extracts it. See [data setup](data/README.md). Do not commit the dataset ZIP or extracted audio.

## Output and interpretation

The instructor's required 2–4 model range is met by three models. Training creates `models/{svm,random_forest,logistic_regression}.joblib` and `models/selected.json`; the app never retrains. Results appear in `results/metrics/`, `results/figures/` and `results/model_comparison/comparison.csv`. The winner is selected by held-out actor macro F1. Read the per-class report and confusion matrices before presenting any accuracy claim. The selected model and observed scores are **pending a real run** in the current workspace.

Use `python -m src.eda` to regenerate graphs and `python -m src.train` to regenerate model results. To launch only the app after training, use `python -m streamlit run app.py`. The app explains missing model files and asks you to train first.

## Demo for the instructor

1. Open the [dataset source](https://zenodo.org/records/1188976) and run `python scripts/validate_dataset.py`.
2. Show `results/figures/emotion_distribution.png`, `actor_distribution.png` and `sample_audio.png` after EDA.
3. Show the actual comparison CSV and three confusion matrices after training; explain macro F1 and the selected model.
4. Start Streamlit. Upload one of the RAVDESS WAV files, press **Analyze speech**, then use the microphone for a separate live recording.
5. Explain that the predictions describe acted expressions in training data, not a person's true feelings.

## Repository and submission

`REQUIREMENTS_CHECKLIST.md`, `reports/project_report.md`, `reports/model_explanation.md`, `reports/team_roles.md`, `reports/viva_preparation.md`, and `TESTING.md` contain the report, method, roles and audit. The generated models, metrics and graphs can be committed after the real run; the large dataset remains excluded. Once genuine results have been generated and the report updated, create a GitHub repository and run:

```powershell
git init
git add .
git commit -m "Build speech emotion recognition project"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with the repository you created. Submit that GitHub link, the official dataset link, the report, and a live app demo. Nothing has been pushed from this workspace.

## Limitations

RAVDESS is acted studio speech. Speaker separation reduces one source of leakage, but performance on spontaneous speech, other accents and noisy microphones may differ. Model score bars are estimates, not diagnostic certainty. The planned fixed test split is one evaluation, not a broad population study. See [report](reports/project_report.md).
