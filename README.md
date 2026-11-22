# VocalLens

**Hear the voice. See the emotion.** Local Streamlit demonstration of speech, face and experimental audiovisual expression cues. Team: Ahmad Ali Shah, Saboor Khan, Sardar Bilal.

VocalLens analyzes the current upload or browser recording. It shows real saved-model scores, input details and optional developer diagnostics. Its charcoal interface uses the original `#e46b72` red identity, a page for each mode, and a responsive expression distribution. The speech model was trained in this project; the facial FER+ model is pretrained. Neither can determine a person's actual feelings.

## Windows setup

Use Python **3.12**. From PowerShell in this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_face_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The selected model and reports are included. Speech datasets are not needed merely to run the app. Face and video modes need the ~35 MB official FER+ model, verified by SHA-256. No GPU or paid API is needed. From Command Prompt, use `cd /d "C:\Users\ahmad\Desktop\Emotion Recognition"` then `.venv\Scripts\python.exe -m streamlit run app.py`. The system `python` from `C:\Users\ahmad` may select Python 3.14 without Streamlit.

## Modes

- **Voice only:** upload WAV, MP3, FLAC or OGG spoken audio or record via the browser microphone. Preview, then select **Analyze voice**. Silence, corrupt or too-short files fail; a conservative single-frequency check rejects pure tones. Clipped and very short clips receive a warning. Music, noise and crying can still yield unreliable labels because there is no validated speech or out-of-distribution detector. The optional developer trace gives the file hash, processed signal, feature version, saved-model hash and exact scores.
- **Face only:** capture a camera still or upload a video. OpenCV finds faces; FER+ scores the largest detected face in each sampled frame. Missing faces produce no prediction; multiple faces trigger a warning. No identity recognition is performed.
- **Voice + Face:** upload a short video with a visible face and audio track. The app extracts the actual audio, samples up to 12 frames, shows both independent results, and displays an explicitly experimental shared-class average. Disagreement remains visible. The fusion has no measured accuracy.

Browser camera and microphone permission are needed for capture. Uploaded video is limited to 50 MB and two minutes. The first 12 seconds of decoded audio are used for voice scoring. MP3 decoder support can vary; WAV is safest for a demo.

## Speech training and actual results

The retained **baseline calibrated SVM** used 46 features and RAVDESS only. Its six-actor held-out accuracy was 44.17% and macro F1 39.75%; baseline logistic regression was 40.00% / 39.27%, random forest 39.17% / 37.38%. A previous tuned SVM reached 44.72% / 40.86% but was not deployed because its gain was uncertain.

The selected **logistic regression with 130 features** adds MFCC delta and delta-delta summaries plus spectral bandwidth and rolloff. Training-only three-fold actor GroupKFold selected it by macro F1 (**0.4093**). The candidate uses 1,079 RAVDESS clips from 18 training actors plus 490 TESS clips from one actor (70 per compatible class). The six original RAVDESS test actors, TESS's second actor and all ten EmoDB actors remained untouched during model selection.

| Untouched test | Clips / speakers | Saved SVM accuracy / macro F1 | Selected model accuracy / macro F1 |
|---|---:|---:|---:|
| RAVDESS unseen actors | 360 / 6 | 44.17% / 39.75% | **45.83% / 45.12%** |
| TESS unseen actor | 1,400 / 1 | 33.86% / 24.08% | **37.79% / 33.37%** |
| EmoDB unseen German corpus | 454 / 10 | 11.01% / 10.72% | **31.94% / 30.04%** |

These results show better transfer on two *acted* external tests. EmoDB performance remains low, and TESS has only two female speakers. There is no labeled test of natural conversation, accents, phone recordings, noisy rooms or music; **real-world generalization has not been established**. Scores are model outputs, not calibrated certainty about a person. The full per-class reports, precision, recall, class distribution and confusion matrices are in [evaluation.json](results/expanded/evaluation.json) and [model report](reports/expanded_model.md).

The notorious near-fixed Fearful score of the old SVM on stock audio came from RBF saturation far outside its training features, not stale uploads or fabricated values. See [deep diagnostics](TESTING.md). The new model is a genuine classifier trained on more data, but an unsuitable file can still yield a misleading output.

## Reproduce training

Large raw files are excluded from Git. The [dataset audit and access guide](data/EXPANDED_DATASETS.md) explains source licenses and label maps. RAVDESS, TESS and EmoDB are downloaded from official repositories; TESS and EmoDB archives have recorded SHA-256 checks. To regenerate the experiment:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\download_expanded_datasets.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe scripts\train_expanded.py
.\.venv\Scripts\python.exe scripts\deploy_expanded.py
.\.venv\Scripts\python.exe -m pytest -q
```

`train_expanded.py` caches extracted features under ignored `data/cache/`, selects candidates using training actor groups, evaluates once on untouched tests, and writes `results/expanded/`. `deploy_expanded.py` verifies that the cached model matches the recorded training-CV winner. The old SVM, logistic regression and forest artifacts remain in `models/`. The experiment is CPU-compatible; initial feature extraction can take several minutes.

## Face and video evidence

The [ONNX Model Zoo FER+ network](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md) is a CNN with 64×64 grayscale input and eight expression outputs. It was trained outside this project. Local checks confirmed its input/output shape, finite scores, detection on a sample face, no prediction on blank frames, and video audio/frame extraction. An unlabeled sample image and composited video are pipeline fixtures, not accuracy benchmarks. FER2013 and RAF-DB access is described in the [dataset guide](data/EXPANDED_DATASETS.md). No independent face or audiovisual accuracy is claimed. See [testing log](TESTING.md).

## Project files

[Training report](reports/expanded_model.md) · [Dataset guide](data/EXPANDED_DATASETS.md) · [Testing evidence](TESTING.md) · [Project report](reports/project_report.md) · [Viva preparation](reports/viva_preparation.md). The code and small models are in this repository; all raw datasets, cache files and the large FER+ ONNX binary remain local.
