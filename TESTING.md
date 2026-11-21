# Testing log

## VocalLens multimodal upgrade (22 November 2026)

| Check | Observed result |
|---|---|
| `python -m pytest -q` | **14 passed**, three Python 3.12 `audioread` deprecation warnings. |
| `python -m compileall -q app.py src scripts` and `pip check` | Passed; no broken requirements. |
| Official FER+ model download | 35,040,571 bytes; SHA-256 `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`. ONNX Runtime input `(1,1,64,64)` and output `(1,8)` confirmed. |
| Real face inference | OpenCV detected one face in a local OpenCV sample image; FER+ produced eight finite scores summing to one. A black image produced no face result. Sample image kept only in ignored `data/cache/`. |
| Multiple and missing faces | Duplicating the sample image side by side yielded two detections and set the multi-face warning count to one; a ten-frame blank video yielded no face result. |
| Video decoding | A local three-second MP4 with one repeated face frame and RAVDESS audio produced 12 sampled frames, a WAV audio track, separate predictions and a combined display score. A silent MP4 returned `audio=None`. These are pipeline tests, not accuracy results. |
| Browser exact MP3 | `universfield-female-crying-291098.mp3` uploaded and analyzed in VocalLens. The displayed short hash `597fa4a4bb08` matches local bytes; Fearful score 0.407635 and the full bar chart rendered. |
| Browser combined mode | Uploaded local MP4, saw voice Disgust, face Neutral and **experimental** combined Neutral, with separate charts and explanatory text. Switching from voice to combined removed the voice upload and its result. |
| Physical microphone and webcam | Controls render, but hardware capture and permission flow remain **unverified** in this browser session. |

### Exact stock-audio diagnosis

`scripts/diagnose_uploads.py` inspected three stock MP3s from Downloads. All decoded, had different hashes, audio durations and 46-feature vectors. The crying file processed to 12.0 seconds, 16 kHz, 192,000 samples; the angry file to 2.116 seconds; the laughter file to 7.008 seconds. Crying and angry feature vectors were 716.93 Euclidean units apart, and angry and laughter 700.68 apart. The three were all labeled Fearful, but scores were respectively **0.407635, 0.444648, and 0.441609**. In the original browser app, crying and angry produced visibly different score charts at those same values. The repeated top class reflects model generalization limits on stock non-speech material, not identical input or stale chart data in the tested path. The new app displays a hash and detailed processed-input information and cautions about this domain mismatch.

Repeat the check on the user's laptop with:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_uploads.py "C:\Users\ahmad\Downloads\universfield-female-crying-291098.mp3" "C:\Users\ahmad\Downloads\universfield-angry-female-144242.mp3" "C:\Users\ahmad\Downloads\floraphonic-child-laughter-3-186730.mp3"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In **Voice only**, upload each file in turn and select **Analyze voice**. Check that the filename, hash and score bars update. For known speech labels, use `scripts/diagnose_predictions.py`, which checks 24 real RAVDESS WAVs across all eight classes and three actors. Face and combined checks require `python scripts/download_face_model.py` first.

## Checks actually performed in this environment

| Check | Result |
|---|---|
| `python -m compileall -q app.py src scripts tests` using bundled Python 3.12 | **Passed**: syntax compiles. |
| Parse `notebooks/exploratory_analysis.ipynb` as JSON | **Passed**. |
| Standard-library dataset parser and duplicate smoke check | **Passed** with two temporary WAV files; one duplicate excluded. |
| Install required libraries from PyPI into `.venv` | **Passed** after retry with network access; imports and versions checked. |
| Verify pinned model dependencies | **Passed**: `pip check` reported no broken requirements and `pip install --no-index -r requirements.txt` found all requirements installed. |
| Download official RAVDESS ZIP | **Passed** with curl after Python downloader stalled; MD5 matched `bc696df654c87fed845eb13823edef8a`. |
| Validate dataset | **Passed**: 1,440 filenames, 24 actors, one byte-identical duplicate excluded, no corrupt files. |
| Run `python -m src.eda` | **Passed**: three PNG graphs generated; class and sample-audio plots visually inspected. |
| Run `python -m pytest -q` | **Passed**: 9 tests after adding a distinct-audio feature regression test. Three deprecation warnings came from `audioread` on Python 3.12. |
| Train three models | **Passed**: SVM, logistic regression and random forest saved with reports and confusion matrices. |
| Real saved-model inference | **Passed**: selected SVM loaded without retraining; two held-out WAVs produced labels and eight scores summing to 1. The labels matched the highest scores after calibration. |
| Invalid audio and speaker separation | **Passed**: corrupt bytes raised a clear error; the six test actors and 18 training actors are disjoint. |
| Launch Streamlit | **Passed** with local loopback access: server returned HTTP 200 and browser page rendered. |
| Browser upload and prediction | **Passed**: real WAV uploaded; audio player, result card and score chart rendered. |
| Browser microphone capture | **Unverified**: Record control rendered, but the test browser did not provide a captured recording. |

`tests/test_dataset.py` covers label mapping, rejection of song files and duplicate byte detection. `tests/test_audio_predict.py` covers feature shape, finite values, empty audio, missing model, audio-to-prediction mechanics through a clearly synthetic **DummyClassifier**, and a regression check that calibrated SVM labels match the highest score. Synthetic tests are only mechanics tests; they are not performance results. The real model was separately tested on held-out RAVDESS WAVs.

## Next verification on a connected Windows laptop

The data audit reports 1,079 training and 360 test clips, with actors 05, 10, 15, 20, 23 and 24 held out. The comparison CSV and confusion matrices were inspected; `models/selected.json` names the top macro-F1 model. The browser upload flow was verified with a real WAV. For a classroom demo, grant browser microphone access and make a short live speech recording; that hardware-dependent capture remains to be checked on the student's laptop.

## Known limitations

The selected SVM's macro F1 is 0.3975 on one fixed six-actor split, with weak neutral, happy and sad recall. MP3 decoding may depend on local decoder availability; use WAV for the demo. The browser must grant microphone permission, and actual microphone capture was not confirmed in this test session.

## Repeated-prediction investigation (22 November 2026)

The old app rendered upload and microphone widgets at the same time, then used `recorded if recorded is not None else uploaded`. If a recording remained in the microphone widget, choosing another file on the Upload tab still submitted the old recording. This input-selection defect can produce identical predictions for apparently different uploads. This browser environment could not capture a microphone sample to reproduce that exact before-state interactively; the precedence follows directly from the old code.

`python scripts/diagnose_predictions.py` tested **24 real WAV files**: eight emotion labels for each of Actors 01, 05 and 10. It records audio duration, sample rate, signal range, SHA-256, feature shape/values, full score vectors and predicted labels in `results/metrics/prediction_diagnostics.json`. All 24 audio hashes were distinct. The app's bytes path and training's file-path extraction produced matching 46-value vectors. Every vector was finite; the minimum pairwise feature distance was **30.5620**, and the minimum pairwise score-vector distance was **0.045868**. The selected SVM predicted all eight classes across these files: surprised 5, calm 5, fearful 4, angry 3, happy 3, disgust 2, sad 1 and neutral 1. Its saved 360-clip confusion matrix shows predictions in all eight classes too, although neutral (6) and sad (17) are rare. The model is therefore weak on some classes but is **not** globally stuck on one label.

After the app fix, a browser upload of `Actor_05/03-01-05-01-01-01-05.wav` displayed **Angry**; replacing it with `Actor_10/03-01-03-01-01-01-10.wav` displayed **Happy**. The displayed filenames changed, and the first result's score chart had Angry as its highest score. Switching to microphone mode removed the uploader; switching back showed an empty uploader. A stale microphone input can no longer override an uploaded file. No model or evaluation metric changed, so retraining was unnecessary. Microphone capture itself remains unverified in the test browser; check it with browser permission and a real microphone on the presentation laptop.
