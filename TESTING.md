# Testing log

## Final presentation pass — 23 November 2026

Existing environment: Python 3.12, existing `.venv`; a fresh installation in a separate environment was **not** tested. The documented explicit interpreter starts Streamlit on localhost. No models were retrained or replaced. The face downloader and model paths were reviewed; existing local weights were used. Syntax compilation and `pip check` passed. No TODO/FIXME markers were found in `src/` or `app.py`.

Command: `.\.venv\Scripts\python.exe -m pytest -q`. Initial result: `29 passed, 3 warnings in 40.54s`. Final result after image-layout polish: `29 passed, 3 warnings in 69.54s (0:01:09)`. The warnings are existing `audioread` Python 3.12 deprecations. This covers byte-driven audio prediction, tone rejection, silence/invalid input paths, image formats and size bounds, multiple/no faces, fusion class mapping, and local video audio/blank/silent fixtures. Model tests use genuine inference when local models/fixtures exist; none skipped in this run.

`scripts/final_demo_check.py` ran genuine speech, photo, video and fusion inference for the first Happy, Sad, Angry and Neutral clips of Actor 05. Results are in `results/final_demo_check.json`; these four deterministic probes are not a new accuracy benchmark. A dim photo detected one face and produced a dim-light warning; a blurred photo still detected one face without a blur warning; the duplicated photo detected two faces and a blank photo detected none. Glasses, physical hardware, live capture, GPU, and comprehensive angle/noise robustness were not newly tested.

Browser workflows tested: upload WAV → Analyze voice; switch modes (previous results removed); upload JPG → preview → Analyze photo; video → 12-frame face result; same-video voice/face → separate scores, disagreement and experimental fusion; webcam permission/unavailable interface. Happy-labeled audio scored Angry 43.9%, the sampled Happy photo scored Happy 99.8%, Happy video face averaged Happy 99.0%, and fusion scored Happy 54.4%. Sad video face scored Neutral 80.8%. Results reflect errors openly. Screenshots are real 1280×720 captures, individually framed and documented in `docs/SCREENSHOTS.md`. An initial browser tab expired; reopening and repeating succeeded. Physical microphone/webcam and GPU inference remain unverified. No new latency or peak-memory benchmark was performed.

UI changes add layered midnight/rose surfaces, brighter coral controls, local system typography, compact image previews, visible keyboard focus and reduced-motion rules. Existing model caching, bounded 12-frame sampling, image limits, 50 MB/two-minute video limit and CPU fallback are retained. No cloud inference or additional UI dependencies were added.

## Face replacement and photo upload (22 November 2026)

The baseline suite passed **19 tests** before this change. The final suite passed **29 tests** with three existing Python 3.12 `audioread` deprecation warnings. The new tests cover JPG, PNG and WebP decoding, corrupt and oversized images, no-face output, two separately numbered faces, real FER+ and OpenCV Zoo inference on a local ignored sample, video audio/frame extraction, blank and silent video, and a 960-pixel sampled-frame limit. The final command was `.\.venv\Scripts\python.exe -m pytest -q`.

The official RAVDESS Actor 01 and Actor 05 audiovisual ZIPs were checksum-verified, then 60 videos per actor were streamed from ZIP for two face models. The 52 clips per actor with common face labels had accuracy, macro precision, macro recall, macro F1, per-class scores and confusion matrices saved under `results/face/`. FER+ frequently predicted Neutral (Actor 05: 37/52) and missed all eight Sad clips. OpenCV Zoo reduced Neutral predictions to 14/52 and recovered three Sad clips; its Actor 05 full-frame accuracy/macro F1 were **0.5192/0.4500**, versus FER+ **0.3462/0.2850**. The exact model comparison and caveats are in `reports/face_improvement.md`. On Actor 05, the new face/voice full-frame fusion reached **0.6154 accuracy / 0.575 macro F1**; this remains a one-speaker acted-corpus result.

The Streamlit browser check showed Photo, Video and Webcam as selectable Face-only inputs. Uploading the local sample `lena.jpg` produced a 512×512 preview, then a numbered face box/crop, a real OpenCV Zoo top score of Neutral **81.3%**, alternative Happy **14.9%**, and all seven probability bars. Those single-image scores have no ground-truth annotation and are not an accuracy result. Browser video upload and voice result switching were checked in the previous upgrade. Physical webcam/microphone capture and GPU inference were not tested. CPU inference was used; ONNX Runtime's available providers were Azure and CPU, without CUDA.

## Expanded model and premium UI (22 November 2026)

The expanded data audit found 1,439 valid RAVDESS clips (24 speakers; one exact duplicate excluded), 2,800 valid TESS clips (two speakers), and 454 compatible EmoDB clips (ten speakers; 81 boredom clips excluded). There were no TESS or EmoDB corrupt files or byte duplicates. TESS and EmoDB official archive SHA-256 hashes were verified before extraction. `scripts/train_expanded.py` cached 46- and 130-feature vectors for all **4,693 compatible clips**, then compared six candidates using training-only actor GroupKFold. The winner was the 130-feature balanced logistic model, CV macro F1 **0.4093**. `scripts/deploy_expanded.py` checked the report and saved artifact before changing `models/selected.json`. The original SVM remains in `models/`.

| Untouched test | Saved SVM accuracy / macro F1 | New model accuracy / macro F1 |
|---|---:|---:|
| RAVDESS six actors, 360 clips | 0.4417 / 0.3975 | **0.4583 / 0.4512** |
| TESS YAF, 1,400 clips | 0.3386 / 0.2408 | **0.3779 / 0.3337** |
| EmoDB, ten unseen speakers, 454 clips | 0.1101 / 0.1072 | **0.3194 / 0.3004** |

The exact per-class reports, precision, recall, confusion matrices, speaker lists and counts are in `results/expanded/evaluation.json`. These external tests are acted speech; no real-world corpus was measured.

The redesigned Streamlit app rendered in the browser at localhost:8502. All three mode selectors changed the active card and workflow. A local three-second video upload showed 12 sampled frames, an extracted audio track, **Voice: Surprised 83.3%**, **Face: Neutral 97.0%**, and an explicitly experimental **Combined: Neutral 54.5%** result. The distinct voice and face results remained visible. These numbers are outputs for the fixture, not ground-truth accuracy. Removing a first voice upload removed its displayed result; a second WAV produced a different chart and result. The first RAVDESS file displayed Angry 79.3%, the second Surprised 58.8%. Input bytes, model output and chart values were checked in the browser. The hero, navigation and red Analyze button were visually inspected at laptop viewport width. Physical microphone and camera permission/capture were not verified.

`python -m pytest -q` passed **19 tests** after adjusting the selected-model tests and adding external-dataset mapping checks; three warnings come from `audioread` deprecated Python 3.12 modules. The suite includes current model feature-version compatibility, real WAV score changes and rejection of pure tones before app prediction. A separate local fixture check produced a 3.0-second, 12-frame audiovisual result (audio 95,652 bytes; voice Surprised, face Neutral, combined Neutral), `None` for blank-video face analysis, and `None` for silent-video audio. The input quality guard scanned all 1,440 RAVDESS WAVs with **zero false rejections or warnings**. FER2013/RAF-DB labeled benchmark data and official RAVDESS audiovisual videos are not present; no face or audiovisual accuracy result is claimed.

## Deep audio investigation and red accent restoration (22 November 2026)

`scripts/deep_audio_diagnostic.py` processed **24 known-label RAVDESS clips** (all eight emotions from Actors 01, 05 and 10), **three stock MP3s**, and **two generated pure tones**. It recorded each file's hash, processed duration and signal shape, feature dimensions, all eight probabilities from each saved model, top-two margin, and repeated-inference agreement. All 29 inputs had distinct hashes and distinct 46-feature vectors; repeated inference matched exactly. All three models use the same class order: angry, calm, disgust, fearful, happy, neutral, sad, surprised. Their `predict()` outputs matched the top `predict_proba()` class. The JSON and CSV are in `results/diagnostics/`.

The diagnostic independently reloaded all three **saved** models and re-predicted all **360 held-out actor clips**. It exactly reproduced the original metrics, so the saved artifacts match the recorded evaluation:

| Saved model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Calibrated SVM | 0.4417 | 0.4391 | 0.4219 | 0.3975 |
| Logistic regression | 0.4000 | 0.4020 | 0.3932 | 0.3927 |
| Random forest | 0.3917 | 0.3740 | 0.3854 | 0.3738 |

The exact source of the repeatedly reported ~44.4% is the SVM's `predict_proba()` result for audio far from acted speech. The stock angry MP3 returns Fearful **0.444647515** (displayed as **44.5%** with one-decimal rounding); the laughter MP3 returns Fearful **0.441609287**; the crying MP3 returns Fearful **0.407634972**. Their charts are different. More decisively, two generated one-second tones with different audio bytes and feature vectors both return the **identical full score vector**, with Fearful **0.444324897**. Their maximum RBF similarity to each calibration fold's nearest support vector is around **10⁻²⁰ to 10⁻²¹**. With negligible similarity to support vectors, SVM decision scores approach a nearly fixed value and the saved calibration maps them to nearly fixed probabilities. The stock angry file's maximum similarity is only about **0.006**. This explains genuine model saturation on unsuitable inputs; there is no hardcoded 44.4%, class-index mismatch, cached score vector, or decoder identity issue in the tested path.

`scripts/evaluate_audio_tuning.py` tested five SVM C/gamma settings with **3-fold actor GroupKFold on the 18 training actors**, then evaluated the CV-selected candidate once on the untouched six-actor test set. The candidate's test accuracy was **0.4472** and macro F1 **0.4086**, versus saved baseline **0.4417 / 0.3975**. Its actor-bootstrap 95% interval for the macro-F1 gain is **-0.0146 to +0.0373**. The CV difference between the candidate and current C=10 setting was also only about **0.00056** macro F1. This is insufficient evidence for a dependable improvement, so **no model was replaced or retrained for the app**. The experiment and train-only feature-distance audit are saved in `results/diagnostics/audio_tuning.json`.

The distance audit is descriptive, not a validated out-of-distribution classifier. After a scaler fit on training actors only, nearest-training-clip distance among held-out RAVDESS clips had median **4.54** and maximum **8.26**. The crying, angry and laughter MP3 distances were **8.18**, **15.16** and **11.88**, respectively. The stock files have no reliable eight-class ground-truth annotation, so no external-audio accuracy is claimed.

The live Streamlit app was tested with the stock angry MP3, then the crying MP3, then an Actor_05 angry WAV. The filename, developer SHA-256, feature sample, result card and chart changed together: Fearful **44.5%**, Fearful **40.8%**, then Angry **38.7%**. The chart's Fearful bars were **0.444647515** and **0.407634972**, exactly matching the trace and saved-model diagnostic. No stale result remained when the upload changed. The Analyze button and result highlight visibly used the existing VocalLens red `#e46b72`; the dark background and cyan chart remained. The repository history contained this red value in the VocalLens header dot but no earlier custom red button rule.

`python -m pytest -q` passed **16 tests** after the final changes (three existing `audioread` deprecation warnings). The suite includes a real held-out angry/happy WAV score-change check and a deliberate saturation check showing that different pure tones can legitimately have equal probabilities. Webcam and microphone hardware capture remain unverified; face and video behavior was left intact.

Repeat the diagnostic from this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\deep_audio_diagnostic.py "C:\Users\ahmad\Downloads\universfield-female-crying-291098.mp3" "C:\Users\ahmad\Downloads\universfield-angry-female-144242.mp3" "C:\Users\ahmad\Downloads\floraphonic-child-laughter-3-186730.mp3"
.\.venv\Scripts\python.exe scripts\evaluate_audio_tuning.py "C:\Users\ahmad\Downloads\universfield-female-crying-291098.mp3" "C:\Users\ahmad\Downloads\universfield-angry-female-144242.mp3" "C:\Users\ahmad\Downloads\floraphonic-child-laughter-3-186730.mp3"
```

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

`scripts/diagnose_uploads.py` inspected three stock MP3s from Downloads. All decoded, had different hashes, audio durations and 46-feature vectors. The crying file processed to 12.0 seconds, 16 kHz, 192,000 samples; the angry file to 2.116 seconds; the laughter file to 7.008 seconds. Crying and angry feature vectors were 716.93 Euclidean units apart, and angry and laughter 700.68 apart. The three were all labeled Fearful, but scores were respectively **0.407635, 0.444648, and 0.441609**. In the original browser app, crying and angry produced visibly different score charts at those same values. The repeated top class reflects model generalization limits on stock non-speech material, not identical input or stale chart data in the tested path. The updated app shows the hash and processed-input details when **Developer audio trace** is enabled and cautions about the domain mismatch.

Repeat the check on the user's laptop with:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_uploads.py "C:\Users\ahmad\Downloads\universfield-female-crying-291098.mp3" "C:\Users\ahmad\Downloads\universfield-angry-female-144242.mp3" "C:\Users\ahmad\Downloads\floraphonic-child-laughter-3-186730.mp3"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In **Voice only**, enable **Developer audio trace**, upload each file in turn and select **Analyze voice**. Check that the filename, hash and score bars update. For known speech labels, use `scripts/diagnose_predictions.py`, which checks 24 real RAVDESS WAVs across all eight classes and three actors. Face and combined checks require `python scripts/download_face_model.py` first.

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
