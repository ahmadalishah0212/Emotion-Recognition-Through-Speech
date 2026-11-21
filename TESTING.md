# Testing log

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
| Run `python -m pytest -q` | **Passed**: 8 tests after the SVM score regression test was added. |
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
