# Instructor requirements and verification checklist

## VocalLens extension

**Deep audio follow-up:** The saved SVM's repeated ~44% Fearful output on distant inputs was traced to RBF kernel saturation and subsequent calibration. A 29-input diagnostic and a full 360-clip re-evaluation of all three saved models reproduced the original metrics. A small actor-disjoint tuning gain did not justify replacing the model. The app now offers an optional developer trace and restores the existing red `#e46b72` on the Analyze button and result highlight. See `TESTING.md` and `results/diagnostics/`.

The original PDF requirements below concern speech classification and remain satisfied by the existing three-model evaluation. The user-requested multimodal extension adds:

| Extension | Status and evidence |
|---|---|
| VocalLens branding and three clear modes | **Verified in browser:** Voice only, Face only, Voice + Face; dark theme and cyan/red accents preserved. |
| Exact stock MP3 diagnosis | **Verified:** three distinct hashes and feature vectors; all predicted Fearful with different scores. Browser charts differed for crying and angry uploads. See `TESTING.md`. |
| Camera face capture | **Implemented:** `st.camera_input` analyzes a still. **Physical webcam capture unverified** in this browser session. |
| Video face analysis | **Verified:** local three-second sample video yielded 12 sampled frames and a face prediction. Blank frame produced no prediction. |
| Video audio extraction | **Verified:** video with audio yielded a WAV track and speech result; silent video returned no audio. |
| Combined analysis | **Verified as a functional display:** separate voice and face outputs plus shared-class experimental fusion. **No measured multimodal accuracy**. |
| Face model provenance | **Verified:** pretrained MIT-licensed ONNX Model Zoo FER+ downloaded with SHA-256 check. Our group did not train or benchmark its accuracy. |


**Source:** `AI Project Requirements.pdf` (2 pages). **Verified** means code or documentation was executed or inspected; **Pending** identifies a remaining user-facing verification. The instructor requires three team members, 2–4 trained models and a web/mobile app.

| PDF requirement | Location | Status / verification |
|---|---|---|
| Three group members | `reports/team_roles.md`, README | **Verified**: three names and responsibility areas documented. |
| Clear problem statement | `reports/project_report.md` | **Verified**: acted speech-expression classification defined. |
| Dataset description and source | `data/README.md`, report | **Verified**: official RAVDESS ZIP downloaded, checksum matched, source and license documented. |
| Input features and target/output | `src/feature_extraction.py`, `src/dataset_loader.py`, report | **Verified**: real audio produced 46 features; eight target labels audited. |
| Handle missing values | `src/dataset_loader.py`, report | **Verified**: 1,440 filenames, no corrupt WAVs; no tabular nulls to impute. |
| Remove duplicates | `src/dataset_loader.py` | **Verified**: one byte-identical duplicate excluded, leaving 1,439 usable clips. |
| Encode categorical data | `src/train.py`, report | **Verified**: scikit-learn handled string class labels during actual training. |
| Scale features if required | `src/train.py` | **Verified**: StandardScaler fit inside SVM and logistic pipelines on training actors only. |
| Train/test split | `src/train.py`, data audit | **Verified**: 1,079 training and 360 test clips; six actors held out. |
| Analyze dataset | notebook, `src/eda.py`, report | **Verified**: dataset audited and EDA script executed. |
| Appropriate visualizations and graphs | `results/figures/` | **Verified**: class/actor plots, waveform, spectrogram, three matrices and comparison chart saved; key plots visually inspected. |
| Identify important patterns and relationships | report | **Verified**: neutral class has half the usual count; sample waveform has leading/trailing silence. |
| Select 2–4 appropriate ML/DL/NLP models | `src/train.py` | **Verified**: calibrated SVM, random forest and logistic regression trained. |
| Train and test each model | `src/train.py`, `results/metrics/` | **Verified**: all three evaluated on the same held-out actors. |
| Proper evaluation metrics | `results/metrics/` | **Verified**: accuracy, macro precision/recall/F1, class reports and confusion matrices generated. |
| Compare models | `results/model_comparison/comparison.csv`, report | **Verified**: real comparison table and chart generated. |
| Suggest and explain best model; understand it | `models/selected.json`, report, viva guide | **Verified**: calibrated SVM selected by macro F1 0.3975; weaknesses documented. |
| Integrate selected model with web/mobile app | `app.py`, `src/predict.py` | **Verified**: real saved SVM loaded; two browser uploads produced distinct predictions. Exclusive source selection prevents a stale microphone recording from overriding an upload. |
| User input and prediction demonstration | `app.py`, `TESTING.md` | **Verified** for two different uploads and predictions; microphone control rendered but capture remains unverified. |
| GitHub source link | README, remote `origin` | **Verified**: public repository created; baseline, dataset/EDA and model milestones pushed. |
| Project report | `reports/project_report.md` | **Verified**: real dataset, plots, metrics, selection and limitations recorded. |
| Dataset/source link | `data/README.md`, report, README | **Verified**: official Zenodo URL and checksum documented. |
| Web/mobile app demo | README, `TESTING.md` | **Verified** for browser upload/prediction; **pending** classroom microphone demonstration. |

Additional requested deliverables: Windows setup and GitHub commands are in README; executed test evidence is in `TESTING.md`; the division for all three students is in `reports/team_roles.md`. The remaining hardware-dependent check is microphone capture in the classroom browser.
