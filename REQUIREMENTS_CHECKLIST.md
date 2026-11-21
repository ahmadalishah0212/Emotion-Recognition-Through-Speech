# Instructor requirements and verification checklist

**Source:** `AI Project Requirements.pdf` (2 pages). Status **Implemented** means code or documentation exists; **Verified** means it was actually executed or inspected; **Pending** needs the real dataset, dependencies, a demo, or GitHub submission. The instructor requires three team members, 2–4 trained models and a web/mobile app.

| PDF requirement | Location | Status / verification |
|---|---|---|
| Three group members | `reports/team_roles.md`, README | Implemented; three names and responsibility areas documented. |
| Clear problem statement | `reports/project_report.md` | Implemented; speech emotion classification defined. |
| Dataset description and source | `data/README.md`, report | Implemented; official source and license documented. Local data pending. |
| Input features and target/output | `src/feature_extraction.py`, `src/dataset_loader.py`, report | Implemented; 46 features and eight labels specified. Execution pending. |
| Handle missing values | `src/dataset_loader.py`, report | Implemented for files: no tabular nulls; absent/corrupt files are reported. Real audit pending. |
| Remove duplicates | `src/dataset_loader.py` | Implemented byte-hash exclusion; real audit pending. |
| Encode categorical data | `src/train.py`, report | Implemented: string labels are encoded internally by scikit-learn. Real run pending. |
| Scale features if required | `src/train.py` | Implemented: StandardScaler inside SVM and logistic pipelines; real run pending. |
| Train/test split | `src/train.py` | Implemented: six held-out actors; real run pending. |
| Analyze dataset | `notebooks/exploratory_analysis.ipynb`, `src/eda.py` | Implemented; actual run pending. |
| Appropriate visualizations and graphs | `src/eda.py`, `src/train.py` | Implemented class/actor charts, waveform, spectrogram, matrices and comparison; PNG generation pending. |
| Identify important patterns and relationships | notebook, report | **Pending** actual data and plots; no observations invented. |
| Select 2–4 appropriate ML/DL/NLP models | `src/train.py` | Implemented SVM, random forest, logistic regression; training pending. |
| Train and test each model | `src/train.py` | **Pending** dataset and dependencies. |
| Proper evaluation metrics | `src/train.py` | Implemented accuracy, macro precision/recall/F1, class report and confusion matrix; scores pending. |
| Compare models | `src/train.py`, `results/model_comparison/` | Implemented output pipeline; actual table pending. |
| Suggest and explain best model; understand it | `src/train.py`, `reports/model_explanation.md`, viva guide | Selection logic implemented; **actual winner and explanation pending** real results. |
| Integrate selected model with web/mobile app | `app.py`, `src/predict.py` | Implemented saved-model Streamlit integration; live verification pending. |
| User input and prediction demonstration | `app.py`, README | Upload and microphone paths implemented; **demo pending** training and browser test. |
| GitHub source link | README, `.gitignore` | Repository ready; **pending** user repository creation and push. No link invented. |
| Project report | `reports/project_report.md` | Implemented draft; empirical sections explicitly pending. |
| Dataset/source link | `data/README.md`, report, README | Verified official Zenodo URL in browser research. |
| Web/mobile app demo | README | Demo procedure documented; **pending** actual model and live run. |

Additional requested deliverables: Windows setup and GitHub commands are in README; testing evidence is in `TESTING.md`; the division for all three students is in `reports/team_roles.md`; the notebook, training script and app are present. Update this checklist only after actual runs confirm the pending items.
