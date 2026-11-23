# ML pipeline audit — 23 November 2026

Baseline commit: dc8ff75. Speech uses 16 kHz mono, peak normalization, a 12-second maximum, 20 MFCC mean/std plus delta/delta-delta and spectral summaries (130 values). The selected saved pipeline is balanced Logistic Regression with StandardScaler. There is no explicit learned feature selection. Dataset validation removes byte duplicates; it cannot detect every near-duplicate encoding.

Training: 1,079 RAVDESS recordings from 18 actors plus 490 TESS OAF recordings (70 per common class). Three-fold RAVDESS actor GroupKFold selects models, adding TESS OAF only to fit folds. Scaling is inside pipelines. Previous SVM sigmoid calibration uses ordinary inner CV, so its calibration is not independently speaker validated. No actor crosses the outer train/validation split.

Historical tests: 360 RAVDESS clips from actors 05/10/15/20/23/24, 1,400 TESS YAF clips, 454 EmoDB clips. These have already been examined during previous upgrades. They must now be described as historical comparison data, not a newly untouched final test. The new experiment selects using training actors only. No new broad accuracy claim may be made from those historical tests.

Face: official OpenCV Zoo MobileFaceNet/Progressive Teacher model; YuNet five-point landmarks, affine alignment to 112×112, RGB scaled [-1,1], seven ordered logits converted by softmax. FER+ is retained as a baseline. Face and fusion comparisons used Actor 01 development and Actor 05 check; both have since been inspected. Existing face metrics are clip-level acted-label metrics, not photo accuracy.

Fusion: seven shared classes, renormalize voice after excluding Calm, equal average. No joint training or calibration. Actor 01 speech predictions from the deployed model would be in-sample and cannot legitimately tune fusion. Honest tuning needs out-of-fold speech predictions or actors withheld before training, plus additional AV actor data. Existing AV archives cover only actors 01 and 05.

Inference loads the selected model artifact and uses the same feature extractor as training. Face sessions and YuNet instances are cached. Speech currently reloads its small artifact for every prediction; safe caching can be added using file metadata to invalidate when artifacts change. Errors produce exceptions/unavailable cues, not fabricated scores.

Raw data/model downloads are ignored by Git. Existing speech/face reports and artifacts will be preserved. Two unrelated TeX files must remain untouched. New experiments record seeds, actor splits, parameters, per-class metrics, confusion matrices, and runtime separately.
