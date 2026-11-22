# Facial expression investigation — 22 November 2026

## What changed

VocalLens now uses OpenCV Zoo's Progressive Teacher / MobileFaceNet expression model with YuNet face and landmark detection. The retained FER+ path remains available for reproduction. The app offers photo, video, and webcam still inputs. Every detected photo face has a numbered box, crop and its own seven-class score distribution. Video still samples up to 12 frames and scores the largest face per frame.

## FER+ audit and Neutral diagnosis

The [official FER+ ONNX model](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md) is a CNN trained on FER+ annotations of FER2013. It accepts N×1×64×64 grayscale *raw pixel* tensors and outputs eight logits in this order: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt. The project used that class order, raw pixels, and softmax; no wrong class index or normalization was found. It used a Haar rectangle crop without facial landmark alignment. The model output is a distribution over trained expression labels, not calibrated certainty about inner emotion. The available evidence points to model and crop/domain limitations on these acted videos, not a proven preprocessing bug. The exact contribution of crop versus model cannot be isolated by this comparison because both changed.

The [OpenCV Zoo model](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) uses seven labels (angry, disgust, fearful, happy, neutral, sad, surprised). The official preprocessing is a landmark-aligned 112×112 RGB face with values scaled to [-1, 1]. YuNet supplies five landmarks. VocalLens applies those steps, then softmax to the model logits. It neither reweights classes nor suppresses Neutral. YuNet is MIT licensed; the OpenCV Zoo expression model is Apache 2.0 licensed. Both ONNX files are checksum-verified by `scripts/download_face_model.py` and excluded from Git.

## Labeled comparison

The official [RAVDESS audiovisual speech archives](https://zenodo.org/records/1188976) provide acted intended-emotion labels. Actor 01 was inspected for model and frame-sampling choices. Actor 05 was held out from the **speech** model training and is the independent actor check for voice/fusion. Calm clips have no face class and are excluded; the seven shared classes contain 52 clips per actor (Neutral 4; six others 8 each). Each video contributes up to 12 uniformly sampled frames; the full-frame setting was selected because it had slightly higher macro F1 than central-six on Actor 01. Its Actor 05 result was then checked once. Model comparisons on only two actors and acted labels have substantial uncertainty.

| Actor and method | Accuracy | Macro F1 | Sad recall | Neutral predictions / 52 |
|---|---:|---:|---:|---:|
| 01 FER+ full | 23.08% | 0.168 | 0/8 | 41 |
| 01 OpenCV Zoo full | 36.54% | 0.328 | 1/8 | 16 |
| 05 FER+ full | 34.62% | 0.285 | 0/8 | 37 |
| 05 OpenCV Zoo full | **51.92%** | **0.450** | **3/8** | 14 |

On Actor 05, FER+ macro precision/recall were **0.281/0.393**; OpenCV Zoo reached **0.453/0.554**. The JSON stores all seven per-class precision, recall, and F1 values plus the confusion matrix.

On Actor 05, FER+ classified all four Neutral clips correctly but also classified many non-Neutral clips as Neutral. The replacement improves Sad recall without a forced threshold, yet misses five of eight Sad clips. Inspect the per-class precision, recall, F1, and confusion matrices in `results/face/ravdess_actor05_ferplus.json` and `results/face/ravdess_actor05_opencv_zoo.json`. Actor 01 results are in the corresponding Actor 01 JSON files. The original Actor 01 FER+ result was produced before the report gained its `face_model` field; its method is FER+ by the run command and filename.

### Multimodal check on held-out Actor 05

| Method, 52 common-label clips | Accuracy | Macro precision | Macro recall | Macro F1 | Sad recall |
|---|---:|---:|---:|---:|---:|
| Speech only | 44.23% | 0.514 | 0.429 | 0.417 | 4/8 |
| OpenCV Zoo face only, full frames | 51.92% | 0.453 | 0.554 | 0.450 | 3/8 |
| Equal mean of shared-class probabilities, full frames | **61.54%** | **0.593** | **0.625** | **0.575** | 4/8 |

The fusion renormalizes speech scores after excluding Calm, then averages them equally with face scores. These values are from one speaker in one acted corpus, with no joint calibration. They do not establish real-world or general fusion benefit. The app keeps the combined output labeled experimental and preserves separate cue results, including disagreement. `scripts/evaluate_fusion_report.py` adds the full-frame fusion metrics to the actor report.

## Reproduction and limits

```powershell
.\.venv\Scripts\python.exe scripts\download_face_model.py --include-ferplus
.\.venv\Scripts\python.exe scripts\download_ravdess_av.py 01 05
.\.venv\Scripts\python.exe scripts\evaluate_ravdess_av.py data\cache\Video_Speech_Actor_01.zip --model ferplus --output results\face\ravdess_actor01_ferplus.json
.\.venv\Scripts\python.exe scripts\evaluate_ravdess_av.py data\cache\Video_Speech_Actor_01.zip --model opencv_zoo --output results\face\ravdess_actor01_opencv_zoo.json
.\.venv\Scripts\python.exe scripts\evaluate_ravdess_av.py data\cache\Video_Speech_Actor_05.zip --model ferplus --output results\face\ravdess_actor05_ferplus.json
.\.venv\Scripts\python.exe scripts\evaluate_ravdess_av.py data\cache\Video_Speech_Actor_05.zip --model opencv_zoo --output results\face\ravdess_actor05_opencv_zoo.json
.\.venv\Scripts\python.exe scripts\evaluate_fusion_report.py results\face\ravdess_actor05_opencv_zoo.json data\cache\Video_Speech_Actor_05.zip
```

Two video ZIP downloads total about 1.1 GB and are optional for running the app. The RAVDESS license is CC BY-NC-SA 4.0. The ZIPs and model weights remain ignored. FER2013 requires acceptance of Kaggle competition terms; RAF-DB requires researcher access. Neither was used to make accuracy claims. Photos, real webcams, non-frontal faces, occlusions, poor light, and natural conversation have no measured expression accuracy here. Video samples can switch between people because there is no person tracker. CPU inference was tested; CUDA availability was checked but no CUDA provider was installed, so GPU inference was not tested.
