# Expanded speech dataset audit

VocalLens keeps all raw recordings in ignored `data/raw/` or `data/external/`. The current experiment uses RAVDESS, TESS and EmoDB. The counts below distinguish the full validated collection from clips assigned to training.

| Corpus | Official source | Recordings / speakers | Conditions and labels | Access and fit |
|---|---|---:|---|---|
| RAVDESS speech | [Zenodo](https://zenodo.org/records/1188976) | 1,439 usable / 24 | English, studio acted sentences; angry, calm, disgust, fearful, happy, neutral, sad, surprised | CC BY-NC-SA 4.0; original baseline, 1 byte-duplicate removed. |
| TESS | [Borealis / University of Toronto](https://doi.org/10.5683/SP2/E8H2MF) | 2,800 / 2 | English, clean acted words, two female actors; angry, disgust, fear, happy, neutral, sad, pleasant surprise | CC BY-NC 4.0; `pleasant surprise` maps to `surprised`. Only 490 clips from OAF enter candidate training; YAF (1,400) is held out. One age and gender slice limits diversity. |
| EmoDB | [Zenodo release](https://zenodo.org/records/7447302) | 535 / 10; 454 compatible | German, studio acted speech; anger, boredom, disgust, fear, happiness, neutral, sadness | CC BY 4.0; all compatible clips are untouched cross-language test. 81 boredom files excluded because the application has no boredom class. |
| CREMA-D | [Official repository](https://github.com/CheyneyComputerScience/CREMA-D) | 7,442 / 91 | English, acted sentences in varied intensities; angry, disgust, fear, happy, neutral, sad; audio and video | ODbL 1.0 / DbCL; authors request an [access/usage form](https://github.com/CheyneyComputerScience/CREMA-D/blob/master/docs/README.md). Large Git LFS archive, so not downloaded here. The filename parser is implemented for a future authorized local copy. |
| SAVEE | [Official Surrey page](https://kahlan.eps.surrey.ac.uk/savee/Download.html) | 480 / 4 | English, male actors, audiovisual acted speech; anger, disgust, fear, happiness, neutral, sadness, surprise | Registration for research download; not used. Very small speaker diversity. |
| IEMOCAP | [USC SAIL](https://sail.usc.edu/iemocap/iemocap_release.htm) | ~12 hours / 10 | English, improvised and scripted audiovisual dialogue; several labels including anger, sadness, happiness, neutral, frustration, excitement | Signed academic release agreement; not used. Frustration/excitement require careful label policy. |
| MSP-IMPROV | [UT Dallas MSP Lab](https://www.lab-msp.com/MSP/MSP-Improv.html) | 8,438 turns / 12 | English, improvised audiovisual interaction; anger, happiness, sadness, neutral and other annotations | Academic license with signed agreement; not used. |

The `src/dataset_sources.py` scanners preserve corpus, speaker, language, path, label and SHA-256 for each validated sample. Invalid labels, unreadable or <0.4-second audio and byte-identical duplicates are rejected. The training script fails if expected counts, speaker separation or validation checks fail. It excludes incompatible labels rather than relabeling them. Hash duplicates cannot detect near-duplicate recordings with different encodings.

## Reproduce the public downloads

From the project directory, after installing `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\download_expanded_datasets.py
.\.venv\Scripts\python.exe scripts\train_expanded.py
```

The expanded downloader uses the official Borealis dataset ZIP API and official Zenodo EmoDB file, checks SHA-256 against this project's inspected archives, then extracts safely. For CREMA-D, complete the authors' usage form and obtain the official Git LFS repository; put its WAV files under `data/external/cremad/` before using `scan_cremad`. SAVEE, IEMOCAP and MSP-IMPROV require their respective official access steps. None of their files are needed to run the current app.

The speaker-independent protocol and actual results are in `results/expanded/evaluation.json` and `reports/expanded_model.md`. TESS OAF-only training is an exploratory, narrow expansion. EmoDB is held out entirely, so its result probes transfer to German acted speech. No tested corpus establishes real-world accuracy on uncontrolled speech, accents, phone audio, music or noisy conditions.

## Face and audiovisual data access

The deployed face network is [OpenCV Zoo's Progressive Teacher](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) (Apache 2.0) with [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT) landmarks and 112×112 alignment. The former FER+ model is retained for comparisons. [RAF-DB](http://www.whdeng.cn/RAF/model1.html) requires researcher access; the original [FER2013 competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) requires accepting Kaggle terms. Neither is silently used or assigned a local benchmark score. The [official RAVDESS record](https://zenodo.org/records/1188976) provides audiovisual Actor 01 and Actor 05 ZIPs under CC BY-NC-SA 4.0; they were downloaded locally with official MD5 verification into ignored `data/cache/` and used for the labeled acted-video comparison in `reports/face_improvement.md`.

After obtaining `fer2013.csv` through Kaggle under its terms, run `python scripts/benchmark_face_fer2013.py PATH_TO_CSV --usage PrivateTest`. This measures the retained FER+ classifier on 48×48 cropped FER2013 pixels; it does not test face detection or video and is not an independent in-the-wild benchmark. The source CSV must stay outside Git. For the current acted audiovisual evaluation, run `python scripts/download_ravdess_av.py 01 05` and the commands in `reports/face_improvement.md`. The two large ZIPs are optional for running the app and remain outside Git. CREMA-D, SAVEE, IEMOCAP and MSP-IMPROV remain future dataset opportunities subject to their access terms; none has a claimed new score.
