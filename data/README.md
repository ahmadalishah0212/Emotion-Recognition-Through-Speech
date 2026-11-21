# RAVDESS dataset setup

Download the **audio-only speech** archive from the [official Zenodo record](https://zenodo.org/records/1188976), file `Audio_Speech_Actors_01-24.zip` (about 209 MB as displayed by Zenodo). It contains 1,440 WAV recordings from 24 actors. Do not download the song or video archives for this project. The record lists a CC BY-NC-SA 4.0 license; cite Livingstone and Russo (2018) and use the data in accordance with the license.

From the project root, use `python scripts/download_dataset.py` for automatic download and checksum verification. If network access fails, download the named ZIP in a browser, place it at `data/raw/Audio_Speech_Actors_01-24.zip`, then run the script again. It verifies the official MD5 `bc696df654c87fed845eb13823edef8a` before extracting.

Expected structure: `data/raw/Actor_01/03-01-01-01-01-01-01.wav` through `Actor_24/...`. Run `python scripts/validate_dataset.py`. It checks filename labels, readable WAV headers, empty files and exact duplicate bytes. The downloaded official archive passed its MD5 check and contained **1,440 WAV filenames from 24 actors**. One pair of recordings for Actor_07 (`03-01-03-01-02-01-07.wav` and `03-01-03-01-02-02-07.wav`) has identical bytes; the loader excludes the second, leaving **1,439 usable clips**. No other invalid files were found. The audio and archive are excluded from Git.

Only audio-only speech is modeled. Filename fields give modality, vocal channel, emotion, intensity, statement, repetition, actor. Emotion IDs 01–08 map to neutral, calm, happy, sad, angry, fearful, disgust and surprised. There is no null tabular feature to impute; missing or corrupted files are reported and excluded. Duplicate audio bytes are excluded. Labels are handled directly by scikit-learn; no one-hot encoding is needed.

## Dataset choice

| Dataset | Suitability and access | Reason for decision |
|---|---|---|
| RAVDESS | 1,440 speech WAVs, eight labels, 24 actors, official direct download, CC BY-NC-SA 4.0 | Selected: clear labels, moderate download, actor-level evaluation. Neutral has fewer clips than most classes. |
| CREMA-D | 7,442 clips, six labels, 91 actors; official repository uses Git LFS and a large clone | Useful diversity, but more download and preprocessing complexity for a course project. |
| TESS | 2,800 clips, seven labels, CC BY-NC 4.0 | Only two female speakers make speaker-generalization assessment weak. |
| SAVEE | 480 utterances, seven labels, four male speakers; registration required | Small and access requires registration. |

Combining these sets would require label mapping and would mix recording conditions. We keep one corpus to make the evaluation understandable. RAVDESS is acted studio speech, so real-world voice performance may differ substantially.

Sources: [RAVDESS](https://zenodo.org/records/1188976), [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), [TESS](https://doi.org/10.5683/SP2/E8H2MF), [SAVEE](https://kahlan.eps.surrey.ac.uk/savee/).
