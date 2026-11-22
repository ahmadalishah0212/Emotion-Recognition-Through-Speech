# VocalLens classroom demo

Allow 5–7 minutes. Start from CMD:

```bat
cd /d "C:\Users\ahmad\Desktop\Emotion Recognition"
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_face_model.py
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the displayed localhost address. Use the existing Python 3.12 virtual environment. If it is missing, first run `py -3.12 -m venv .venv`.

## Prepare the samples

```bat
.venv\Scripts\python.exe scripts\download_ravdess_av.py 05
.venv\Scripts\python.exe scripts\prepare_demo.py
```

The optional Actor 05 download is about 560 MB. This creates local `demo/assets/` files named happy, sad, angry and neutral, each as WAV, JPG and MP4. They come from the first matching official audiovisual speech clip, without selecting for a correct model prediction. The WAV is the extracted audio and the JPG is a central sampled frame. Raw samples are ignored by Git.

Source: Livingstone & Russo (2018), RAVDESS, https://zenodo.org/records/1188976, CC BY-NC-SA 4.0. Images are extracted frames; audio is decoded to mono WAV. Use only within the source license, retain attribution, and apply the same license when sharing derivatives. The app needs neither the archive nor demo assets for ordinary use.

## Demonstration sequence

1. Voice only: upload `assets/happy.wav`, preview it, press Analyze voice. Explain the actual top class and alternatives, even if different from Happy.
2. Face only → Photo: upload `assets/happy.jpg`, press Analyze photo. Show the numbered detection and expression bars.
3. Face only → Video: upload `assets/sad.mp4`, press Analyze video faces. Explain that up to 12 frames are sampled, with the largest face scored in each.
4. Voice + Face: upload `assets/happy.mp4`, press Analyze both cues. Compare separate predictions and the experimental shared-class average.
5. Face only → Webcam: show the capture control. Only demonstrate actual capture after testing the device yourself.
6. Explain corpus results and limitations using `DEMO_SCRIPT.md`.

Remove the uploaded file or switch modes to reset. A new upload requires a new Analyze click; scores are not retained as a session history. Before presenting, rehearse once, download models, and keep the samples locally so the presentation works offline.

If Streamlit is missing, use the explicit `.venv\Scripts\python.exe` command rather than system Python. If face models are missing, rerun the model downloader. For a no-face message, use a larger front-facing image. For a busy port, add `--server.port 8502`. Stop with Ctrl+C. Camera/microphone permission and GPU inference require a separate hardware check.
