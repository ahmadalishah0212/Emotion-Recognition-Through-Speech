"""VocalLens: voice, face, and experimental combined expression cues."""
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
from src.feature_extraction import load_audio
from src.predict import predict_audio
from src.face import aggregate_frames, decode_image, load_face_model
from src.video import analyze_video_payload
from src.fusion import experimental_fusion

st.set_page_config(page_title="VocalLens | Hear the voice. See the emotion.", page_icon="◉", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#080d18;color:#e5edf7;}
.block-container {max-width:1100px;padding-top:2.6rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em;}
.hero {border:1px solid #25364e;background:linear-gradient(120deg,#101c2b,#0b1421);border-radius:24px;padding:2.1rem 2.3rem;margin-bottom:1.6rem;}
.hero h1 {font-size:3.2rem;margin:.3rem 0 .1rem;color:#f6fbff;}
.hero .tagline {font-size:1.25rem;color:#65d9e9;font-weight:600;margin:0 0 .4rem;}
.hero .support {color:#a9b9ca;margin:0;}
.eyebrow {color:#52d5e8;font-size:.75rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;}
.result {border:1px solid #2d4659;border-left:4px solid #42d4e7;background:#0e1b29;border-radius:18px;padding:1.25rem 1.5rem;margin:.65rem 0;}
.result .kind {color:#99aebe;font-size:.77rem;letter-spacing:.12em;text-transform:uppercase;font-weight:700;}
.result .emotion {color:#e9fcff;font:700 2rem 'Space Grotesk',sans-serif;line-height:1.25;}
.result .score {color:#7ddce9;font-size:.88rem;}
.stButton button[kind="primary"] {background:#09bdd2;color:#06121b;border:none;font-weight:700;border-radius:10px;}
.stButton button[kind="primary"]:hover {background:#5de1ed;color:#06121b;}
div[data-testid="stAlert"] {border-radius:12px;}
.footer {border-top:1px solid #223146;margin-top:2rem;padding-top:1rem;color:#879bad;font-size:.83rem;}
</style>""", unsafe_allow_html=True)


def show_result(result, modality, model):
    score = result["probabilities"][result["emotion"]]
    st.markdown(f"<div class='result'><div class='kind'>{modality} · {model}</div>"
                f"<div class='emotion'>{result['emotion'].title()}</div>"
                f"<div class='score'>Top model score {score:.1%}</div></div>", unsafe_allow_html=True)
    if score < 0.45:
        st.warning("The top score is low. Several expression categories remain plausible.")
    chart = pd.DataFrame({"Expression": [key.title() for key in result["probabilities"]],
                          "Model score": list(result["probabilities"].values())}).set_index("Expression")
    st.bar_chart(chart, color="#37cfe2", horizontal=True)


def input_details(name, payload, signal=None, sr=None):
    with st.expander("Input details · verify the analyzed file"):
        st.write(f"**File:** {name} · **Size:** {len(payload):,} bytes · **SHA-256:** `{hashlib.sha256(payload).hexdigest()}`")
        if signal is not None:
            st.write(f"**Processed audio:** {len(signal) / sr:.2f} s · {sr:,} Hz · {len(signal):,} samples · range {signal.min():.3f} to {signal.max():.3f}")
            if len(signal) / sr >= 12:
                st.caption("The speech model uses at most the first 12 seconds.")


def voice_analysis(payload, name):
    if len(payload) > 25 * 1024 * 1024:
        raise ValueError("Audio file must be under 25 MB")
    signal, sr = load_audio(payload)
    result = predict_audio(payload)
    show_result(result, "Voice", result["model"].replace("_", " ").upper())
    input_details(name, payload, signal, sr)
    st.caption("The speech model was trained on acted spoken sentences. Crying, laughter, music and unrelated recordings may receive misleading labels. Scores are model outputs, not a person's feelings.")
    return result


def face_analysis(frames):
    result = aggregate_frames(frames, load_face_model())
    if result is None:
        st.warning("No face was detected in the sampled image or video frames. No facial prediction was made.")
        return None
    show_result(result, "Face", "pretrained FER+")
    st.caption(f"Face found in {result['frames_with_face']} of {result['frames_sampled']} sampled frame(s). Frame scores were averaged once for the video result.")
    if result["multi_face_frames"]:
        st.warning(f"Multiple faces appeared in {result['multi_face_frames']} sampled frame(s). Only the largest detected face in each frame was analyzed; this may switch people between frames.")
    st.caption("Facial expression is an observable cue, not a reliable measure of internal emotion. FER+ has not been independently benchmarked on this project's video inputs.")
    return result


st.markdown("<div class='hero'><div class='eyebrow'>Multimodal expression analysis</div>"
            "<h1>VocalLens<span style='color:#e46b72'>.</span></h1>"
            "<p class='tagline'>Hear the voice. See the emotion.</p>"
            "<p class='support'>Multimodal emotion analysis through speech and facial expressions.</p></div>", unsafe_allow_html=True)
mode = st.radio("Choose analysis mode", ["Voice only", "Face only", "Voice + Face"], horizontal=True)
st.caption("Choose one mode. Each analysis uses the file or capture shown below; changing mode clears the displayed result.")

if mode == "Voice only":
    source = st.radio("Voice input", ["Upload audio", "Record with microphone"], horizontal=True)
    audio = (st.file_uploader("Upload a spoken recording", type=["wav", "mp3", "flac", "ogg"], key="voice_upload")
             if source == "Upload audio" else st.audio_input("Record a spoken sample", key="voice_microphone"))
    if audio is None:
        st.info("Upload a clear spoken clip or record one with your microphone to begin.")
    else:
        raw = audio.getvalue()
        st.audio(raw)
        st.caption(f"Ready: {audio.name} · {len(raw):,} bytes · SHA-256 {hashlib.sha256(raw).hexdigest()[:12]}")
        if st.button("Analyze voice", type="primary", use_container_width=True):
            try:
                with st.spinner("Decoding audio and scoring the speech model…"):
                    voice_analysis(raw, audio.name)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The audio could not be analyzed: {exc}")

elif mode == "Face only":
    source = st.radio("Face input", ["Camera photo", "Upload video"], horizontal=True)
    if source == "Camera photo":
        st.caption("The browser camera captures one still frame. For changing expressions over time, upload a video.")
        picture = st.camera_input("Capture a face", key="face_camera")
        if picture is not None and st.button("Analyze face", type="primary", use_container_width=True):
            try:
                with st.spinner("Detecting and scoring the face…"):
                    face_analysis([decode_image(picture.getvalue())])
                input_details(picture.name, picture.getvalue())
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The image could not be analyzed: {exc}")
    else:
        video = st.file_uploader("Upload a video", type=["mp4", "mov", "webm", "avi"], key="face_video")
        if video is not None:
            st.video(video)
            if st.button("Analyze video faces", type="primary", use_container_width=True):
                try:
                    with st.spinner("Sampling frames and analyzing faces…"):
                        clip = analyze_video_payload(video.getvalue(), Path(video.name).suffix)
                        face_analysis(clip["frames"])
                    st.caption(f"Video: {clip['duration']:.1f} s · {clip['frame_count']} frames · {clip['sampled']} sampled")
                    input_details(video.name, video.getvalue())
                except (ValueError, FileNotFoundError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"The video could not be analyzed: {exc}")

else:
    st.caption("Upload a video with a visible face and spoken audio. Browser camera and microphone widgets cannot capture a synchronized clip here; video upload keeps both cues together.")
    video = st.file_uploader("Upload a video with voice and face", type=["mp4", "mov", "webm", "avi"], key="combined_video")
    if video is not None:
        st.video(video)
        if st.button("Analyze both cues", type="primary", use_container_width=True):
            try:
                with st.spinner("Extracting video audio and sampling frames…"):
                    clip = analyze_video_payload(video.getvalue(), Path(video.name).suffix)
                st.caption(f"Video: {clip['duration']:.1f} s · {clip['sampled']} of {clip['frame_count']} frames sampled")
                left, right = st.columns(2)
                with left:
                    st.subheader("Voice cue")
                    if clip["audio"]:
                        try:
                            voice_result = voice_analysis(clip["audio"], "Video audio track")
                        except ValueError as exc:
                            st.warning(f"Audio cue unavailable: {exc}")
                            voice_result = None
                    else:
                        st.warning("This video has no decodable audio track. No voice prediction was made.")
                        voice_result = None
                with right:
                    st.subheader("Face cue")
                    face_result = face_analysis(clip["frames"])
                combined = experimental_fusion(voice_result, face_result)
                if combined:
                    st.subheader("Experimental combined cue")
                    show_result(combined, "Combined", "unvalidated late fusion")
                    st.caption("Shared classes only: neutral, happy, sad, angry, fearful, disgust, surprised. Calm (voice) and contempt (face) are excluded; each remaining vector is renormalized, then averaged equally. The models are not jointly calibrated, and this output has no measured accuracy.")
                else:
                    st.info("A combined cue needs both valid voice and face results. Available single-cue results are shown above.")
                input_details(video.name, video.getvalue())
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The video could not be analyzed: {exc}")

with st.expander("Models, sources and limits"):
    st.write("**Voice:** 46 MFCC, energy, zero-crossing and spectral features from 16 kHz mono audio feed the saved calibrated SVM. Held-out RAVDESS actor accuracy: 44.17%; macro F1: 39.75%.")
    st.write("**Face:** OpenCV detects face regions; pretrained ONNX Model Zoo FER+ scores a 64×64 grayscale crop. It performs no identity recognition. Download the model separately with checksum verification.")
    st.write("[RAVDESS data](https://zenodo.org/records/1188976) · [FER+ model and MIT license](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md)")
st.markdown("<div class='footer'>VocalLens · Ahmad Ali Shah, Saboor Khan, Sardar Bilal · Academic demonstration · Outputs are uncertain expression cues.</div>", unsafe_allow_html=True)
