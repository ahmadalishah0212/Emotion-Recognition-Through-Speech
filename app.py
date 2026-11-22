"""VocalLens: voice, face, and experimental combined expression cues."""
import hashlib
from pathlib import Path
import cv2
import streamlit as st
from src.predict import predict_audio
from src.audio_debug import trace_audio
from src.face import decode_image
from src.face_opencv import aggregate_frames, analyze_photo, load_model as load_face_model
from src.video import analyze_video_payload
from src.fusion import experimental_fusion
from src.ui_theme import CSS, hero, mode_cards, result_dashboard
from src.audio_quality import assess_audio

st.set_page_config(page_title="VocalLens | Hear the voice. See the emotion.", page_icon="◉", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def show_result(result, modality, model):
    result_dashboard(result, modality, model)


def input_details(name, payload, signal=None, sr=None):
    with st.expander("Input details · verify the analyzed file"):
        st.write(f"**File:** {name} · **Size:** {len(payload):,} bytes · **SHA-256:** `{hashlib.sha256(payload).hexdigest()}`")
        if signal is not None:
            st.write(f"**Processed audio:** {len(signal) / sr:.2f} s · {sr:,} Hz · {len(signal):,} samples · range {signal.min():.3f} to {signal.max():.3f}")
            if len(signal) / sr >= 12:
                st.caption("The speech model uses at most the first 12 seconds.")


def voice_analysis(payload, name, debug=False):
    if len(payload) > 25 * 1024 * 1024:
        raise ValueError("Audio file must be under 25 MB")
    quality = assess_audio(payload)
    result = predict_audio(payload)
    show_result(result, "Voice", result["model"].replace("_", " ").upper())
    st.markdown(f"<div class='micro-note'>Analyzed audio: {quality['duration']:.1f} seconds · {quality['sample_rate']:,} Hz mono</div>", unsafe_allow_html=True)
    for warning in quality["warnings"]:
        st.warning(warning)
    if debug:
        with st.expander("Developer audio trace", expanded=True):
            st.write(f"**Current file:** {name}")
            st.json(trace_audio(payload, result))
    st.caption("The speech model was trained on acted spoken sentences. Crying, laughter, music and unrelated recordings may receive misleading labels. Scores are model outputs, not a person's feelings.")
    return result


def face_analysis(frames):
    result = aggregate_frames(frames, load_face_model())
    if result is None:
        st.warning("No face was detected in the sampled image or video frames. No facial prediction was made.")
        return None
    show_result(result, "Face", "OpenCV Zoo expression model")
    st.caption(f"Face found in {result['frames_with_face']} of {result['frames_sampled']} sampled frame(s). Frame scores were averaged once for the video result.")
    if result["multi_face_frames"]:
        st.warning(f"Multiple faces appeared in {result['multi_face_frames']} sampled frame(s). Only the largest detected face in each frame was analyzed; this may switch people between frames.")
    st.caption("Facial expression is an observable cue, not a reliable measure of internal emotion. This model was evaluated on acted RAVDESS video; personal photos may differ substantially.")
    return result


def photo_analysis(payload, name):
    result = analyze_photo(payload, load_face_model())
    if not result["faces"]:
        st.warning("No usable face was detected. Try a clearer, front-facing photo with better light.")
        return result
    st.success(f"Detected {len(result['faces'])} face(s). Each numbered region has its own result.")
    st.image(result["image"], caption="Detected face regions · numbers correspond to results below", use_container_width=True)
    for face in result["faces"]:
        st.subheader(f"Face {face['number']}")
        left, right = st.columns([1, 3])
        with left:
            st.image(face["crop"], caption=f"Detected face {face['number']}", use_container_width=True)
        with right:
            show_result(face, "Face", "OpenCV Zoo expression model")
            for warning in face["warnings"]:
                st.warning(warning)
    st.caption("Predictions describe visible expression cues, not identity or a person's feelings. This model has not been independently calibrated on personal photos.")
    input_details(name, payload)
    return result


hero()
mode = st.segmented_control("Analysis mode", ["Voice only", "Face only", "Voice + Face"], default="Voice only", label_visibility="collapsed")
mode_cards(mode)

if mode == "Voice only":
    st.markdown("<div class='section-label'>01 / Voice studio</div><div class='content-card'><h3>Listen beneath the words</h3><p>Upload spoken audio or record a sample. Review the exact recording before analysis.</p></div>", unsafe_allow_html=True)
    source = st.radio("Voice input", ["Upload audio", "Record with microphone"], horizontal=True)
    with st.sidebar:
        st.markdown("### Studio settings")
        debug_audio = st.toggle("Developer audio trace", value=False, help="Show decoded audio, feature and model details after analysis.")
    audio = (st.file_uploader("Upload a spoken recording", type=["wav", "mp3", "flac", "ogg"], key="voice_upload")
             if source == "Upload audio" else st.audio_input("Record a spoken sample", key="voice_microphone"))
    if audio is None:
        st.info("Upload a clear spoken clip or record one with your microphone to begin.")
    else:
        raw = audio.getvalue()
        st.audio(raw)
        st.caption(f"Ready: {audio.name} · {len(raw):,} bytes")
        if st.button("Analyze voice", type="primary", use_container_width=True):
            try:
                with st.spinner("Decoding audio and scoring the speech model…"):
                    voice_analysis(raw, audio.name, debug=debug_audio)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The audio could not be analyzed: {exc}")

elif mode == "Face only":
    st.markdown("<div class='section-label'>02 / Expression studio</div><div class='content-card'><h3>See the visible expression</h3><p>Analyze a photo, short video, or webcam still. Every detected face in a photo gets its own numbered result.</p></div>", unsafe_allow_html=True)
    source = st.segmented_control("Choose your input method", ["Photo", "Video", "Webcam"], default="Photo")
    descriptions = {"Photo": "▧  Still image · each detected face is scored separately",
                    "Video": "▷  Motion clip · up to 12 sampled frames",
                    "Webcam": "◉  Camera still · capture in your browser"}
    st.markdown(f"<div class='input-choice'><b>{source}</b><span>{descriptions[source]}</span></div>", unsafe_allow_html=True)
    if source == "Photo":
        st.caption("JPG, PNG or WebP · under 12 MB · up to 24 megapixels. Photos stay in this session only.")
        photo = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"], key="face_photo")
        if photo is not None:
            raw = photo.getvalue()
            try:
                frame = decode_image(raw)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Preview · {photo.name}", use_container_width=True)
                st.caption(f"Ready: {photo.name} · {frame.shape[1]} × {frame.shape[0]} pixels · {len(raw):,} bytes")
                if st.button("Analyze photo", type="primary", use_container_width=True):
                    with st.spinner("Detecting each face and scoring visible expressions…"):
                        photo_analysis(raw, photo.name)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("The photo could not be analyzed. Try a clear JPG, PNG or WebP image.")
    elif source == "Webcam":
        st.caption("Capture one still frame. Your browser may ask for camera permission.")
        picture = st.camera_input("Capture a face", key="face_camera")
        if picture is not None and st.button("Analyze webcam photo", type="primary", use_container_width=True):
            try:
                with st.spinner("Detecting and scoring the face…"):
                    photo_analysis(picture.getvalue(), picture.name)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("The camera photo could not be analyzed.")
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
    st.markdown("<div class='section-label'>03 / Multimodal studio</div><div class='content-card'><h3>Two cues. One moment.</h3><p>Upload a video with spoken audio and a visible face to compare what each channel suggests.</p></div>", unsafe_allow_html=True)
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
                if voice_result and face_result:
                    if voice_result["emotion"] == face_result["emotion"]:
                        st.info(f"Both models ranked {voice_result['emotion'].title()} first. Agreement between model outputs does not establish a person's actual emotion.")
                    else:
                        st.info(f"The cues differ: voice ranked {voice_result['emotion'].title()} first; face ranked {face_result['emotion'].title()} first. Both results remain visible above.")
                combined = experimental_fusion(voice_result, face_result)
                if combined:
                    st.subheader("Experimental combined cue")
                    show_result(combined, "Combined", "exploratory late fusion")
                    st.caption("Seven shared classes; voice Calm is excluded. The remaining voice scores are renormalized and averaged equally with face scores. This exploratory fusion was evaluated on one held-out actor and is not jointly calibrated or validated for real-world recordings.")
                else:
                    st.info("A combined cue needs both valid voice and face results. Available single-cue results are shown above.")
                input_details(video.name, video.getvalue())
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The video could not be analyzed: {exc}")

with st.expander("Models, sources and limits"):
    st.write("**Voice:** The selected logistic regression uses 130 MFCC, delta-MFCC, energy and spectral summary features from 16 kHz mono audio. On 360 untouched RAVDESS clips from six unseen actors: accuracy **45.83%**, macro F1 **45.12%**. On TESS's other speaker: **37.79% / 33.37%**; on unseen German EmoDB: **31.94% / 30.04%**. These are acted-corpus tests, not real-world accuracy.")
    st.write("**Face:** YuNet detects face landmarks; OpenCV Zoo's Progressive Teacher scores aligned 112×112 crops. On 52 acted clips from one held-out RAVDESS actor, accuracy was **51.92%** and macro F1 **45.00%**. It performs no identity recognition. Photo performance is unmeasured.")
    st.write("[RAVDESS data](https://zenodo.org/records/1188976) · [TESS](https://doi.org/10.5683/SP2/E8H2MF) · [EmoDB](https://zenodo.org/records/7447302) · [OpenCV expression model](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition)")
st.markdown("<div class='footer'>VocalLens · Ahmad Ali Shah, Saboor Khan, Sardar Bilal · Academic demonstration · Outputs are uncertain expression cues.</div>", unsafe_allow_html=True)
