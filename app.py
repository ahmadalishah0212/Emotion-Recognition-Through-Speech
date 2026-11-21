"""Local Streamlit interface for the trained RAVDESS speech model."""
import pandas as pd
import streamlit as st
from src.predict import predict_audio

st.set_page_config(page_title="Vocal Lens | Speech Emotion", page_icon="🎙️", layout="centered")
st.markdown("""<style>
.block-container {max-width: 900px; padding-top: 2rem;}
h1 {letter-spacing: -.04em;} .subtle {color:#64748b;}
.result {background:#0f172a;color:white;padding:1.5rem;border-radius:18px;margin:1rem 0;}
.result small {color:#94a3b8;} .result strong {font-size:2rem;color:#67e8f9;}
</style>""", unsafe_allow_html=True)

st.caption("UNIVERSITY AI PROJECT  /  AHMAD · SABOOR · BILAL")
st.title("Vocal Lens")
st.markdown("<p class='subtle'>Explore the emotion conveyed by a short speech recording.</p>", unsafe_allow_html=True)
st.info("Predictions are approximate. A voice recording cannot reliably reveal a person's actual feelings.")

tab_upload, tab_record = st.tabs(["Upload audio", "Record with microphone"])
with tab_upload:
    uploaded = st.file_uploader("Choose a speech recording", type=["wav", "mp3", "flac", "ogg"], help="WAV is recommended. Use a clear recording of at least 0.4 seconds.")
with tab_record:
    recorded = st.audio_input("Record a short speech sample")

audio = recorded if recorded is not None else uploaded
if recorded is not None and uploaded is not None:
    st.caption("Using the microphone recording. Clear it to use the uploaded file.")

if audio is not None:
    raw = audio.getvalue()
    st.audio(raw)
    if st.button("Analyze speech", type="primary", use_container_width=True):
        if len(raw) > 25 * 1024 * 1024:
            st.error("Audio file is too large. Choose a file under 25 MB.")
        else:
            try:
                with st.spinner("Extracting audio features and running the model..."):
                    result = predict_audio(raw)
                st.markdown(f"<div class='result'><small>PREDICTED EXPRESSION</small><br><strong>{result['emotion'].title()}</strong><br><small>Model: {result['model'].replace('_', ' ').title()}</small></div>", unsafe_allow_html=True)
                if result["probabilities"]:
                    st.subheader("Model score by emotion")
                    chart = pd.DataFrame({"Emotion": list(result["probabilities"]), "Score": list(result["probabilities"].values())}).set_index("Emotion")
                    st.bar_chart(chart, color="#22d3ee")
                    st.caption("Scores are model estimates, not calibrated certainty about a person's feelings.")
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("The recording could not be processed. Try a short, clear WAV file.")
                st.caption(f"Technical detail: {exc}")

with st.expander("How it works"):
    st.write("The app converts audio to 16 kHz mono, extracts MFCC, energy, zero-crossing and spectral-centroid summaries, then passes the 46 values to the selected trained model. The model was selected using macro F1 on speakers excluded from training.")
    st.write("Source dataset: [RAVDESS audio-only speech](https://zenodo.org/records/1188976).")
