"""Presentation helpers; all scores displayed here come from model outputs."""
from html import escape

import streamlit as st


CSS = """
<style>
:root {color-scheme:dark}
.stApp {background:radial-gradient(ellipse 85% 50% at 82% -15%,#38212a 0%,transparent 70%),radial-gradient(ellipse 70% 30% at 0% 65%,#191d2a 0%,transparent 75%),#090b10;color:#eef0f4;font-family:'DM Sans',sans-serif}
.block-container {max-width:1180px;padding-top:1.6rem;padding-bottom:3rem}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em!important}
h2 {font-size:1.55rem!important;margin-top:1.2rem!important}
p,span,label {font-family:'DM Sans',sans-serif}
.brand-hero {position:relative;overflow:hidden;border:1px solid #383039;border-radius:24px;padding:1.75rem 2.3rem;margin-bottom:1.25rem;background:linear-gradient(108deg,#1b1b25 0%,#18151e 55%,#311d26 100%);box-shadow:0 25px 65px #0005}
.brand-hero:before {content:'';position:absolute;width:360px;height:360px;border-radius:50%;right:-70px;top:-170px;border:1px solid #e46b7245;box-shadow:0 0 0 35px #e46b7209,0 0 0 70px #e46b7207,0 0 0 105px #e46b7205}
.brand-kicker {display:flex;gap:.7rem;align-items:center;color:#eca5aa;font-size:.72rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase}
.brand-dot {width:8px;height:8px;border-radius:100%;background:#e46b72;box-shadow:0 0 18px #e46b72}
.brand-hero h1 {font-size:clamp(2.8rem,6vw,4.2rem);line-height:1.04;margin:.6rem 0 .35rem;color:#fff}
.brand-hero h1 em {font-style:normal;color:#e46b72}
.brand-tagline {color:#f6e2e4;font-size:1.18rem;font-weight:600;margin:0 0 .55rem}
.brand-copy {color:#aeb1bf;max-width:570px;margin:0;font-size:.96rem;line-height:1.65}
.wave {display:flex;gap:4px;align-items:center;height:38px;margin-top:.65rem;opacity:.82}
.wave i {display:block;width:3px;border-radius:4px;background:linear-gradient(#f2a1a7,#a64c5d);height:var(--h);box-shadow:0 0 12px #e46b7235}
.section-label {font-size:.72rem;font-weight:700;letter-spacing:.18em;color:#ad8890;text-transform:uppercase;margin:1.5rem 0 .65rem}
.mode-card {border:1px solid #34333d;background:linear-gradient(140deg,#1b1d27,#15161f);border-radius:18px;padding:1.15rem 1.2rem;min-height:112px;transition:border-color .2s,transform .2s,box-shadow .2s}
.mode-card:hover {transform:translateY(-2px);border-color:#99525d;box-shadow:0 12px 25px #0003}
.mode-card.active {border-color:#c76a74;background:linear-gradient(140deg,#30212c,#1e1b26);box-shadow:inset 0 0 0 1px #e46b722b,0 12px 28px #0003}
.mode-card strong {display:block;color:#f7f4f5;font:600 1.05rem 'Space Grotesk',sans-serif;margin:.4rem 0 .25rem}
.mode-card small {display:block;color:#9fa1b0;font-size:.79rem;line-height:1.4}
.mode-icon {font-size:1.15rem;color:#e78a92}
.content-card {border:1px solid #30313b;border-radius:20px;background:linear-gradient(150deg,#191b24,#12141b);padding:1.35rem 1.5rem;margin:.7rem 0 1.2rem;box-shadow:0 12px 28px #0003}
.content-card h3 {margin:0 0 .35rem;color:#f7f4f6;font-size:1.25rem}
.content-card p {margin:0;color:#9da0ae;font-size:.88rem;line-height:1.55}
.result-card {position:relative;overflow:hidden;border:1px solid var(--emotion-border,#65424b);border-left:4px solid var(--emotion,#e46b72);border-radius:22px;background:linear-gradient(110deg,#24222e,#1c1b25 68%,#161922);padding:1.35rem 1.6rem;margin:1rem 0;box-shadow:0 20px 35px #0003}
.result-card:after {content:'◌';position:absolute;right:1.1rem;top:-1.8rem;font-size:10rem;line-height:1;color:#e46b7215}
.result-kicker {font-size:.69rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:#e7a2a8}
.result-emotion {font:600 clamp(2.2rem,5vw,3.6rem) 'Space Grotesk',sans-serif;color:#fff;line-height:1.1;margin:.5rem 0}
.result-score {color:#eee2e4;font-size:1rem}
.result-score b {color:var(--emotion,#f09299)}
.result-note {color:#a8a8b4;font-size:.82rem;margin-top:.55rem}
.distribution {border:1px solid #30313b;border-radius:19px;background:#15171f;padding:1.3rem 1.5rem;margin:.7rem 0 1.2rem}
.distribution-title {font:600 1rem 'Space Grotesk',sans-serif;color:#f0eff2;margin-bottom:1rem}
.score-row {display:grid;grid-template-columns:92px 1fr 54px;align-items:center;gap:.8rem;margin:.72rem 0;color:#aeb0bb;font-size:.83rem}
.score-row.winner {color:#fff;font-weight:700}
.score-track {height:9px;background:#2c2d37;border-radius:20px;overflow:hidden}
.score-fill {height:100%;border-radius:20px;background:var(--emotion,#737783);opacity:.65}
.winner .score-fill {opacity:1;box-shadow:0 0 13px var(--emotion,#e46b72)}
.input-choice {display:flex;gap:.5rem;align-items:center;color:#aeb1bf;font-size:.82rem;margin:.3rem 0 1rem}
.input-choice b {color:#f0c1c4}
.score-value {text-align:right;font-variant-numeric:tabular-nums}
.micro-note {color:#9396a4;font-size:.8rem;line-height:1.55;margin:.35rem 0 1.25rem}
.stButton>button[kind="primary"] {background:#e46b72!important;color:#1d1116!important;border:1px solid #f09399!important;border-radius:12px!important;font-weight:700!important;min-height:3.1rem;box-shadow:0 8px 22px #e46b7233;transition:transform .2s,box-shadow .2s}
.stButton>button[kind="primary"]:hover {background:#f08b93!important;transform:translateY(-2px);box-shadow:0 12px 25px #e46b7250}
.stButton>button[kind="secondary"] {background:#1e202a;border:1px solid #3d3e49;color:#dedfe7;border-radius:12px}
div[data-testid="stFileUploader"] {border-radius:16px;background:#181a23;padding:.45rem;border:1px solid #393a47}
div[data-testid="stFileUploaderDropzone"] {border:1px dashed #79505b;background:#1e1d28;border-radius:13px}
div[data-testid="stAlert"] {border-radius:13px}
.footer {border-top:1px solid #30303b;margin-top:2.2rem;padding-top:1.1rem;color:#898d9b;font-size:.78rem}
@media(max-width:700px){.brand-hero{padding:1.6rem}.block-container{padding-left:1rem;padding-right:1rem}.score-row{grid-template-columns:75px 1fr 46px;gap:.5rem}.mode-card{min-height:0}}
/* Final studio treatment: local system fonts, brighter depth, accessible controls. */
.stApp {background:radial-gradient(ellipse at 90% 0%,#52283666,transparent 48%),radial-gradient(ellipse at 0% 65%,#243f5360,transparent 55%),#090b10;font-family:'Segoe UI',Arial,sans-serif}
p,span,label {font-family:'Segoe UI',Arial,sans-serif}
h1,h2,h3,.brand-hero h1,.mode-card strong,.result-emotion,.distribution-title {font-family:'Segoe UI',Arial,sans-serif!important}
.brand-hero {background:linear-gradient(115deg,#232532,#211b29 58%,#502936);border-color:#80505e;box-shadow:0 18px 65px #0006,inset 0 1px 0 #ffffff12}
.brand-hero h1 {font-weight:750;letter-spacing:-.065em!important}
.brand-hero:after {content:'VOICE / VISION / EXPRESSION';position:absolute;right:2rem;bottom:1.5rem;color:#eebbc4;font-size:.65rem;letter-spacing:.17em}
.brand-copy,.content-card p,.mode-card small {color:#bdc1cf}
.mode-card {background:linear-gradient(135deg,#252835,#151923);border-color:#434655;min-height:102px;box-shadow:inset 0 1px 0 #ffffff09}
.mode-card.active {background:linear-gradient(125deg,#4b2939,#242333);border-color:#f28b98;box-shadow:0 8px 35px #e46b7220,inset 0 1px 0 #ffc5cc30}
.content-card {background:linear-gradient(125deg,#222633,#171b25);border-color:#414451}
.result-card {background:radial-gradient(ellipse at 100% 0%,#73516a40,transparent 70%),linear-gradient(110deg,#292b3b,#181c28);box-shadow:0 15px 45px #0005,inset 0 1px 0 #ffffff12;animation:result-reveal .3s ease-out}
.result-score b {font-size:1.55rem;font-variant-numeric:tabular-nums}
.distribution {background:linear-gradient(135deg,#1e2330,#141923);border-color:#3b4251}
.score-fill {opacity:.85}.score-track {height:10px;background:#333948}
div[data-testid="stFileUploader"] {background:linear-gradient(130deg,#242532,#191e2b);border-color:#595263;box-shadow:0 10px 30px #0002}
div[data-testid="stFileUploaderDropzone"] {border-color:#b56b7c;background:#2a2432;transition:border-color .2s,background .2s}
div[data-testid="stFileUploaderDropzone"]:hover {border-color:#ff9caa;background:#342735}
button[aria-checked="true"] {box-shadow:0 0 0 1px #ed8495,0 5px 20px #e46b7225!important}
button:focus-visible {outline:2px solid #ffb4bf!important;outline-offset:3px}
.input-choice {padding:.7rem 1rem;border-left:3px solid #e46b72;background:#242332;border-radius:0 10px 10px 0}
div[data-testid="stImage"] img {max-height:360px;object-fit:contain;border-radius:14px}
@keyframes result-reveal {from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;transform:none!important}}
@media(max-width:700px){.brand-hero:after{display:none}}
</style>
"""

EMOTION_COLORS = {"angry": "#e17a7e", "calm": "#86b9c2", "contempt": "#bd9c88",
                  "disgust": "#9dba8d", "fearful": "#b39bd2", "happy": "#e8bd76",
                  "neutral": "#b0b4c1", "sad": "#86a9dd", "surprised": "#d6a2c4"}


def hero():
    heights = [7, 15, 26, 10, 35, 19, 9, 30, 41, 18, 8, 24, 35, 13, 27, 43, 21, 10, 31, 16, 7, 22, 38, 15, 8, 29, 17, 7]
    bars = "".join(f'<i style="--h:{value}px"></i>' for value in heights)
    st.markdown(f"<div class='brand-hero'><div class='brand-kicker'><span class='brand-dot'></span>Multimodal expression intelligence</div><h1>VocalLens<em>.</em></h1><p class='brand-tagline'>Hear the voice. See the emotion.</p><p class='brand-copy'>Explore expression cues in a voice recording, a face, or a synchronized video. Every result is drawn from the analyzed input.</p><div class='wave' aria-hidden='true'>{bars}</div></div>", unsafe_allow_html=True)


def mode_cards(active):
    cards = [("Voice only", "◖))", "Voice analysis", "Listen to speech patterns"),
             ("Face only", "◉", "Facial expression", "Read visible face cues"),
             ("Voice + Face", "◈", "Multimodal video", "Compare sound and sight")]
    st.markdown("<div class='section-label'>Choose your analysis</div>", unsafe_allow_html=True)
    for col, (key, icon, title, description) in zip(st.columns(3), cards):
        with col:
            state = " active" if active == key else ""
            st.markdown(f"<div class='mode-card{state}'><span class='mode-icon'>{icon}</span><strong>{title}</strong><small>{description}</small></div>", unsafe_allow_html=True)


def result_dashboard(result, modality, model):
    label = str(result["emotion"])
    score = float(result["probabilities"][label])
    ordered = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    tone = EMOTION_COLORS.get(label, "#e46b72")
    alternative = ordered[1] if len(ordered) > 1 else None
    secondary = f"Next possible: {escape(str(alternative[0]).title())} {alternative[1]:.1%}. " if alternative else ""
    st.markdown(f"<div class='result-card' style='--emotion:{tone};--emotion-border:{tone}77'><div class='result-kicker'>{escape(modality)} · {escape(model)}</div><div class='result-emotion'>{escape(label.title())}</div><div class='result-score'>Top model score <b>{score:.1%}</b></div><div class='result-note'>{secondary}An expression cue from this input, not a measure of inner feeling.</div></div>", unsafe_allow_html=True)
    rows = []
    for emotion, probability in ordered:
        winning = " winner" if emotion == label else ""
        width = max(0, min(100, probability * 100))
        color = EMOTION_COLORS.get(emotion, "#9397a6")
        rows.append(f"<div class='score-row{winning}' style='--emotion:{color}'><span>{escape(str(emotion).title())}</span><div class='score-track'><div class='score-fill' style='width:{width:.2f}%'></div></div><span class='score-value'>{probability:.1%}</span></div>")
    st.markdown("<div class='distribution'><div class='distribution-title'>Expression distribution</div>" + "".join(rows) + "</div>", unsafe_allow_html=True)
    if score < 0.45:
        st.warning("The top score is low; other expressions remain plausible.")
