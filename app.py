import streamlit as st
from detector import load_audio_model, analyze_form_risk, predict_deepfake

st.set_page_config(page_title="Campus Anti-Fraud Shield", page_icon="🛡️", layout="wide")
st.title("🛡️ Campus Anti-Fraud Shield")

@st.cache_resource
def get_model():
    return load_audio_model()

feature_extractor, voice_model = get_model()
tab1, tab2 = st.tabs(["📝 Form & Email Risk Analyzer", "🎙️ AI Voice Verifier"])

with tab1:
    st.subheader("Analyze Form URL & Sender Email")
    url_in = st.text_input("Form URL:")
    email_in = st.text_input("Sender Email Address:")
    if st.button("Analyze Offer"):
        status, score, flags = analyze_form_risk(url_in, email_in)
        st.write(f"### Risk Rating: {status} ({score}/100)")
        for flag in flags:
            st.write(f"- ⚠️ {flag}")

with tab2:
    st.subheader("Verify Call Recording")
    audio_file = st.file_uploader("Upload recording (.wav or .mp3):", type=["wav", "mp3"])
    if audio_file and st.button("Run Voice Analysis"):
        fake_p, real_p = predict_deepfake(audio_file, feature_extractor, voice_model)
        if fake_p > 55.0:
            st.error(f"🚨 AI Cloned / Synthetic Voice Detected ({fake_p:.1f}% confidence)")
        else:
            st.success(f"✅ Genuine Human Voice ({real_p:.1f}% confidence)")