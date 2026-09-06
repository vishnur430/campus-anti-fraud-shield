import gc
import streamlit as st
from detector import load_audio_model, analyze_form_risk, predict_deepfake

st.set_page_config(page_title="Campus Anti-Fraud Shield", page_icon="🛡️", layout="wide")
st.title("🛡️ Campus Anti-Fraud Shield")

# Hide spinner to avoid UI freeze delays
@st.cache_resource(show_spinner=False)
def get_model():
    return load_audio_model()

try:
    feature_extractor, voice_model = get_model()
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    st.stop()

tab1, tab2 = st.tabs(["📝 Form & Email Risk Analyzer", "🎙️ AI Voice Verifier"])

with tab1:
    st.subheader("Analyze Form URL & Sender Email")
    url_in = st.text_input("Form URL:", placeholder="https://example-form.com/login")
    email_in = st.text_input("Sender Email Address:", placeholder="scholarship-update@example.xyz")
    
    if st.button("Analyze Offer"):
        if not url_in and not email_in:
            st.warning("Please provide at least a Form URL or a Sender Email address.")
        else:
            status, score, flags = analyze_form_risk(url_in, email_in)
            if status == "HIGH RISK":
                st.error(f"### Risk Rating: {status} ({score}/100)")
            elif status == "MODERATE RISK":
                st.warning(f"### Risk Rating: {status} ({score}/100)")
            else:
                st.success(f"### Risk Rating: {status} ({score}/100)")
                
            for flag in flags:
                st.write(f"- ⚠️ {flag}")

with tab2:
    st.subheader("Verify Call Recording")
    audio_file = st.file_uploader("Upload recording (.wav or .mp3):", type=["wav", "mp3"])
    
    if audio_file and st.button("Run Voice Analysis"):
        with st.spinner("Analyzing..."):
            try:
                fake_p, real_p = predict_deepfake(audio_file, feature_extractor, voice_model)
                if fake_p > 55.0:
                    st.error(f"🚨 AI Cloned / Synthetic Voice Detected ({fake_p:.1f}% confidence)")
                else:
                    st.success(f"✅ Genuine Human Voice ({real_p:.1f}% confidence)")
                st.progress(int(fake_p) / 100, text=f"Deepfake Probability: {fake_p:.1f}%")
            except Exception as err:
                st.error(f"An error occurred: {err}")
            finally:
                gc.collect()