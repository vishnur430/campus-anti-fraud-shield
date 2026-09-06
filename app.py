import streamlit as st
from detector import analyze_form_risk, predict_deepfake

# Page Configuration
st.set_page_config(page_title="Campus Anti-Fraud Shield", page_icon="🛡️", layout="wide")
st.title("🛡️ Campus Anti-Fraud Shield")

tab1, tab2 = st.tabs(["📝 Form & Email Risk Analyzer", "🎙️ AI Voice Verifier"])

# Tab 1: Form & Email Risk Analyzer
with tab1:
    st.subheader("Analyze Form URL & Sender Email")
    url_in = st.text_input("Form URL:", placeholder="https://forms.gle/...")
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

# Tab 2: AI Voice Verifier
with tab2:
    st.subheader("Verify Call Recording")
    audio_file = st.file_uploader("Upload recording (.wav or .mp3):", type=["wav", "mp3"])
    
    if audio_file and st.button("Run Voice Analysis"):
        with st.spinner("Analyzing audio remotely..."):
            fake_p, real_p = predict_deepfake(audio_file)
            
            if fake_p > 55.0:
                st.error(f"🚨 AI Cloned / Synthetic Voice Detected ({fake_p:.1f}% confidence)")
            else:
                st.success(f"✅ Genuine Human Voice ({real_p:.1f}% confidence)")
                
            st.progress(int(fake_p) / 100, text=f"Deepfake Probability: {fake_p:.1f}%")