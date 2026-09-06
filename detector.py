import torch
import librosa
from urllib.parse import urlparse
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

def load_audio_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return feature_extractor, model

def analyze_form_risk(form_url, sender_email=""):
    score = 0
    flags = []
    domain = urlparse(form_url).netloc.lower()

    free_hosts = ["docs.google.com", "forms.office.com", "typeform.com", "forms.gle"]
    if any(host in domain for host in free_hosts):
        score += 35
        flags.append("Form uses a free hosting service (e.g., Google/Microsoft Forms) rather than an official company portal.")

    if sender_email:
        email_domain = sender_email.split("@")[-1].lower()
        free_emails = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
        if email_domain in free_emails:
            score += 45
            flags.append(f"Sender uses a personal email address (@{email_domain}) instead of an official corporate domain.")

    status = "HIGH RISK" if score >= 50 else ("MEDIUM RISK" if score >= 30 else "LOW RISK")
    return status, score, flags

def predict_deepfake(audio_file, feature_extractor, model):
    audio_data, _ = librosa.load(audio_file, sr=16000, mono=True)
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    real_score = probs[0][0].item() * 100
    fake_score = probs[0][1].item() * 100
    return fake_score, real_score