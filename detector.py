import gc
import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Force single-threaded execution to prevent CPU context-switching lag on Render
torch.set_num_threads(1)
DEVICE = torch.device("cpu")
MODEL_NAME = "facebook/wav2vec2-base"

def load_audio_model():
    """Fast model loader using safetensors and low memory overhead."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    
    # Fast load parameters
    voice_model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    voice_model.to(DEVICE)
    voice_model.eval()
    
    gc.collect()
    return feature_extractor, voice_model

def analyze_form_risk(url: str, email: str):
    """Heuristic risk analysis for form URLs and emails."""
    flags = []
    score = 0
    
    suspicious_tlds = [".xyz", ".top", ".club", ".info", ".online", ".site"]
    suspicious_keywords = ["free", "reward", "urgent", "claim", "verify", "update", "login"]
    
    url_lower = url.lower().strip()
    email_lower = email.lower().strip()
    
    if any(url_lower.endswith(tld) or tld + "/" in url_lower for tld in suspicious_tlds):
        flags.append("Suspicious domain extension (TLD) detected in form URL.")
        score += 35
        
    if any(kw in url_lower for kw in suspicious_keywords):
        flags.append("Phishing / urgency keywords found in form URL.")
        score += 25
        
    if any(kw in email_lower for kw in suspicious_keywords):
        flags.append("Suspicious keywords detected in sender email.")
        score += 20
        
    if "@" in email_lower:
        domain = email_lower.split("@")[-1]
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            flags.append("Sender email domain uses a high-risk TLD.")
            score += 20
            
    if score >= 60:
        status = "HIGH RISK"
    elif score >= 30:
        status = "MODERATE RISK"
    else:
        status = "LOW RISK"
        if not flags:
            flags.append("No immediate structural risk flags detected.")
            
    return status, min(score, 100), flags

def predict_deepfake(audio_file, feature_extractor, voice_model):
    """Sub-2-second audio inference pipeline."""
    try:
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert stereo to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Hard limit to 5 seconds max (sufficient for classification, 2x faster execution)
        max_samples = 5 * sample_rate
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
            
        inputs = feature_extractor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=False
        )
        
        with torch.no_grad():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits = voice_model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
        del inputs, logits
        gc.collect()
        
        fake_p = float(probabilities[0] * 100)
        real_p = float(probabilities[1] * 100)
        
        return fake_p, real_p
        
    except Exception as e:
        gc.collect()
        raise RuntimeError(f"Processing error: {str(e)}")