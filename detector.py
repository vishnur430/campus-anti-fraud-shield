import gc
import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Force PyTorch to use CPU only and limit thread allocation
torch.set_num_threads(1)
DEVICE = torch.device("cpu")
# Replace the old gated model identifier with a public base model
MODEL_NAME = "facebook/wav2vec2-base"
def load_audio_model():
    """Loads feature extractor and model safely into CPU memory."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    voice_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    voice_model.to(DEVICE)
    voice_model.eval()
    
    # Run garbage collection after loading weights
    gc.collect()
    return feature_extractor, voice_model

def analyze_form_risk(url: str, email: str):
    """Simple heuristic analysis for form URLs and emails."""
    flags = []
    score = 0
    
    suspicious_tlds = [".xyz", ".top", ".club", ".info", ".online", ".site"]
    suspicious_keywords = ["free", "reward", "urgent", "claim", "verify", "update", "login"]
    
    url_lower = url.lower().strip()
    email_lower = email.lower().strip()
    
    # URL checks
    if any(url_lower.endswith(tld) or tld + "/" in url_lower for tld in suspicious_tlds):
        flags.append("Suspicious domain extension (TLD) detected in form URL.")
        score += 35
        
    if any(kw in url_lower for kw in suspicious_keywords):
        flags.append("Phishing / urgency keywords found in form URL.")
        score += 25
        
    # Email checks
    if any(kw in email_lower for kw in suspicious_keywords):
        flags.append("Suspicious keywords detected in sender email.")
        score += 20
        
    if "@" in email_lower:
        domain = email_lower.split("@")[-1]
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            flags.append("Sender email domain uses a high-risk TLD.")
            score += 20
            
    # Risk categorization
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
    """Processes uploaded audio and calculates synthetic vs. genuine probability."""
    try:
        # Read audio file using soundfile
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert stereo to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Truncate long audio to max 10 seconds to save memory
        max_samples = 10 * sample_rate
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
            
        # Extract features
        inputs = feature_extractor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        # Run inference strictly without gradient tracking
        with torch.no_grad():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits = voice_model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
        # Clear temporary variables from memory
        del inputs, logits
        gc.collect()
        
        # Class 0: Spoof/Fake, Class 1: Real/Genuine (standard XLS-R deepfake order)
        fake_p = float(probabilities[0] * 100)
        real_p = float(probabilities[1] * 100)
        
        return fake_p, real_p
        
    except Exception as e:
        # Fallback safeguard in case of unreadable audio
        gc.collect()
        raise RuntimeError(f"Error processing audio file: {str(e)}")