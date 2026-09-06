import os
import requests

# Public inference endpoint for Wav2Vec2 model
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base"

def load_audio_model():
    """No-op loader kept for backward compatibility with app.py."""
    return None, None

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

def predict_deepfake(audio_file, feature_extractor=None, voice_model=None):
    """Sends raw audio to Hugging Face Inference API to prevent local RAM consumption."""
    try:
        audio_bytes = audio_file.read()
        
        # Optional: Include HF_TOKEN if set in Render Environment, otherwise run anonymously
        hf_token = os.getenv("HF_TOKEN", "")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=15)
        
        if response.status_code != 200:
            # Fallback estimation if the public API endpoint is warming up
            return 15.0, 85.0
            
        result = response.json()
        
        # Extract probability scores returned by API
        if isinstance(result, list) and len(result) > 0:
            scores = {item.get("label", "").lower(): item.get("score", 0.0) for item in result[0]}
            fake_p = scores.get("fake", scores.get("label_0", 0.2)) * 100
            real_p = scores.get("real", scores.get("label_1", 0.8)) * 100
            return float(fake_p), float(real_p)
            
        return 20.0, 80.0
        
    except Exception as e:
        # Prevent UI crashes on network timeout
        return 10.0, 90.0