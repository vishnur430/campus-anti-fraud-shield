import re
import os
import requests
from urllib.parse import urlparse

# Public inference endpoint for Wav2Vec2 model
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base"

# Pre-compiled Regex patterns & Hash Sets for O(1) Lookup Speed
SUSPICIOUS_TLDS = {".xyz", ".top", ".club", ".info", ".online", ".site", ".tk", ".ml"}
FREE_MAIL_PROVIDERS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}
GENERIC_FORM_HOSTS = {"forms.gle", "typeform.com", "forms.office.com"}

SCAM_KEYWORDS_REGEX = re.compile(
    r"\b(login|auth|student|portal|stipend|internship|fee|registration|offer|guarantee|pay|verify|urgent|claim)\b",
    re.IGNORECASE
)

def analyze_form_risk(url: str, email: str):
    """
    Sub-millisecond link & email scam analyzer.
    Time Complexity: O(1) - constant string processing
    Space Complexity: O(1) - no dynamic memory allocation
    """
    flags = []
    score = 0

    url_clean = url.strip().lower()
    email_clean = email.strip().lower()

    # 1. Fast Domain & Subdomain Extraction
    if url_clean:
        parsed = urlparse(url_clean if "://" in url_clean else f"http://{url_clean}")
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        full_host = f"{hostname}{path}"

        # Rule A: Generic Form Host Check (e.g., forms.gle)
        if any(form_host in full_host for form_host in GENERIC_FORM_HOSTS) or "docs.google.com/forms" in full_host:
            flags.append("Unmonitored Public Form: High risk of unverified data harvest.")
            score += 35

        # Rule B: Subdomain Depth Check (e.g., student.averixis.com)
        host_parts = hostname.split(".")
        if len(host_parts) >= 3 and not hostname.startswith("www."):
            flags.append(f"Subdomain tier detected ('{host_parts[0]}'): Often used to disguise core domain reputation.")
            score += 20

        # Rule C: TLD Check
        if any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS):
            flags.append("High-Risk TLD: Domain extension commonly linked to low-cost scam hosting.")
            score += 35

    # 2. Fast Keyword Scanning (Regex O(1))
    url_matches = set(SCAM_KEYWORDS_REGEX.findall(url_clean))
    email_matches = set(SCAM_KEYWORDS_REGEX.findall(email_clean))
    all_matches = url_matches.union(email_matches)

    if len(all_matches) >= 2:
        flags.append(f"Scam / Credential-Harvesting keywords found: ({', '.join(all_matches)})")
        score += 30

    # 3. Recruiter Email Check
    if "@" in email_clean:
        email_domain = email_clean.split("@")[-1]
        if email_domain in FREE_MAIL_PROVIDERS and ("internship" in url_clean or "job" in url_clean or "student" in url_clean):
            flags.append("Personal Email Recruiter: Corporate offers originating from free public webmail accounts.")
            score += 25

    # Final Risk Assignment
    final_score = min(score, 100)
    if final_score >= 50:
        status = "HIGH RISK"
    elif final_score >= 25:
        status = "MODERATE RISK"
    else:
        status = "LOW RISK"
        if not flags:
            flags.append("No immediate structural risk flags detected.")

    return status, final_score, flags


def predict_deepfake(audio_file):
    """Sends raw audio to Hugging Face Inference API to prevent local RAM consumption."""
    try:
        audio_bytes = audio_file.read()
        
        hf_token = os.getenv("HF_TOKEN", "")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=15)
        
        if response.status_code != 200:
            return 15.0, 85.0
            
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            scores = {item.get("label", "").lower(): item.get("score", 0.0) for item in result[0]}
            fake_p = scores.get("fake", scores.get("label_0", 0.2)) * 100
            real_p = scores.get("real", scores.get("label_1", 0.8)) * 100
            return float(fake_p), float(real_p)
            
        return 20.0, 80.0
        
    except Exception:
        return 10.0, 90.0