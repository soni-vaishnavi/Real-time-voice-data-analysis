"""
pipeline/phase3_analysis/sarcasm_rules.py
==========================================
Rule-based sarcasm detection for Hindi/Hinglish audio.
Replaces cardiffnlp/twitter-roberta-base-irony — saves 400 MB RAM.
"""

import re
from typing import Dict

_SARCASM_POSITIVE = {
    "bilkul", "zaroor", "wah", "shukriya",
    "great", "amazing", "wonderful", "perfect",
    "fantastic", "super", "excellent", "brilliant",
}

_NEGATIVE_CONTEXT = {
    "nahi", "mat", "kyun", "problem", "issue", "galat",
    "bekar", "kharab", "bura", "faltu", "bakwas", "pagal",
    "bore", "boring", "pareshaan", "tang", "thaka",
}

# Minimal override — only words that are never used sarcastically
_GENUINE_DISTRESS_OVERRIDE = {
    "bachao", "khoon", "blood", "goli", "shoot", "suicide",
}

_PATTERNS = [
    r'\b(haan haan|acha acha|theek hai theek hai|bilkul bilkul|wah wah)\b',
    r'\bbilkul\b.{0,30}\btheek\b',
    r'\b(oh great|wow great|super|brilliant)\b.{0,40}\b(problem|issue|kharab|bekar)\b',
    r'\bkya main\b.{0,20}\b(doctor|police|hero|actor|servant|naukar)\b',
    r'\bmaar hi dala\b',
    r'\bbore ho (gaya|gayi|gaye)\b',
    r'\bkhatam kar diya\b.{0,30}\b(assignment|project|presentation|game|kaam)\b',
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def detect_sarcasm(text: str) -> Dict:
    """
    Returns same schema as old RoBERTa detect_sarcasm() — backward compatible.
    method="rules" distinguishes it from the old model-based output.
    """
    if not text or len(text.strip()) < 3:
        return _not_sarcastic()

    text_lower = text.lower()
    for word in _GENUINE_DISTRESS_OVERRIDE:
        if word in text_lower:
            return _not_sarcastic()

    words = set(text_lower.split())
    score = 0.0

    for p in _COMPILED:
        if p.search(text_lower):
            score += 0.40
            break

    if (words & _SARCASM_POSITIVE) and (words & _NEGATIVE_CONTEXT):
        score += 0.30

    if re.search(r'[?!]{2,}', text):
        score += 0.15

    word_list = text_lower.split()
    for i in range(len(word_list) - 1):
        w1 = re.sub(r'[^\w]', '', word_list[i])
        w2 = re.sub(r'[^\w]', '', word_list[i + 1])
        if w1 == w2 and len(w1) >= 2:
            score += 0.20
            break

    score        = min(round(score, 3), 1.0)
    is_sarcastic = score >= 0.40
    confidence   = "high" if score > 0.70 else ("medium" if score >= 0.40 else "low")
    return {"is_sarcastic": is_sarcastic, "sarcasm_score": score,
            "confidence": confidence, "method": "rules"}


def _not_sarcastic() -> Dict:
    return {"is_sarcastic": False, "sarcasm_score": 0.0, "confidence": "low", "method": "rules"}