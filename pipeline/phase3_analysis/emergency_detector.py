"""
Phase 3 - Step 2: Emergency Detector
Uses BART zero-shot classification to detect emergency type in text.

Model: facebook/bart-large-mnli
- Zero-shot = no training needed, works on any labels we define
- We define our own emergency categories as labels
- Model scores how well text matches each label
- ~1.6GB but loads once and stays in RAM

Why zero-shot:
- We can't train on emergency audio (rare real data)
- Zero-shot lets us define "fire emergency", "medical emergency" etc
- as plain English labels — model understands them directly
"""

import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_emergency_pipeline = None

# ── EMERGENCY LABELS ──────────────────────────────────────────────────────────
# These are the zero-shot labels we pass to BART
# Plain English descriptions = better model understanding
EMERGENCY_LABELS = [
    "medical emergency, someone needs a doctor or ambulance",
    "fire emergency, there is a fire or smoke",
    "violence or physical assault, someone is being attacked or hurt",
    "accident or injury, someone has fallen or been injured",
    "theft or robbery, someone is stealing or being robbed",
    "mental health crisis, someone wants to harm themselves",
    "normal conversation, no emergency",
]

# Short names for each label (same order as EMERGENCY_LABELS)
EMERGENCY_SHORT_NAMES = [
    "medical",
    "fire",
    "violence",
    "accident",
    "theft",
    "mental_health",
    "normal",
]

# Emergency threshold — above this = flag as emergency
EMERGENCY_THRESHOLD = 0.35

# Category risk levels for decision engine
CATEGORY_RISK = {
    "medical":      0.90,
    "fire":         0.95,
    "violence":     0.85,
    "accident":     0.80,
    "theft":        0.70,
    "mental_health": 0.85,
    "normal":       0.0,
}


def get_emergency_model():
    """Load BART zero-shot model once and cache"""
    global _emergency_pipeline
    if _emergency_pipeline is None:
        import torch
        from transformers import pipeline
        logger.info("Loading emergency detection model (facebook/bart-large-mnli)...")
        logger.info("First run downloads ~1.6GB — please wait...")
        _emergency_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            framework="pt",
            device=-1       # CPU
        )
        logger.info("Emergency detection model ready ✅")
    return _emergency_pipeline


def detect_emergency(text: str) -> Dict:
    """
    Detect emergency type in text using zero-shot classification.
    
    Returns:
        {
            top_category: "medical",
            top_score: 0.72,
            is_emergency: True,
            all_scores: {medical: 0.72, fire: 0.05, ...},
            risk_level: 0.90
        }
    """
    if not text or len(text.strip()) < 3:
        return _empty_emergency_result()

    try:
        model = get_emergency_model()

        result = model(
            text[:512],
            candidate_labels=EMERGENCY_LABELS,
            multi_label=False       # single best category
        )

        # Map long labels back to short names
        all_scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            idx = EMERGENCY_LABELS.index(label)
            short = EMERGENCY_SHORT_NAMES[idx]
            all_scores[short] = round(score, 4)

        top_category = max(all_scores, key=all_scores.get)
        top_score = all_scores[top_category]

        # Is it actually an emergency?
        is_emergency = (
            top_category != "normal" and
            top_score >= EMERGENCY_THRESHOLD
        )

        risk_level = CATEGORY_RISK.get(top_category, 0.0) if is_emergency else 0.0

        logger.info(
            f"Emergency: {top_category} ({top_score:.2f}) | "
            f"is_emergency={is_emergency} | risk={risk_level}"
        )

        return {
            "top_category": top_category,
            "top_score": top_score,
            "is_emergency": is_emergency,
            "all_scores": all_scores,
            "risk_level": risk_level,
        }

    except Exception as e:
        logger.error(f"Emergency detection failed: {e}")
        return _empty_emergency_result()


def _empty_emergency_result() -> Dict:
    return {
        "top_category": "normal",
        "top_score": 0.0,
        "is_emergency": False,
        "all_scores": {},
        "risk_level": 0.0,
    }


def analyze_emergency_batch(transcripts: List[Dict]) -> List[Dict]:
    """Run emergency detection on all transcripts"""
    logger.info(f"Running emergency detection on {len(transcripts)} transcripts")
    get_emergency_model()

    for t in transcripts:
        text = t.get("text", "")

        # Also boost using existing keyword analysis from Phase 2
        keyword_boost = t.get("keyword_analysis", {}).get("total_boost", 0.0)
        keyword_category = t.get("keyword_analysis", {}).get("top_category", None)

        emergency = detect_emergency(text)

        # If Phase 2 keyword matched a category and model score is borderline
        # give it a nudge — keywords are very precise signals
        if keyword_boost > 0 and keyword_category:
            kw_cat = keyword_category.lower()
            if kw_cat in emergency["all_scores"]:
                boosted = min(emergency["all_scores"][kw_cat] + keyword_boost, 1.0)
                emergency["all_scores"][kw_cat] = round(boosted, 4)
                # Re-evaluate top category after boost
                new_top = max(emergency["all_scores"], key=emergency["all_scores"].get)
                emergency["top_category"] = new_top
                emergency["top_score"] = emergency["all_scores"][new_top]
                emergency["is_emergency"] = (
                    new_top != "normal" and
                    emergency["top_score"] >= EMERGENCY_THRESHOLD
                )
                logger.info(
                    f"{t.get('chunk_id')} | Keyword boost applied: "
                    f"{kw_cat} +{keyword_boost} → {emergency['top_score']:.2f}"
                )

        t["emergency_analysis"] = emergency

    logger.info("Emergency detection complete ✅")
    return transcripts
