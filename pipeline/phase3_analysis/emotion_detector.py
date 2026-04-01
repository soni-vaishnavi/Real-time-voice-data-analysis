"""
Phase 3 - Step 1: Emotion Detector
Detects emotion in transcript text using HuggingFace model.
Also runs sarcasm detection to avoid false positives.

Model: j-hartmann/emotion-english-distilroberta-base
- 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
- Lightweight ~500MB
- Works on English and Roman Hinglish
"""

import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── MODEL CACHE ───────────────────────────────────────────────────────────────
_emotion_pipeline = None
_sarcasm_pipeline = None


def get_emotion_model():
    """Load emotion model once and cache"""
    global _emotion_pipeline
    if _emotion_pipeline is None:
        import torch
        from transformers import pipeline
        logger.info("Loading emotion model (j-hartmann/emotion-english-distilroberta-base)...")
        logger.info("First run downloads ~500MB — please wait...")
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            truncation=True,
            max_length=512,
            framework="pt",       # explicitly use PyTorch
            device=-1             # CPU
        )
        logger.info("Emotion model ready ✅")
    return _emotion_pipeline


def get_sarcasm_model():
    """Load sarcasm/irony model once and cache"""
    global _sarcasm_pipeline
    if _sarcasm_pipeline is None:
        import torch
        from transformers import pipeline
        logger.info("Loading sarcasm/irony model (cardiffnlp/twitter-roberta-base-irony)...")
        _sarcasm_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony",
            truncation=True,
            max_length=512,
            framework="pt",
            device=-1
        )
        logger.info("Sarcasm model ready ✅")
    return _sarcasm_pipeline


# ── EMOTION DETECTION ─────────────────────────────────────────────────────────

# Map model emotion labels to our standard names
EMOTION_LABEL_MAP = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
    # Some model versions use these labels
    "LABEL_0":  "anger",
    "LABEL_1":  "disgust",
    "LABEL_2":  "fear",
    "LABEL_3":  "joy",
    "LABEL_4":  "neutral",
    "LABEL_5":  "sadness",
    "LABEL_6":  "surprise",
}

# Emergency-relevant emotions and their weight multipliers
# Higher = more likely to indicate real emergency
EMOTION_EMERGENCY_WEIGHT = {
    "fear":     1.0,    # highest emergency signal
    "anger":    0.7,    # fight/conflict signal
    "sadness":  0.5,    # distress signal
    "surprise": 0.3,    # shock signal
    "disgust":  0.2,    # low emergency signal
    "joy":      0.0,    # not emergency
    "neutral":  0.0,    # not emergency
}


def detect_emotion(text: str) -> Dict:
    """
    Detect emotion in text.
    Returns all emotion scores + dominant emotion.
    
    Args:
        text: transcript text (English or Roman Hinglish)
    
    Returns:
        {
            dominant_emotion: "fear",
            dominant_score: 0.91,
            all_scores: {fear: 0.91, anger: 0.05, ...},
            emergency_weight: 1.0
        }
    """
    if not text or len(text.strip()) < 3:
        return _empty_emotion_result()

    try:
        model = get_emotion_model()
        results = model(text[:512])  # truncate to model max

        # results is list of lists: [[{label, score}, ...]]
        if isinstance(results[0], list):
            scores_raw = results[0]
        else:
            scores_raw = results

        # Normalize labels
        all_scores = {}
        for item in scores_raw:
            label = EMOTION_LABEL_MAP.get(item["label"].lower(), item["label"].lower())
            all_scores[label] = round(item["score"], 4)

        # Find dominant emotion
        dominant = max(all_scores, key=all_scores.get)
        dominant_score = all_scores[dominant]
        emergency_weight = EMOTION_EMERGENCY_WEIGHT.get(dominant, 0.0)

        return {
            "dominant_emotion": dominant,
            "dominant_score": dominant_score,
            "all_scores": all_scores,
            "emergency_weight": emergency_weight,
            "fear_score": all_scores.get("fear", 0.0),
            "anger_score": all_scores.get("anger", 0.0),
        }

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return _empty_emotion_result()


def _empty_emotion_result() -> Dict:
    return {
        "dominant_emotion": "neutral",
        "dominant_score": 0.0,
        "all_scores": {},
        "emergency_weight": 0.0,
        "fear_score": 0.0,
        "anger_score": 0.0,
    }


# ── SARCASM DETECTION ─────────────────────────────────────────────────────────

def detect_sarcasm(text: str) -> Dict:
    """
    Detect if text is sarcastic/ironic.
    Model: cardiffnlp/twitter-roberta-base-irony
    Labels: non_irony (LABEL_0) / irony (LABEL_1)
    """
    if not text or len(text.strip()) < 3:
        return {"is_sarcastic": False, "sarcasm_score": 0.0, "confidence": "low"}

    try:
        model = get_sarcasm_model()
        result = model(text[:512])

        if isinstance(result, list):
            result = result[0]

        label = result["label"].lower()
        score = result["score"]

        # cardiffnlp model: LABEL_0 = non_irony, LABEL_1 = irony
        is_sarcastic = label in ["irony", "label_1"]

        # Normalize score to sarcasm probability
        sarcasm_score = score if is_sarcastic else 1 - score

        if sarcasm_score > 0.75:
            confidence = "high"
        elif sarcasm_score > 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "is_sarcastic": is_sarcastic,
            "sarcasm_score": round(sarcasm_score, 4),
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Sarcasm detection failed: {e}")
        return {"is_sarcastic": False, "sarcasm_score": 0.0, "confidence": "low"}


# ── SARCASM-EMOTION CONFLICT RESOLUTION ──────────────────────────────────────

def resolve_sarcasm_conflict(emotion_result: Dict, sarcasm_result: Dict) -> Dict:
    """
    Apply Phase 3 sarcasm-emotion conflict resolution matrix.
    
    Rules (from our planning phase):
    - Fear > 85%  → REAL regardless of sarcasm score
    - Fear 60-85% + Sarcasm < 60% → REAL
    - Fear 60-85% + Sarcasm > 60% → UNCERTAIN (check context)
    - Fear < 60%  + Sarcasm > 60% → SARCASM (reduce score by 70%)
    - Anger > 80% + Sarcasm > 70% → REAL (passive aggressive)
    
    Returns:
        {
            sarcasm_override: bool (True = sarcasm ignored, treat as real)
            score_penalty: float (0.0 to 0.7, applied to emergency score)
            resolution: "real"/"sarcasm"/"uncertain"
        }
    """
    fear = emotion_result.get("fear_score", 0.0)
    anger = emotion_result.get("anger_score", 0.0)
    sarcasm = sarcasm_result.get("sarcasm_score", 0.0)
    is_sarcastic = sarcasm_result.get("is_sarcastic", False)

    # Rule 1: Extreme fear — sarcasm irrelevant
    if fear > 0.85:
        return {
            "sarcasm_override": True,
            "score_penalty": 0.0,
            "resolution": "real",
            "reason": f"Fear {fear:.2f} > 0.85 — sarcasm ignored"
        }

    # Rule 2: High fear, low sarcasm — treat as real
    if fear > 0.60 and sarcasm < 0.60:
        return {
            "sarcasm_override": True,
            "score_penalty": 0.0,
            "resolution": "real",
            "reason": f"Fear {fear:.2f} high, sarcasm {sarcasm:.2f} low"
        }

    # Rule 3: Passive aggressive anger
    if anger > 0.80 and is_sarcastic and sarcasm > 0.70:
        return {
            "sarcasm_override": True,
            "score_penalty": 0.0,
            "resolution": "real",
            "reason": f"Angry sarcasm — potential aggression"
        }

    # Rule 4: Uncertain — medium fear + medium sarcasm
    if fear > 0.60 and sarcasm > 0.60:
        return {
            "sarcasm_override": False,
            "score_penalty": 0.30,
            "resolution": "uncertain",
            "reason": f"Mixed signals — fear:{fear:.2f} sarcasm:{sarcasm:.2f}"
        }

    # Rule 5: Low fear + high sarcasm — casual expression
    if fear < 0.60 and is_sarcastic and sarcasm > 0.60:
        return {
            "sarcasm_override": False,
            "score_penalty": 0.70,
            "resolution": "sarcasm",
            "reason": f"Sarcasm confirmed — reducing emergency score"
        }

    # Default — no sarcasm concern
    return {
        "sarcasm_override": True,
        "score_penalty": 0.0,
        "resolution": "real",
        "reason": "No significant sarcasm detected"
    }


# ── FULL EMOTION ANALYSIS ─────────────────────────────────────────────────────

def analyze_emotion(transcript: Dict) -> Dict:
    """
    Run full emotion + sarcasm analysis on one transcript.
    
    Args:
        transcript: dict from Phase 2 (needs 'text' field)
    
    Returns:
        emotion_analysis dict with all results
    """
    text = transcript.get("text", "")
    chunk_id = transcript.get("chunk_id", "?")

    if not text:
        return {
            "emotion": _empty_emotion_result(),
            "sarcasm": {"is_sarcastic": False, "sarcasm_score": 0.0},
            "conflict_resolution": {"resolution": "real", "score_penalty": 0.0},
        }

    # Run both models
    emotion = detect_emotion(text)
    sarcasm = detect_sarcasm(text)

    # Resolve conflict
    resolution = resolve_sarcasm_conflict(emotion, sarcasm)

    logger.info(
        f"{chunk_id} | emotion={emotion['dominant_emotion']} "
        f"({emotion['dominant_score']:.2f}) | "
        f"sarcasm={sarcasm['is_sarcastic']} ({sarcasm['sarcasm_score']:.2f}) | "
        f"→ {resolution['resolution']}"
    )

    return {
        "emotion": emotion,
        "sarcasm": sarcasm,
        "conflict_resolution": resolution,
    }


def analyze_emotion_batch(transcripts: List[Dict]) -> List[Dict]:
    """Run emotion analysis on all transcripts"""
    logger.info(f"Running emotion analysis on {len(transcripts)} transcripts")

    # Pre-load models once
    get_emotion_model()
    get_sarcasm_model()

    results = []
    for t in transcripts:
        analysis = analyze_emotion(t)
        t["emotion_analysis"] = analysis
        results.append(t)

    logger.info("Emotion analysis complete ✅")
    return results
