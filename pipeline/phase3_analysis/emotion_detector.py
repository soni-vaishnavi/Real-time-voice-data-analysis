"""
pipeline/phase3_analysis/emotion_detector.py
=============================================
Phase 3 - Text Emotion Detector.

STAGE 0 CHANGES:
  - Sarcasm model (cardiffnlp/twitter-roberta-base-irony) removed — saves 400 MB
  - detect_sarcasm() now imported from sarcasm_rules.py (rule-based, 0 MB)
  - get_sarcasm_model() and _sarcasm_pipeline global removed
"""

import logging
from typing import Dict, List

from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_emotion_pipeline = None


def get_emotion_model():
    """Load emotion model once and cache."""
    global _emotion_pipeline
    if _emotion_pipeline is None:
        from transformers import pipeline
        logger.info("Loading emotion model (j-hartmann/emotion-english-distilroberta-base)...")
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None, truncation=True, max_length=512,
            framework="pt", device=-1,
        )
        logger.info("Emotion model ready ✅")
    return _emotion_pipeline


EMOTION_LABEL_MAP = {
    "anger":"anger","disgust":"disgust","fear":"fear","joy":"joy",
    "neutral":"neutral","sadness":"sadness","surprise":"surprise",
    "LABEL_0":"anger","LABEL_1":"disgust","LABEL_2":"fear","LABEL_3":"joy",
    "LABEL_4":"neutral","LABEL_5":"sadness","LABEL_6":"surprise",
}

EMOTION_EMERGENCY_WEIGHT = {
    "fear":1.0, "anger":0.7, "sadness":0.5, "surprise":0.3,
    "disgust":0.2, "joy":0.0, "neutral":0.0,
}


def detect_emotion(text: str) -> Dict:
    """Detect emotion in transcript text. Returns 7-class distribution."""
    if not text or len(text.strip()) < 3:
        return _empty_emotion()
    try:
        model   = get_emotion_model()
        results = model(text[:512])
        raw     = results[0] if isinstance(results[0], list) else results
        all_sc  = {EMOTION_LABEL_MAP.get(r["label"].lower(), r["label"].lower()): round(r["score"], 4)
                   for r in raw}
        dominant      = max(all_sc, key=all_sc.get)
        return {
            "dominant_emotion": dominant,
            "dominant_score":   all_sc[dominant],
            "all_scores":       all_sc,
            "emergency_weight": EMOTION_EMERGENCY_WEIGHT.get(dominant, 0.0),
            "fear_score":       all_sc.get("fear",  0.0),
            "anger_score":      all_sc.get("anger", 0.0),
        }
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return _empty_emotion()


def _empty_emotion() -> Dict:
    return {"dominant_emotion":"neutral","dominant_score":0.0,"all_scores":{},
            "emergency_weight":0.0,"fear_score":0.0,"anger_score":0.0}


def resolve_sarcasm_conflict(emotion_result: Dict, sarcasm_result: Dict) -> Dict:
    """Text-only sarcasm-emotion conflict resolution."""
    fear    = emotion_result.get("fear_score",  0.0)
    anger   = emotion_result.get("anger_score", 0.0)
    sarcasm = sarcasm_result.get("sarcasm_score", 0.0)
    is_sarc = sarcasm_result.get("is_sarcastic", False)

    if fear > 0.85:
        return {"sarcasm_override":True,"score_penalty":0.0,"resolution":"real",
                "reason":f"Fear {fear:.2f} > 0.85"}
    if fear > 0.60 and sarcasm < 0.60:
        return {"sarcasm_override":True,"score_penalty":0.0,"resolution":"real",
                "reason":f"High fear {fear:.2f}, low sarcasm {sarcasm:.2f}"}
    if anger > 0.80 and is_sarc and sarcasm > 0.70:
        return {"sarcasm_override":True,"score_penalty":0.0,"resolution":"real",
                "reason":"Angry sarcasm — possible passive aggression"}
    if fear > 0.60 and sarcasm > 0.60:
        return {"sarcasm_override":False,"score_penalty":0.30,"resolution":"uncertain",
                "reason":f"Mixed signals fear:{fear:.2f} sarcasm:{sarcasm:.2f}"}
    if fear < 0.60 and is_sarc and sarcasm > 0.60:
        return {"sarcasm_override":False,"score_penalty":0.70,"resolution":"sarcasm",
                "reason":f"Sarcasm confirmed (score={sarcasm:.2f})"}
    return {"sarcasm_override":True,"score_penalty":0.0,"resolution":"real",
            "reason":"No sarcasm signal"}


def analyze_emotion(transcript: Dict) -> Dict:
    """Run emotion + sarcasm on one transcript dict."""
    text = transcript.get("text", "")
    if not text:
        return {"emotion":_empty_emotion(),
                "sarcasm":{"is_sarcastic":False,"sarcasm_score":0.0,"method":"rules"},
                "conflict_resolution":{"resolution":"real","score_penalty":0.0}}
    emotion    = detect_emotion(text)
    sarcasm    = detect_sarcasm(text)
    resolution = resolve_sarcasm_conflict(emotion, sarcasm)
    logger.info(f"{transcript.get('chunk_id','?')} | {emotion['dominant_emotion']} "
                f"({emotion['dominant_score']:.2f}) | sarcasm={sarcasm['is_sarcastic']} | "
                f"→ {resolution['resolution']}")
    return {"emotion": emotion, "sarcasm": sarcasm, "conflict_resolution": resolution}


def analyze_emotion_batch(transcripts: List[Dict]) -> List[Dict]:
    """Batch emotion analysis."""
    logger.info(f"Running emotion analysis on {len(transcripts)} transcripts")
    get_emotion_model()
    for t in transcripts:
        t["emotion_analysis"] = analyze_emotion(t)
    logger.info("Emotion analysis complete ✅")
    return transcripts