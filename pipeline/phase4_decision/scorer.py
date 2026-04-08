"""
pipeline/phase4_decision/scorer.py
====================================
Phase 4 - Scorer. STAGE 0: weights imported from pipeline.core.config.

Formula:
    score = (emotion_component * WEIGHT_EMOTION) +
            (emergency_component * WEIGHT_EMERGENCY) +
            (keyword_component * WEIGHT_KEYWORD)
            - sarcasm_penalty
"""

import logging
from typing import Dict

from pipeline.core.config import WEIGHT_EMOTION, WEIGHT_EMERGENCY, WEIGHT_KEYWORD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_score(transcript: Dict) -> Dict:
    """
    Compute final combined emergency score for one transcript chunk.
    Reads emotion_analysis{}, emergency_analysis{}, keyword_analysis{}.
    Returns score dict WITHOUT zone (zone_classifier adds that).
    """
    emotion_data   = transcript.get("emotion_analysis", {})
    emergency_data = transcript.get("emergency_analysis", {})
    keyword_data   = transcript.get("keyword_analysis", {})

    fused = emotion_data.get("fused_emotion") or emotion_data.get("emotion", {})

    emotion_component   = fused.get("emergency_weight", 0.0) * fused.get("dominant_score", 0.0)
    emergency_score     = emergency_data.get("top_score", 0.0)
    is_emergency        = emergency_data.get("is_emergency", False)
    emergency_component = emergency_score if is_emergency else emergency_score * 0.3
    keyword_component   = min(keyword_data.get("total_boost", 0.0), 1.0)

    # Sarcasm penalty — check both key names (sarcasm_resolution from updated analyzer,
    # conflict_resolution from legacy analyzer)
    sarcasm_penalty_rate = (
        emotion_data.get("sarcasm_resolution", {}).get("score_penalty", 0.0)
        or emotion_data.get("conflict_resolution", {}).get("score_penalty", 0.0)
    )

    base_score        = ((emotion_component * WEIGHT_EMOTION) +
                         (emergency_component * WEIGHT_EMERGENCY) +
                         (keyword_component * WEIGHT_KEYWORD))
    sarcasm_deduction = base_score * sarcasm_penalty_rate
    final_score       = round(max(0.0, min(base_score - sarcasm_deduction, 1.0)), 4)

    logger.debug(
        f"Score | emo={emotion_component:.3f}x{WEIGHT_EMOTION} | "
        f"emrg={emergency_component:.3f}x{WEIGHT_EMERGENCY} | "
        f"kw={keyword_component:.3f}x{WEIGHT_KEYWORD} | "
        f"sarc=-{sarcasm_deduction:.3f} | final={final_score:.4f}"
    )

    return {
        "final_score": final_score,
        "components": {
            "emotion_component":   round(emotion_component   * WEIGHT_EMOTION,   4),
            "emergency_component": round(emergency_component * WEIGHT_EMERGENCY, 4),
            "keyword_component":   round(keyword_component   * WEIGHT_KEYWORD,   4),
            "sarcasm_deduction":   round(sarcasm_deduction,                      4),
        },
        "dominant_emotion":   fused.get("dominant_emotion", "neutral"),
        "emergency_category": emergency_data.get("top_category", "normal"),
        "is_emergency":       is_emergency,
    }


def score_all_chunks(transcripts: list) -> list:
    logger.info(f"Scoring {len(transcripts)} chunks...")
    for t in transcripts:
        t["score"] = compute_score(t)
    logger.info("Scoring complete ✅")
    return transcripts