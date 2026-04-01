"""
Phase 4 - Scorer
Combines emotion, emergency, and keyword scores into one final score.

Formula:
    score = (emotion_component × 0.35) + (emergency_component × 0.40) + (keyword_component × 0.25)
            - sarcasm_penalty

This is the ONLY place the scoring formula lives.
All weights are configurable here.
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── SCORE WEIGHTS ──────────────────────────────────────────────────────────────
# Must sum to 1.0
WEIGHT_EMOTION    = 0.35
WEIGHT_EMERGENCY  = 0.40
WEIGHT_KEYWORD    = 0.25


def compute_score(transcript: Dict) -> Dict:
    """
    Compute final combined emergency score for one transcript chunk.

    Reads from:
        transcript["emotion_analysis"]   ← from Phase 3 emotion_detector
        transcript["emergency_analysis"] ← from Phase 3 emergency_detector
        transcript["keyword_analysis"]   ← from Phase 2 keyword_normalizer

    Returns score dict (does NOT include zone — that's zone_classifier's job):
        {
            "final_score": 0.312,
            "components": {
                "emotion_component":   0.120,
                "emergency_component": 0.156,
                "keyword_component":   0.050,
                "sarcasm_deduction":   0.014,
            },
            "dominant_emotion":   "fear",
            "emergency_category": "medical",
            "is_emergency":       True,
        }
    """
    emotion_data   = transcript.get("emotion_analysis", {})
    emergency_data = transcript.get("emergency_analysis", {})
    keyword_data   = transcript.get("keyword_analysis", {})

    # ── Emotion component ──────────────────────────────────────────────────────
    # emotion_weight = how dangerous this emotion is (fear=1.0, joy=0.0 etc.)
    # dominant_score = how strongly the model detected that emotion
    emotion_weight    = emotion_data.get("emotion", {}).get("emergency_weight", 0.0)
    dominant_score    = emotion_data.get("emotion", {}).get("dominant_score", 0.0)
    emotion_component = emotion_weight * dominant_score   # 0.0 → 1.0

    # ── Emergency component ────────────────────────────────────────────────────
    # Full score if is_emergency=True, heavily discounted if not
    emergency_score     = emergency_data.get("top_score", 0.0)
    is_emergency        = emergency_data.get("is_emergency", False)
    emergency_component = emergency_score if is_emergency else emergency_score * 0.3

    # ── Keyword component ──────────────────────────────────────────────────────
    # keyword_boost comes from Phase 2 — direct Hindi/English emergency keyword matches
    # Cap at 1.0 to prevent over-boosting
    keyword_boost     = keyword_data.get("total_boost", 0.0)
    keyword_component = min(keyword_boost, 1.0)

    # ── Sarcasm penalty ────────────────────────────────────────────────────────
    # score_penalty is a fraction (e.g. 0.5 = reduce score by 50%)
    # Set by emotion_fusion when sarcasm detected but not overridden by audio fear
    sarcasm_penalty_rate = emotion_data.get(
        "conflict_resolution", {}
    ).get("score_penalty", 0.0)

    # ── Base score ─────────────────────────────────────────────────────────────
    base_score = (
        (emotion_component   * WEIGHT_EMOTION)   +
        (emergency_component * WEIGHT_EMERGENCY) +
        (keyword_component   * WEIGHT_KEYWORD)
    )

    # Apply sarcasm penalty as a percentage reduction
    sarcasm_deduction = base_score * sarcasm_penalty_rate
    final_score = max(0.0, base_score - sarcasm_deduction)
    final_score = round(min(final_score, 1.0), 4)

    logger.debug(
        f"Score | emotion={emotion_component:.3f}×{WEIGHT_EMOTION} | "
        f"emergency={emergency_component:.3f}×{WEIGHT_EMERGENCY} | "
        f"keyword={keyword_component:.3f}×{WEIGHT_KEYWORD} | "
        f"sarcasm_deduction={sarcasm_deduction:.3f} | "
        f"final={final_score:.4f}"
    )

    return {
        "final_score": final_score,
        "components": {
            "emotion_component":   round(emotion_component   * WEIGHT_EMOTION,   4),
            "emergency_component": round(emergency_component * WEIGHT_EMERGENCY, 4),
            "keyword_component":   round(keyword_component   * WEIGHT_KEYWORD,   4),
            "sarcasm_deduction":   round(sarcasm_deduction,                      4),
        },
        "dominant_emotion":   emotion_data.get("emotion", {}).get("dominant_emotion", "neutral"),
        "emergency_category": emergency_data.get("top_category", "normal"),
        "is_emergency":       is_emergency,
    }


def score_all_chunks(transcripts: list) -> list:
    """Run compute_score on every transcript chunk. Adds 'score' key to each."""
    logger.info(f"Scoring {len(transcripts)} chunks...")
    for t in transcripts:
        t["score"] = compute_score(t)
    logger.info("Scoring complete ✅")
    return transcripts