"""
Phase 3 - Emotion Fusion
Combines text-based and audio-based emotion for final emotion score.

Why fusion beats single-source:
- Text alone: misses vocal tone, pitch, tremor
- Audio alone: misses semantic meaning, specific words
- Combined: catches "I'm fine" said while crying
            catches "I'm going to die" said sarcastically

Fusion rules:
1. If text and audio AGREE on emotion → high confidence, weighted average
2. If text and audio DISAGREE → audio wins (70/30 split)
   Voice pitch/tremor cannot be faked. Text can be misleading.
3. If audio unavailable → fall back to text only (existing behavior)
4. Special case: "fear" from audio ALWAYS treated as real signal
   (people don't accidentally sound afraid)
"""

import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fusion weights when text and audio DISAGREE
AUDIO_WEIGHT_ON_DISAGREE = 0.70
TEXT_WEIGHT_ON_DISAGREE  = 0.30

# Fusion weights when text and audio AGREE
AUDIO_WEIGHT_ON_AGREE = 0.50
TEXT_WEIGHT_ON_AGREE  = 0.50

# Emotions considered "emergency signals"
EMERGENCY_EMOTIONS = {"fear", "anger", "sadness"}

# Emergency weight per emotion (same as before)
EMOTION_EMERGENCY_WEIGHT = {
    "fear":     1.0,
    "anger":    0.7,
    "sadness":  0.5,
    "surprise": 0.3,
    "disgust":  0.2,
    "joy":      0.0,
    "neutral":  0.0,
}


def fuse_emotions(
    text_emotion: Dict,
    audio_emotion: Dict,
    audio_available: bool = True
) -> Dict:
    """
    Fuse text and audio emotion results into one final emotion.

    Args:
        text_emotion:     result from emotion_detector.detect_emotion()
        audio_emotion:    result from audio_emotion.detect_audio_emotion()
        audio_available:  False if audio file was not accessible

    Returns:
        fused emotion dict with source tracking
    """

    # ── Audio not available → use text only ──────────────────
    if not audio_available or not audio_emotion.get("all_scores"):
        logger.debug("Audio unavailable — using text emotion only")
        return {
            **text_emotion,
            "fusion_method": "text_only",
            "fusion_confidence": "low",
            "text_dominant": text_emotion.get("dominant_emotion", "neutral"),
            "audio_dominant": "unavailable",
        }

    text_dominant  = text_emotion.get("dominant_emotion", "neutral")
    audio_dominant = audio_emotion.get("dominant_emotion", "neutral")
    text_scores    = text_emotion.get("all_scores", {})
    audio_scores   = audio_emotion.get("all_scores", {})

    # ── Get all unique emotion labels ─────────────────────────
    all_emotions = set(text_scores.keys()) | set(audio_scores.keys())

    # ── Check agreement ───────────────────────────────────────
    emotions_agree = (text_dominant == audio_dominant)

    # Special: if audio detects fear, always treat as real
    audio_fear = audio_scores.get("fear", 0.0)
    audio_fear_strong = audio_fear > 0.45   # strong fear signal from voice

    if audio_fear_strong:
        fusion_method = "audio_fear_override"
        a_weight = 0.80   # audio dominates when voice sounds genuinely afraid
        t_weight = 0.20
        logger.info(f"Audio fear override: {audio_fear:.2f} — audio dominant")
    elif emotions_agree:
        fusion_method = "agreement"
        a_weight = AUDIO_WEIGHT_ON_AGREE
        t_weight = TEXT_WEIGHT_ON_AGREE
    else:
        fusion_method = "audio_wins_disagreement"
        a_weight = AUDIO_WEIGHT_ON_DISAGREE
        t_weight = TEXT_WEIGHT_ON_DISAGREE

    # ── Fuse scores ───────────────────────────────────────────
    fused_scores = {}
    for emotion in all_emotions:
        t_score = text_scores.get(emotion, 0.0)
        a_score = audio_scores.get(emotion, 0.0)
        fused_scores[emotion] = round(
            (t_score * t_weight) + (a_score * a_weight), 4
        )

    # Re-normalize to sum to 1.0
    total = sum(fused_scores.values())
    if total > 0:
        fused_scores = {k: round(v / total, 4) for k, v in fused_scores.items()}

    # ── Find final dominant ───────────────────────────────────
    final_dominant = max(fused_scores, key=fused_scores.get)
    final_score    = fused_scores[final_dominant]
    emergency_weight = EMOTION_EMERGENCY_WEIGHT.get(final_dominant, 0.0)

    # ── Log what happened ─────────────────────────────────────
    if not emotions_agree:
        logger.info(
            f"Emotion disagreement: text={text_dominant} audio={audio_dominant} "
            f"→ fused={final_dominant} ({fusion_method})"
        )
    else:
        logger.debug(
            f"Emotion agreement: both={text_dominant} "
            f"→ fused={final_dominant} ({fusion_method})"
        )

    return {
        "dominant_emotion":  final_dominant,
        "dominant_score":    final_score,
        "all_scores":        fused_scores,
        "emergency_weight":  emergency_weight,
        "fear_score":        fused_scores.get("fear", 0.0),
        "anger_score":       fused_scores.get("anger", 0.0),
        "fusion_method":     fusion_method,
        "fusion_confidence": "high" if emotions_agree else "medium",
        "text_dominant":     text_dominant,
        "audio_dominant":    audio_dominant,
        "text_weight_used":  t_weight,
        "audio_weight_used": a_weight,
    }


def resolve_sarcasm_with_audio(
    text_emotion: Dict,
    audio_emotion: Dict,
    sarcasm_result: Dict,
    audio_available: bool
) -> Dict:
    """
    Enhanced sarcasm resolution using audio signals.

    Audio is the ultimate sarcasm detector:
    - "I'm totally fine" (sarcastic text) + scared voice = NOT sarcasm → real distress
    - "I'm going to die" (scary text) + laughing voice = sarcasm → casual expression

    This overrides the text-only sarcasm conflict resolution.
    """
    is_sarcastic    = sarcasm_result.get("is_sarcastic", False)
    sarcasm_score   = sarcasm_result.get("sarcasm_score", 0.0)
    text_fear       = text_emotion.get("fear_score", 0.0)

    if not audio_available:
        # Fall back to text-only resolution
        return {"use_audio_override": False, "reason": "audio_unavailable"}

    audio_fear    = audio_emotion.get("fear_score", 0.0)
    audio_joy     = audio_emotion.get("all_scores", {}).get("joy", 0.0)
    audio_neutral = audio_emotion.get("all_scores", {}).get("neutral", 0.0)
    audio_dominant = audio_emotion.get("dominant_emotion", "neutral")

    # Rule 1: Text says sarcasm, but audio sounds genuinely afraid
    # → Override sarcasm, treat as real emergency
    if is_sarcastic and audio_fear > 0.50:
        return {
            "use_audio_override": True,
            "score_penalty": 0.0,   # no penalty — audio says it's real
            "resolution": "real",
            "reason": f"Sarcasm overridden — audio fear={audio_fear:.2f} is genuine"
        }

    # Rule 2: Text sounds scary, but audio sounds calm/happy
    # → Likely sarcasm or casual expression
    if text_fear > 0.60 and (audio_joy > 0.40 or audio_neutral > 0.60):
        return {
            "use_audio_override": True,
            "score_penalty": 0.60,
            "resolution": "sarcasm",
            "reason": f"Text fear={text_fear:.2f} but audio={audio_dominant} (calm) → sarcasm"
        }

    # Rule 3: Both text and audio show fear
    # → Definitely real, no penalty
    if text_fear > 0.40 and audio_fear > 0.40:
        return {
            "use_audio_override": True,
            "score_penalty": 0.0,
            "resolution": "real",
            "reason": f"Both text_fear={text_fear:.2f} audio_fear={audio_fear:.2f} → confirmed real"
        }

    # Default: no audio override, use existing text-based sarcasm logic
    return {"use_audio_override": False, "reason": "no_audio_override_needed"}
