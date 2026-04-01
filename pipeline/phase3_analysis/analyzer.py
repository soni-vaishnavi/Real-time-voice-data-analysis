"""
Phase 3 - Combined Analyzer (v2 - Dual Channel)
Text emotion + Audio emotion + Sarcasm + Emergency + Fusion scoring
"""

import os
import json
import logging
from typing import Dict, List, Optional

from .emotion_detector import (
    detect_emotion, detect_sarcasm, resolve_sarcasm_conflict,
    get_emotion_model, get_sarcasm_model
)
from .audio_emotion import analyze_audio_emotion, get_audio_emotion_model
from .emotion_fusion import fuse_emotions, resolve_sarcasm_with_audio
from .emergency_detector import detect_emergency, get_emergency_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHT_EMOTION   = 0.35
WEIGHT_EMERGENCY = 0.40
WEIGHT_KEYWORD   = 0.25
ZONE_GREEN_MAX   = 0.45
ZONE_YELLOW_MAX  = 0.72
MIN_WORDS_FOR_EMERGENCY = 4
EMERGENCY_THRESHOLD     = 0.55   # raised from 0.35 to reduce false positives


def compute_combined_score(transcript: Dict) -> Dict:
    emotion_data   = transcript.get("emotion_analysis", {})
    emergency_data = transcript.get("emergency_analysis", {})
    keyword_data   = transcript.get("keyword_analysis", {})

    fused = emotion_data.get("fused_emotion") or emotion_data.get("emotion", {})

    emotion_weight    = fused.get("emergency_weight", 0.0)
    emotion_score     = fused.get("dominant_score", 0.0)
    emotion_component = emotion_weight * emotion_score

    emergency_score     = emergency_data.get("top_score", 0.0)
    is_emergency        = emergency_data.get("is_emergency", False)
    emergency_component = emergency_score if is_emergency else emergency_score * 0.3

    keyword_component = min(keyword_data.get("total_boost", 0.0), 1.0)

    sarcasm_resolution   = emotion_data.get("sarcasm_resolution", {})
    sarcasm_penalty_rate = sarcasm_resolution.get("score_penalty", 0.0)

    base_score = (
        (emotion_component   * WEIGHT_EMOTION)   +
        (emergency_component * WEIGHT_EMERGENCY) +
        (keyword_component   * WEIGHT_KEYWORD)
    )
    sarcasm_deduction = base_score * sarcasm_penalty_rate
    final_score = round(max(0.0, min(base_score - sarcasm_deduction, 1.0)), 4)

    if final_score < ZONE_GREEN_MAX:
        zone, zone_emoji = "GREEN", "🟢"
    elif final_score < ZONE_YELLOW_MAX:
        zone, zone_emoji = "YELLOW", "🟡"
    else:
        zone, zone_emoji = "RED", "🔴"

    return {
        "final_score":       final_score,
        "zone":              zone,
        "zone_emoji":        zone_emoji,
        "components": {
            "emotion_component":   round(emotion_component * WEIGHT_EMOTION, 4),
            "emergency_component": round(emergency_component * WEIGHT_EMERGENCY, 4),
            "keyword_component":   round(keyword_component * WEIGHT_KEYWORD, 4),
            "sarcasm_deduction":   round(sarcasm_deduction, 4),
        },
        "dominant_emotion":   fused.get("dominant_emotion", "neutral"),
        "emergency_category": emergency_data.get("top_category", "normal"),
        "is_emergency":       is_emergency,
        "emotion_source":     fused.get("fusion_method", "text_only"),
        "text_emotion":       fused.get("text_dominant", fused.get("dominant_emotion", "?")),
        "audio_emotion_used": fused.get("audio_dominant", "unavailable"),
    }


def apply_trend_analysis(scored_transcripts: List[Dict]) -> List[Dict]:
    consecutive_yellow = 0
    for t in scored_transcripts:
        zone = t["score"]["zone"]
        if zone == "YELLOW":
            consecutive_yellow += 1
        else:
            consecutive_yellow = 0
        if consecutive_yellow >= 3 and zone == "YELLOW":
            t["score"]["zone"] = "RED"
            t["score"]["zone_emoji"] = "🔴"
            t["score"]["trend_upgraded"] = True
            logger.warning(f"{t.get('chunk_id')} | TREND UPGRADE: 3x YELLOW → RED")
        else:
            t["score"]["trend_upgraded"] = False
    return scored_transcripts


def run_phase3(
    transcripts_path: str,
    output_dir: str,
    chunks_dir: Optional[str] = "output/chunks"
) -> List[Dict]:
    """
    Full Phase 3 dual-channel pipeline.
    chunks_dir: path to WAV chunks from Phase 1 (for audio emotion)
                Set None to run text-only mode.
    """
    logger.info("=" * 50)
    logger.info("PHASE 3: DUAL-CHANNEL ANALYSIS (Text + Audio)")
    logger.info("=" * 50)

    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    logger.info(f"Loaded {len(transcripts)} transcripts")
    os.makedirs(output_dir, exist_ok=True)

    audio_mode = bool(chunks_dir and os.path.exists(chunks_dir))
    logger.info(f"Audio emotion: {'ENABLED' if audio_mode else 'DISABLED (text-only)'}")

    # Pre-load models
    logger.info("Pre-loading models...")
    get_emotion_model()
    get_sarcasm_model()
    get_emergency_model()
    if audio_mode:
        try:
            get_audio_emotion_model()
        except Exception as e:
            logger.warning(f"Audio model failed to load: {e} — falling back to text-only")
            audio_mode = False
    logger.info("Models ready ✅")

    results = []
    for i, t in enumerate(transcripts):
        chunk_id = t["chunk_id"]
        text     = t.get("text", "")
        logger.info(f"[{i+1}/{len(transcripts)}] {chunk_id}")

        # Text emotion + sarcasm
        text_emotion   = detect_emotion(text)
        sarcasm_result = detect_sarcasm(text)

        # Audio emotion
        audio_result    = {"audio_emotion": {}, "voice_features": {}, "available": False}
        if audio_mode:
            chunk_path = os.path.join(chunks_dir, f"{chunk_id}.wav")
            if os.path.exists(chunk_path):
                audio_result = analyze_audio_emotion(chunk_path)

        audio_available = audio_result["available"]
        audio_emotion   = audio_result["audio_emotion"]

        # Fuse text + audio
        fused_emotion = fuse_emotions(text_emotion, audio_emotion, audio_available)

        # Enhanced sarcasm resolution
        audio_override = resolve_sarcasm_with_audio(
            text_emotion, audio_emotion, sarcasm_result, audio_available
        )
        if audio_override.get("use_audio_override"):
            sarcasm_resolution = {
                "sarcasm_override": audio_override.get("score_penalty", 0) == 0,
                "score_penalty":    audio_override.get("score_penalty", 0.0),
                "resolution":       audio_override.get("resolution", "real"),
                "reason":           audio_override.get("reason", ""),
                "method":           "audio_enhanced"
            }
        else:
            sarcasm_resolution = resolve_sarcasm_conflict(text_emotion, sarcasm_result)
            sarcasm_resolution["method"] = "text_only"

        # Emergency detection (with fixes)
        word_count = len(text.split()) if text else 0
        if word_count >= MIN_WORDS_FOR_EMERGENCY:
            emergency = detect_emergency(text)
            if emergency["top_score"] < EMERGENCY_THRESHOLD:
                emergency["is_emergency"] = False
                emergency["risk_level"]   = 0.0
            # Keyword boost
            kw_boost = t.get("keyword_analysis", {}).get("total_boost", 0.0)
            kw_cat   = t.get("keyword_analysis", {}).get("top_category", None)
            if kw_boost > 0 and kw_cat and kw_cat in emergency.get("all_scores", {}):
                boosted = min(emergency["all_scores"][kw_cat] + kw_boost, 1.0)
                emergency["all_scores"][kw_cat] = round(boosted, 4)
                new_top = max(emergency["all_scores"], key=emergency["all_scores"].get)
                emergency["top_category"] = new_top
                emergency["top_score"]    = emergency["all_scores"][new_top]
                emergency["is_emergency"] = emergency["top_score"] >= EMERGENCY_THRESHOLD
        else:
            emergency = {
                "top_category": "normal", "top_score": 0.0,
                "is_emergency": False, "all_scores": {}, "risk_level": 0.0,
                "skipped": f"too short ({word_count} words)"
            }

        # Attach results
        t["emotion_analysis"] = {
            "text_emotion":       text_emotion,
            "audio_emotion":      audio_emotion,
            "voice_features":     audio_result.get("voice_features", {}),
            "fused_emotion":      fused_emotion,
            "sarcasm":            sarcasm_result,
            "sarcasm_resolution": sarcasm_resolution,
            "emotion":            fused_emotion,   # backward compat key
        }
        t["emergency_analysis"] = emergency
        t["score"] = compute_combined_score(t)

        logger.info(
            f"{chunk_id} | {t['score']['zone_emoji']} {t['score']['zone']} "
            f"({t['score']['final_score']:.3f}) | "
            f"text_emo={fused_emotion.get('text_dominant','?')} "
            f"audio_emo={fused_emotion.get('audio_dominant','N/A')} "
            f"fused={fused_emotion['dominant_emotion']} | "
            f"emergency={emergency.get('top_category','normal')}"
        )

        out_path = os.path.join(output_dir, f"{chunk_id}_analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=2)
        results.append(t)

    results = apply_trend_analysis(results)

    combined_path = os.path.join(output_dir, "all_analysis.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(results)} results → {combined_path}")
    logger.info("PHASE 3 COMPLETE ✅")
    return results
