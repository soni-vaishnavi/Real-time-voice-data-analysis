"""
pipeline/phase3_analysis/analyzer.py
======================================
Phase 3 — Combined Analyzer (Dual Channel)

STAGE 0 CHANGES:
  - Removed: compute_combined_score()  — this lived in BOTH analyzer.py AND scorer.py
             Now scoring happens ONLY in Phase 4 (scorer.py → zone_classifier.py → trend_analyzer.py)
  - Removed: local apply_trend_analysis()  — duplicate of trend_analyzer.py
  - Removed: local WEIGHT_*, ZONE_*, EMERGENCY_THRESHOLD constants
  - Removed: sarcasm model imports — now using sarcasm_rules.py
  - Added:   from pipeline.core.config import (all constants)
  - Added:   from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
  - Kept:    All actual Phase 3 analysis: emotion, sarcasm, audio, emergency, fusion
  - Changed: run_phase3() no longer attaches score{} to chunks
             It only attaches emotion_analysis{} and emergency_analysis{}
             Phase 4 (test_phase4.py / main.py) adds score{} via scorer.py

Phase 3 boundary (what this module does):
  IN  → transcript chunk with text + keyword_analysis
  OUT → same chunk with emotion_analysis{} + emergency_analysis{} added
  NOT: no scoring, no zone assignment, no trend analysis
"""

import os
import json
import logging
from typing import Dict, List, Optional

from pipeline.core.config import (
    EMERGENCY_THRESHOLD,
    MIN_WORDS_FOR_EMERGENCY,
)

from pipeline.phase3_analysis.emotion_detector import (
    detect_emotion,
    resolve_sarcasm_conflict,
    get_emotion_model,
)
from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm   # rule-based, 0 MB
from pipeline.phase3_analysis.audio_emotion import (
    analyze_audio_emotion,
    get_audio_emotion_model,
)
from pipeline.phase3_analysis.emotion_fusion import fuse_emotions, resolve_sarcasm_with_audio
from pipeline.phase3_analysis.emergency_detector import (
    detect_emergency,
    start_background_loading,
    is_bart_ready,
    get_emergency_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── MAIN PHASE 3 RUNNER ───────────────────────────────────────────────────────

def run_phase3(
    transcripts_path: str,
    output_dir:       str,
    chunks_dir:       Optional[str] = "output/chunks",
) -> List[Dict]:
    """
    Full Phase 3 dual-channel analysis pipeline.

    Reads:  all_transcripts.json or all_transcripts_final.json
    Writes: all_analysis.json (each chunk has emotion_analysis{} + emergency_analysis{})

    Does NOT compute score{} — that is Phase 4's job.
    Does NOT apply trend analysis — that is Phase 4's job.

    Args:
        transcripts_path: Path to all_transcripts_final.json
        output_dir:       Directory to save all_analysis.json
        chunks_dir:       Path to Phase 1 WAV chunks (for audio emotion, optional)

    Returns:
        List of transcript dicts with emotion_analysis and emergency_analysis added.
    """
    logger.info("=" * 55)
    logger.info("PHASE 3: ANALYSIS (Text Emotion + Emergency + Sarcasm)")
    logger.info("=" * 55)

    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    logger.info(f"Loaded {len(transcripts)} transcripts")

    if any("keyword_analysis" not in t for t in transcripts):
        logger.info("Transcript metadata missing keyword_analysis; applying Phase 2 keyword normalization fallback")
        from pipeline.phase2_stt.keyword_normalizer import apply_keyword_normalization_all
        transcripts = apply_keyword_normalization_all(transcripts)

    audio_mode = bool(chunks_dir and os.path.exists(chunks_dir))
    logger.info(f"Audio emotion: {'ENABLED (wav2vec2)' if audio_mode else 'DISABLED (text-only)'}")

    # ── Pre-load models ────────────────────────────────────────────────────────
    logger.info("Pre-loading text emotion model...")
    get_emotion_model()

    if audio_mode:
        try:
            get_audio_emotion_model()
        except Exception as e:
            logger.warning(f"Audio emotion model failed to load: {e} — falling back to text-only")
            audio_mode = False

    # BART: start background loading if not already started/ready
    if not is_bart_ready():
        logger.info("Starting BART background load...")
        start_background_loading()
        # For batch processing, wait for BART before starting the loop
        logger.info("Waiting for BART to finish loading (batch mode)...")
        get_emergency_model(wait=True)

    logger.info("All models ready — starting analysis loop ✅")

    # ── Per-chunk analysis ─────────────────────────────────────────────────────
    results = []

    for i, t in enumerate(transcripts):
        chunk_id = t.get("chunk_id", f"chunk_{i:03d}")
        text     = t.get("text", "")

        logger.info(f"[{i+1}/{len(transcripts)}] {chunk_id}")

        # ── Text emotion ──────────────────────────────────────────────────────
        text_emotion   = detect_emotion(text)
        sarcasm_result = detect_sarcasm(text)   # rule-based, fast

        # ── Audio emotion (optional) ──────────────────────────────────────────
        audio_result    = {"audio_emotion": {}, "voice_features": {}, "available": False}
        if audio_mode:
            chunk_wav = os.path.join(chunks_dir, f"{chunk_id}.wav")
            if os.path.exists(chunk_wav):
                audio_result = analyze_audio_emotion(chunk_wav)

        audio_available = audio_result["available"]
        audio_emotion   = audio_result.get("audio_emotion", {})

        # ── Emotion fusion ────────────────────────────────────────────────────
        fused_emotion = fuse_emotions(text_emotion, audio_emotion, audio_available)

        # ── Sarcasm resolution (audio-enhanced if available) ──────────────────
        if audio_available:
            audio_override = resolve_sarcasm_with_audio(
                text_emotion, audio_emotion, sarcasm_result, audio_available
            )
            if audio_override.get("use_audio_override"):
                sarcasm_resolution = {
                    "sarcasm_override": audio_override.get("score_penalty", 0) == 0,
                    "score_penalty":    audio_override.get("score_penalty", 0.0),
                    "resolution":       audio_override.get("resolution", "real"),
                    "reason":           audio_override.get("reason", ""),
                    "method":           "audio_enhanced",
                }
            else:
                sarcasm_resolution = resolve_sarcasm_conflict(text_emotion, sarcasm_result)
                sarcasm_resolution["method"] = "text_rules"
        else:
            sarcasm_resolution = resolve_sarcasm_conflict(text_emotion, sarcasm_result)
            sarcasm_resolution["method"] = "text_rules"

        # ── Emergency classification ──────────────────────────────────────────
        word_count = len(text.split()) if text else 0
        if word_count >= MIN_WORDS_FOR_EMERGENCY:
            emergency = detect_emergency(text, wait_for_model=False)

            # Apply keyword boost from Phase 2
            kw_boost = t.get("keyword_analysis", {}).get("total_boost", 0.0)
            kw_cat   = t.get("keyword_analysis", {}).get("top_category", None)

            if kw_boost > 0 and kw_cat and kw_cat.lower() in emergency.get("all_scores", {}):
                boosted = min(emergency["all_scores"][kw_cat.lower()] + kw_boost, 1.0)
                emergency["all_scores"][kw_cat.lower()] = round(boosted, 4)
                new_top = max(emergency["all_scores"], key=emergency["all_scores"].get)
                emergency["top_category"] = new_top
                emergency["top_score"]    = emergency["all_scores"][new_top]
                emergency["is_emergency"] = (
                    new_top != "normal" and emergency["top_score"] >= EMERGENCY_THRESHOLD
                )
        else:
            emergency = {
                "top_category": "normal",
                "top_score":    0.0,
                "is_emergency": False,
                "all_scores":   {},
                "risk_level":   0.0,
                "skipped":      f"too short ({word_count} words)",
                "bart_used":    False,
            }

        # ── Attach results — NO SCORING HERE ─────────────────────────────────
        # Phase 4 (scorer.py → zone_classifier.py → trend_analyzer.py) adds score{}
        t["emotion_analysis"] = {
            "text_emotion":       text_emotion,
            "audio_emotion":      audio_emotion,
            "voice_features":     audio_result.get("voice_features", {}),
            "fused_emotion":      fused_emotion,
            "sarcasm":            sarcasm_result,
            "sarcasm_resolution": sarcasm_resolution,
            "emotion":            fused_emotion,    # backward compat key (scorer.py reads this)
        }
        t["emergency_analysis"] = emergency

        logger.info(
            f"{chunk_id} | "
            f"text_emo={text_emotion['dominant_emotion']} ({text_emotion['dominant_score']:.2f}) | "
            f"fused={fused_emotion['dominant_emotion']} | "
            f"emergency={emergency.get('top_category','normal')} "
            f"({emergency.get('top_score', 0):.2f}) | "
            f"is_emergency={emergency.get('is_emergency', False)}"
        )

        # Save individual chunk analysis
        out_path = os.path.join(output_dir, f"{chunk_id}_analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=2)

        results.append(t)

    # ── Save combined output ───────────────────────────────────────────────────
    combined_path = os.path.join(output_dir, "all_analysis.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Phase 3 complete — {len(results)} chunks analyzed → {combined_path} ✅")
    return results