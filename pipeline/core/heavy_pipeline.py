"""
pipeline/core/heavy_pipeline.py
=================================
STAGE 6 — Conditional Heavy Pipeline

Runs wav2vec2 audio emotion analysis ONLY on YELLOW and RED chunks.
GREEN chunks skip this entirely — no RAM wasted.

Why conditional?
  - wav2vec2 takes ~3-8 seconds per chunk on CPU
  - 90%+ of live surveillance is GREEN (safe)
  - Running wav2vec2 on every chunk: 8s/chunk, unusable in real-time
  - Running only on flagged chunks: 8s only when something is suspicious

What it does:
  After the light pipeline scores a chunk as YELLOW or RED:
  1. Run wav2vec2 on the audio file → audio emotion scores
  2. Fuse with text emotion using emotion_fusion.py rules
  3. Re-compute score with fused emotion
  4. Re-classify zone (fused emotion may change YELLOW→GREEN or RED→YELLOW)

Loading strategy:
  - wav2vec2 loads on FIRST YELLOW/RED chunk (lazy, not at startup)
  - Stays loaded for the session (cached in module-level global)
  - First flagged chunk takes ~3s extra for model load (~1.2 GB)
  - Subsequent flagged chunks: ~3-8s for inference only

Usage (called from worker.py after initial score):
    from pipeline.core.heavy_pipeline import run_heavy_pipeline_if_needed
    result = run_heavy_pipeline_if_needed(chunk_data, audio_path)
"""

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Global model cache ─────────────────────────────────────────────────────────
_audio_emotion_pipeline = None
_load_lock              = threading.Lock()
_load_attempted         = False


def _load_wav2vec2() -> Optional[object]:
    """
    Load wav2vec2 audio emotion model on demand.
    Thread-safe — only loads once even if multiple RED chunks arrive simultaneously.
    """
    global _audio_emotion_pipeline, _load_attempted

    if _audio_emotion_pipeline is not None:
        return _audio_emotion_pipeline

    with _load_lock:
        # Double-check after acquiring lock
        if _audio_emotion_pipeline is not None:
            return _audio_emotion_pipeline

        if _load_attempted:
            return None  # already tried and failed

        _load_attempted = True
        try:
            # Suppress LOAD REPORT warnings
            try:
                import transformers
                transformers.logging.set_verbosity_error()
            except Exception:
                pass

            from transformers import pipeline as hf_pipeline
            logger.info("Stage 6: Loading wav2vec2 audio emotion model (~1.2 GB)...")
            _audio_emotion_pipeline = hf_pipeline(
                "audio-classification",
                model     = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                framework = "pt",
                device    = -1,   # CPU
            )
            logger.info("Stage 6: wav2vec2 ready ✅")
        except Exception as e:
            logger.error(f"Stage 6: wav2vec2 failed to load: {e}")
            logger.warning("Stage 6: Heavy pipeline disabled — text-only emotion will be used")
            _audio_emotion_pipeline = None

    return _audio_emotion_pipeline


def is_heavy_available() -> bool:
    """Check if wav2vec2 is loaded and ready."""
    return _audio_emotion_pipeline is not None


# ── AUDIO EMOTION LABEL MAP ────────────────────────────────────────────────────
_LABEL_MAP = {
    "angry":    "anger",   "anger":    "anger",
    "disgust":  "disgust",
    "fearful":  "fear",    "fear":     "fear",
    "happy":    "joy",     "joy":      "joy",
    "neutral":  "neutral", "calm":     "neutral",
    "sad":      "sadness", "sadness":  "sadness",
    "surprised":"surprise","surprise": "surprise",
}

_EMERGENCY_WEIGHT = {
    "fear":    1.0, "anger":   0.7, "sadness":  0.5,
    "surprise":0.3, "disgust": 0.2, "joy":      0.0, "neutral":  0.0,
}


def detect_audio_emotion(audio_path: str) -> Optional[Dict]:
    """
    Run wav2vec2 on one audio file.
    Returns emotion dict or None if model unavailable / file missing.
    """
    import os
    if not audio_path or not os.path.exists(audio_path):
        return None

    model = _load_wav2vec2()
    if model is None:
        return None

    try:
        results = model(audio_path, top_k=None)

        # Normalize labels and accumulate scores
        all_scores: Dict[str, float] = {}
        for item in results:
            label = _LABEL_MAP.get(item["label"].lower(), item["label"].lower())
            all_scores[label] = all_scores.get(label, 0.0) + item["score"]

        # Re-normalize to sum = 1.0
        total = sum(all_scores.values())
        if total > 0:
            all_scores = {k: round(v / total, 4) for k, v in all_scores.items()}

        dominant = max(all_scores, key=all_scores.get) if all_scores else "neutral"
        return {
            "dominant_emotion": dominant,
            "dominant_score":   all_scores.get(dominant, 0.0),
            "all_scores":       all_scores,
            "emergency_weight": _EMERGENCY_WEIGHT.get(dominant, 0.0),
            "fear_score":       all_scores.get("fear",  0.0),
            "anger_score":      all_scores.get("anger", 0.0),
            "source":           "audio_wav2vec2",
        }
    except Exception as e:
        logger.error(f"Audio emotion inference failed: {e}")
        return None


# ── EMOTION FUSION ─────────────────────────────────────────────────────────────

def fuse_text_and_audio_emotion(
    text_emotion:  Dict,
    audio_emotion: Dict,
) -> Dict:
    """
    Merge text and audio emotion scores.

    Rules (same as emotion_fusion.py but simplified for worker use):
    - Both agree        → 50/50 average
    - Audio fear > 0.50 → audio wins 80/20 (voice doesn't lie)
    - Disagree          → audio wins 70/30
    """
    text_dom  = text_emotion.get("dominant_emotion", "neutral")
    audio_dom = audio_emotion.get("dominant_emotion", "neutral")
    t_scores  = text_emotion.get("all_scores", {})
    a_scores  = audio_emotion.get("all_scores", {})

    audio_fear = a_scores.get("fear", 0.0)

    if audio_fear > 0.50:
        a_w, t_w, method = 0.80, 0.20, "audio_fear_override"
    elif text_dom == audio_dom:
        a_w, t_w, method = 0.50, 0.50, "agreement"
    else:
        a_w, t_w, method = 0.70, 0.30, "audio_wins_disagreement"
        logger.info(f"Emotion disagreement: text={text_dom} audio={audio_dom} → {method}")

    all_emotions = set(t_scores) | set(a_scores)
    fused = {}
    for emo in all_emotions:
        fused[emo] = round(t_scores.get(emo, 0.0) * t_w + a_scores.get(emo, 0.0) * a_w, 4)

    total = sum(fused.values())
    if total > 0:
        fused = {k: round(v / total, 4) for k, v in fused.items()}

    final_dom = max(fused, key=fused.get) if fused else "neutral"
    return {
        "dominant_emotion":  final_dom,
        "dominant_score":    fused.get(final_dom, 0.0),
        "all_scores":        fused,
        "emergency_weight":  _EMERGENCY_WEIGHT.get(final_dom, 0.0),
        "fear_score":        fused.get("fear", 0.0),
        "anger_score":       fused.get("anger", 0.0),
        "fusion_method":     method,
        "text_dominant":     text_dom,
        "audio_dominant":    audio_dom,
    }


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def run_heavy_pipeline_if_needed(
    chunk_data: Dict,
    audio_path: str,
) -> Dict:
    """
    Called from worker after the light pipeline scores a chunk.
    Runs wav2vec2 only if zone is YELLOW or RED.

    Mutates chunk_data in place — updates emotion_analysis and score.
    Returns updated chunk_data.

    Args:
        chunk_data:  Full chunk result dict (already has score + zone from light pipeline)
        audio_path:  Path to the WAV file for audio emotion analysis

    Returns:
        Same chunk_data, possibly with updated emotion_analysis and score.
        If wav2vec2 is unavailable or zone is GREEN, returns unchanged.
    """
    zone = chunk_data.get("score", {}).get("zone", "GREEN")

    # Skip GREEN chunks entirely — no heavy processing
    if zone == "GREEN":
        chunk_data["heavy_pipeline"] = False
        return chunk_data

    # Attempt audio emotion analysis
    logger.info(f"Stage 6: Running wav2vec2 on {zone} chunk {chunk_data.get('chunk_id','?')}")
    audio_emotion = detect_audio_emotion(audio_path)

    if audio_emotion is None:
        logger.info("Stage 6: wav2vec2 unavailable — keeping light pipeline result")
        chunk_data["heavy_pipeline"] = False
        return chunk_data

    # Get existing text emotion
    text_emotion = chunk_data.get("emotion_analysis", {}).get("emotion", {})
    if not text_emotion:
        chunk_data["heavy_pipeline"] = False
        return chunk_data

    # Fuse text + audio emotions
    fused_emotion = fuse_text_and_audio_emotion(text_emotion, audio_emotion)

    # Update emotion_analysis
    chunk_data["emotion_analysis"].update({
        "audio_emotion":  audio_emotion,
        "fused_emotion":  fused_emotion,
        "emotion":        fused_emotion,  # scorer reads "emotion" key
    })

    # Re-score with fused emotion
    from pipeline.phase4_decision.scorer import compute_score
    from pipeline.phase4_decision.zone_classifier import classify_zone

    old_score = chunk_data.get("score", {}).get("final_score", 0.0)
    old_zone  = zone

    new_score_dict = compute_score(chunk_data)
    classify_zone(new_score_dict)

    new_score = new_score_dict.get("final_score", 0.0)
    new_zone  = new_score_dict.get("zone", "GREEN")

    # Preserve trend/incident info from light pipeline
    for key in ("trend_upgraded", "rising_trend", "incident_id",
                "auto_alert", "requires_confirm"):
        if key in chunk_data.get("score", {}):
            new_score_dict[key] = chunk_data["score"][key]

    chunk_data["score"]          = new_score_dict
    chunk_data["heavy_pipeline"] = True

    if old_zone != new_zone:
        logger.info(
            f"Stage 6: Zone changed {old_zone}→{new_zone} | "
            f"score {old_score:.3f}→{new_score:.3f} | "
            f"fused_emotion={fused_emotion['dominant_emotion']} "
            f"(method={fused_emotion.get('fusion_method','?')})"
        )
    else:
        logger.info(
            f"Stage 6: Zone unchanged ({zone}) | "
            f"score {old_score:.3f}→{new_score:.3f} | "
            f"fused={fused_emotion['dominant_emotion']}"
        )

    return chunk_data