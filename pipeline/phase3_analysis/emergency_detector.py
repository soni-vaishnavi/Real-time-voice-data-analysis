"""
pipeline/phase3_analysis/emergency_detector.py
===============================================
Phase 3 - Emergency Detector using BART zero-shot classification.

STAGE 0 CHANGES:
  - EMERGENCY_THRESHOLD now imported from pipeline.core.config (single source, value = 0.55)
  - BART loads in a background daemon thread at startup — no cold-start freeze on first chunk
  - Added: start_background_loading(), get_emergency_model(wait), is_bart_ready()

Model: facebook/bart-large-mnli (~1.6 GB)
Zero-shot classification — no task-specific training needed.
"""

import threading
import logging
from typing import Dict, List, Optional

from pipeline.core.config import (
    EMERGENCY_MODEL,
    EMERGENCY_LABELS,
    EMERGENCY_LABEL_SHORT,
    EMERGENCY_THRESHOLD,
    MIN_WORDS_FOR_EMERGENCY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── LOADING STATE ──────────────────────────────────────────────────────────────
_emergency_pipeline = None
_bart_ready         = threading.Event()
_bart_load_error    = None


def start_background_loading() -> threading.Thread:
    """
    Start loading BART in a background daemon thread. Returns immediately.
    Call once at app startup (FastAPI lifespan or main.py).
    """
    if _bart_ready.is_set():
        return threading.current_thread()

    def _load():
        global _emergency_pipeline, _bart_load_error
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("BART loading in background (~1.6 GB, 30-60s on first run)...")
            _emergency_pipeline = hf_pipeline(
                "zero-shot-classification",
                model=EMERGENCY_MODEL,
                framework="pt",
                device=-1,
            )
            logger.info("BART emergency model ready ✅")
        except Exception as e:
            _bart_load_error = e
            logger.error(f"BART loading failed: {e}")
        finally:
            _bart_ready.set()

    t = threading.Thread(target=_load, daemon=True, name="BARTLoader")
    t.start()
    return t


def get_emergency_model(wait: bool = False) -> Optional[object]:
    """
    Return BART pipeline if ready, None otherwise.
    wait=True blocks up to 120s (use in batch processing).
    wait=False returns immediately (use in hot real-time path).
    """
    if wait and not _bart_ready.is_set():
        logger.info("Waiting for BART...")
        _bart_ready.wait(timeout=120)
    if _bart_load_error is not None:
        return None
    return _emergency_pipeline if _bart_ready.is_set() else None


def is_bart_ready() -> bool:
    return _bart_ready.is_set() and _emergency_pipeline is not None


CATEGORY_RISK = {
    "medical": 0.90, "fire": 0.95, "violence": 0.85,
    "accident": 0.80, "theft": 0.70, "mental_health": 0.85, "normal": 0.0,
}


def detect_emergency(text: str, wait_for_model: bool = False) -> Dict:
    """
    Classify text into emergency categories using BART zero-shot NLI.
    Returns empty result (not error) when BART is not yet loaded.
    """
    if not text or len(text.strip()) < 3:
        return _empty_result()

    if len(text.split()) < MIN_WORDS_FOR_EMERGENCY:
        return _empty_result(skipped=f"too short ({len(text.split())} words)")

    model = get_emergency_model(wait=wait_for_model)
    if model is None:
        return _empty_result(bart_used=False)

    try:
        result   = model(text[:512], candidate_labels=EMERGENCY_LABELS, multi_label=False)
        all_scores = {
            EMERGENCY_LABEL_SHORT.get(lbl, lbl): round(sc, 4)
            for lbl, sc in zip(result["labels"], result["scores"])
        }
        top_category = max(all_scores, key=all_scores.get)
        top_score    = all_scores[top_category]
        is_emergency = top_category != "normal" and top_score >= EMERGENCY_THRESHOLD
        risk_level   = CATEGORY_RISK.get(top_category, 0.0) if is_emergency else 0.0

        logger.info(f"Emergency: {top_category} ({top_score:.2f}) | is_emergency={is_emergency}")
        return {
            "top_category": top_category, "top_score": top_score,
            "is_emergency": is_emergency, "all_scores": all_scores,
            "risk_level": risk_level, "bart_used": True,
        }
    except Exception as e:
        logger.error(f"Emergency detection failed: {e}")
        return _empty_result()


def _empty_result(skipped: str = None, bart_used: bool = True) -> Dict:
    r = {"top_category":"normal","top_score":0.0,"is_emergency":False,
         "all_scores":{},"risk_level":0.0,"bart_used":bart_used}
    if skipped:
        r["skipped"] = skipped
    return r


def analyze_emergency_batch(transcripts: List[Dict]) -> List[Dict]:
    """Batch emergency detection with keyword boost. Used by file pipeline."""
    logger.info(f"Running emergency detection on {len(transcripts)} transcripts")
    if not is_bart_ready():
        get_emergency_model(wait=True)

    for t in transcripts:
        emergency = detect_emergency(t.get("text",""), wait_for_model=False)
        kw_boost  = t.get("keyword_analysis", {}).get("total_boost", 0.0)
        kw_cat    = t.get("keyword_analysis", {}).get("top_category", None)

        if kw_boost > 0 and kw_cat and kw_cat.lower() in emergency.get("all_scores", {}):
            boosted = min(emergency["all_scores"][kw_cat.lower()] + kw_boost, 1.0)
            emergency["all_scores"][kw_cat.lower()] = round(boosted, 4)
            new_top = max(emergency["all_scores"], key=emergency["all_scores"].get)
            emergency.update({
                "top_category": new_top,
                "top_score":    emergency["all_scores"][new_top],
                "is_emergency": new_top != "normal" and emergency["all_scores"][new_top] >= EMERGENCY_THRESHOLD,
            })
        t["emergency_analysis"] = emergency

    logger.info("Emergency batch detection complete ✅")
    return transcripts