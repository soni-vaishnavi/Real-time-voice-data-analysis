"""
Phase 4 - Zone Classifier
Takes the final_score from scorer.py and assigns GREEN / YELLOW / RED zone.

Zones:
    GREEN  < 0.45  → normal conversation, no concern
    YELLOW 0.45–0.72 → concerning, monitor closely
    RED    > 0.72  → emergency, trigger alert

Also determines severity and alert type per emergency category:
    CRITICAL → fire, violence        → auto-alert immediately
    HIGH     → medical, accident,
               mental_health         → auto-alert immediately
    MEDIUM   → theft                 → hold for human confirmation
    LOW      → normal                → log only
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── ZONE THRESHOLDS ────────────────────────────────────────────────────────────
ZONE_GREEN_MAX  = 0.45   # below this = GREEN
ZONE_YELLOW_MAX = 0.72   # below this = YELLOW, above = RED

# ── SEVERITY MAP ───────────────────────────────────────────────────────────────
# Maps emergency category → severity level
CATEGORY_SEVERITY = {
    "fire":          "CRITICAL",
    "violence":      "CRITICAL",
    "medical":       "HIGH",
    "accident":      "HIGH",
    "mental_health": "HIGH",
    "theft":         "MEDIUM",
    "normal":        "LOW",
}

# Severity → whether to auto-alert or wait for human confirm
AUTO_ALERT_SEVERITIES = {"CRITICAL", "HIGH"}


def classify_zone(score: Dict) -> Dict:
    """
    Assign zone to a scored chunk.

    Args:
        score: dict returned by scorer.compute_score() — must have 'final_score'

    Returns score dict with zone fields added:
        {
            ... (all existing score fields),
            "zone":             "RED",
            "zone_emoji":       "🔴",
            "severity":         "HIGH",
            "auto_alert":       True,
            "requires_confirm": False,
            "trend_upgraded":   False,   ← set by trend_analyzer, default False here
        }
    """
    final_score = score.get("final_score", 0.0)
    category    = score.get("emergency_category", "normal")

    # Zone classification
    if final_score < ZONE_GREEN_MAX:
        zone       = "GREEN"
        zone_emoji = "🟢"
    elif final_score < ZONE_YELLOW_MAX:
        zone       = "YELLOW"
        zone_emoji = "🟡"
    else:
        zone       = "RED"
        zone_emoji = "🔴"

    # Severity — only meaningful if zone is RED
    severity = CATEGORY_SEVERITY.get(category, "LOW")
    if zone != "RED":
        severity = "LOW"

    auto_alert       = zone == "RED" and severity in AUTO_ALERT_SEVERITIES
    requires_confirm = zone == "RED" and severity == "MEDIUM"

    if zone == "RED":
        logger.warning(
            f"🔴 RED zone | score={final_score:.3f} | "
            f"category={category} | severity={severity} | "
            f"auto_alert={auto_alert}"
        )
    elif zone == "YELLOW":
        logger.info(f"🟡 YELLOW zone | score={final_score:.3f} | category={category}")

    score.update({
        "zone":             zone,
        "zone_emoji":       zone_emoji,
        "severity":         severity,
        "auto_alert":       auto_alert,
        "requires_confirm": requires_confirm,
        "trend_upgraded":   False,  # trend_analyzer may override this
    })

    return score


def classify_all_zones(transcripts: list) -> list:
    """Run classify_zone on every transcript chunk (after scoring)."""
    logger.info(f"Classifying zones for {len(transcripts)} chunks...")
    for t in transcripts:
        if "score" not in t:
            logger.warning(f"{t.get('chunk_id')} has no score — run scorer first")
            continue
        t["score"] = classify_zone(t["score"])

    # Log distribution
    zones = [t["score"]["zone"] for t in transcripts if "score" in t]
    total = len(zones)
    if total:
        logger.info(
            f"Zone distribution | "
            f"🟢 GREEN={zones.count('GREEN')} | "
            f"🟡 YELLOW={zones.count('YELLOW')} | "
            f"🔴 RED={zones.count('RED')}"
        )
    logger.info("Zone classification complete ✅")
    return transcripts