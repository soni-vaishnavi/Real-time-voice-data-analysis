"""
pipeline/phase4_decision/zone_classifier.py
=============================================
Phase 4 - Zone Classifier. STAGE 0: thresholds imported from pipeline.core.config.
"""

import logging
from typing import Dict

from pipeline.core.config import ZONE_YELLOW, ZONE_RED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORY_SEVERITY = {
    "fire":"CRITICAL","violence":"CRITICAL","medical":"HIGH",
    "accident":"HIGH","mental_health":"HIGH","theft":"MEDIUM","normal":"LOW",
}
AUTO_ALERT_SEVERITIES = {"CRITICAL", "HIGH"}


def classify_zone(score: Dict) -> Dict:
    """Add zone, severity, auto_alert to a scored chunk."""
    final_score = score.get("final_score", 0.0)
    category    = score.get("emergency_category", "normal")

    if final_score < ZONE_YELLOW:
        zone, zone_emoji = "GREEN",  "🟢"
    elif final_score < ZONE_RED:
        zone, zone_emoji = "YELLOW", "🟡"
    else:
        zone, zone_emoji = "RED",    "🔴"

    severity         = CATEGORY_SEVERITY.get(category, "LOW") if zone == "RED" else "LOW"
    auto_alert       = zone == "RED" and severity in AUTO_ALERT_SEVERITIES
    requires_confirm = zone == "RED" and severity == "MEDIUM"

    if zone == "RED":
        logger.warning(f"🔴 RED | score={final_score:.3f} | {category} | {severity} | auto={auto_alert}")
    elif zone == "YELLOW":
        logger.info(f"🟡 YELLOW | score={final_score:.3f} | {category}")

    score.update({
        "zone": zone, "zone_emoji": zone_emoji, "severity": severity,
        "auto_alert": auto_alert, "requires_confirm": requires_confirm,
        "trend_upgraded": False,
    })
    return score


def classify_all_zones(transcripts: list) -> list:
    logger.info(f"Classifying zones for {len(transcripts)} chunks...")
    for t in transcripts:
        if "score" in t:
            t["score"] = classify_zone(t["score"])
    zones = [t["score"]["zone"] for t in transcripts if "score" in t]
    if zones:
        logger.info(f"Zones | 🟢 GREEN={zones.count('GREEN')} | "
                    f"🟡 YELLOW={zones.count('YELLOW')} | 🔴 RED={zones.count('RED')}")
    logger.info("Zone classification complete ✅")
    return transcripts