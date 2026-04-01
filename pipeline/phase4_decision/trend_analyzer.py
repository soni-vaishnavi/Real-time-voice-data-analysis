"""
Phase 4 - Trend Analyzer
Detects escalating patterns across chunks over time.

Rules:
    Rule 1 — Consecutive YELLOW escalation:
        3 or more consecutive YELLOW chunks → upgrade last one to RED
        Why: A slowly escalating situation might never hit RED in a single chunk
             but 3 YELLOWs in a row = clear escalation pattern

    Rule 2 — Rapid score rise:
        If score increases by ≥ 0.20 across 3 consecutive chunks → flag rising trend
        Why: Even if all GREEN, a fast-rising trend is worth noting

    Rule 3 — Incident grouping:
        RED chunks within 2 GREEN chunks of each other → same incident
        Why: A single incident often causes multiple RED triggers with brief pauses

This runs AFTER scorer + zone_classifier, as the final step before saving.
"""

import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CONSECUTIVE_YELLOW_LIMIT = 3    # yellows in a row before upgrading to RED
RAPID_RISE_THRESHOLD     = 0.20  # score increase across 3 chunks = rapid rise
INCIDENT_GAP_LIMIT       = 2    # green chunks between REDs = same incident


def apply_trend_analysis(scored_chunks: List[Dict]) -> List[Dict]:
    """
    Main entry point. Runs all 3 trend rules on the full chunk list.

    Modifies chunks in-place by updating their score["zone"] and adding
    trend metadata fields.

    Args:
        scored_chunks: list of transcripts, each with score dict from zone_classifier

    Returns:
        same list with trend fields added
    """
    logger.info(f"Running trend analysis on {len(scored_chunks)} chunks...")

    scored_chunks = _apply_consecutive_yellow_rule(scored_chunks)
    scored_chunks = _apply_rapid_rise_rule(scored_chunks)
    scored_chunks = _apply_incident_grouping(scored_chunks)

    logger.info("Trend analysis complete ✅")
    return scored_chunks


# ── RULE 1: Consecutive YELLOW → RED ──────────────────────────────────────────

def _apply_consecutive_yellow_rule(chunks: List[Dict]) -> List[Dict]:
    """
    3 consecutive YELLOW chunks → upgrade 3rd (and any further) to RED.
    Sets trend_upgraded=True on upgraded chunks.
    """
    consecutive_yellow = 0

    for chunk in chunks:
        score = chunk.get("score", {})
        zone  = score.get("zone", "GREEN")

        if zone == "YELLOW":
            consecutive_yellow += 1
        elif zone == "RED":
            consecutive_yellow = 0  # RED resets — we already have an alert
        else:
            consecutive_yellow = 0  # GREEN resets

        if consecutive_yellow >= CONSECUTIVE_YELLOW_LIMIT and zone == "YELLOW":
            # Upgrade to RED
            score["zone"]          = "RED"
            score["zone_emoji"]    = "🔴"
            score["trend_upgraded"] = True
            score["auto_alert"]    = True   # escalated = treat as auto-alert
            logger.warning(
                f"📈 TREND UPGRADE | {chunk.get('chunk_id')} | "
                f"{consecutive_yellow} consecutive YELLOW → RED"
            )
        else:
            # Ensure field always exists
            if "trend_upgraded" not in score:
                score["trend_upgraded"] = False

    return chunks


# ── RULE 2: Rapid Score Rise ───────────────────────────────────────────────────

def _apply_rapid_rise_rule(chunks: List[Dict]) -> List[Dict]:
    """
    If score rises by ≥ 0.20 across any 3-chunk window → flag as rising trend.
    Does not change zone — only adds metadata flag for dashboard display.
    """
    for i in range(len(chunks) - 2):
        s0 = chunks[i].get("score", {}).get("final_score", 0.0)
        s2 = chunks[i + 2].get("score", {}).get("final_score", 0.0)

        if (s2 - s0) >= RAPID_RISE_THRESHOLD:
            chunks[i + 2]["score"]["rising_trend"] = True
            logger.info(
                f"📈 RISING TREND | {chunks[i+2].get('chunk_id')} | "
                f"score rose {s0:.3f} → {s2:.3f} (+{s2-s0:.3f}) in 3 chunks"
            )
        else:
            if "rising_trend" not in chunks[i + 2].get("score", {}):
                chunks[i + 2]["score"]["rising_trend"] = False

    # Fill first two chunks (can't compare 3-window for them)
    for i in range(min(2, len(chunks))):
        if "rising_trend" not in chunks[i].get("score", {}):
            chunks[i]["score"]["rising_trend"] = False

    return chunks


# ── RULE 3: Incident Grouping ──────────────────────────────────────────────────

def _apply_incident_grouping(chunks: List[Dict]) -> List[Dict]:
    """
    Group RED chunks that are close together into the same incident.
    RED chunks separated by ≤ INCIDENT_GAP_LIMIT GREEN chunks = same incident.

    Adds incident_id to each RED chunk's score for Phase 5 dashboard grouping.
    """
    incident_counter = 0
    current_incident_id = None
    green_gap = 0

    for chunk in chunks:
        score = chunk.get("score", {})
        zone  = score.get("zone", "GREEN")

        if zone == "RED":
            if current_incident_id is None or green_gap > INCIDENT_GAP_LIMIT:
                # New incident
                incident_counter += 1
                current_incident_id = f"INC_{incident_counter:03d}"
                logger.info(
                    f"🚨 New incident: {current_incident_id} | "
                    f"triggered by {chunk.get('chunk_id')}"
                )

            score["incident_id"] = current_incident_id
            green_gap = 0

        elif zone == "YELLOW":
            # YELLOW extends gap counter but stays in incident range
            if current_incident_id is not None:
                green_gap += 1
                if green_gap <= INCIDENT_GAP_LIMIT:
                    score["incident_id"] = current_incident_id  # still same incident
                else:
                    current_incident_id = None
                    score["incident_id"] = None
            else:
                score["incident_id"] = None

        else:
            # GREEN
            if current_incident_id is not None:
                green_gap += 1
                if green_gap > INCIDENT_GAP_LIMIT:
                    current_incident_id = None
            score["incident_id"] = None

    return chunks


# ── SUMMARY ────────────────────────────────────────────────────────────────────

def get_trend_summary(chunks: List[Dict]) -> Dict:
    """
    Return a summary of trend analysis results.
    Used by test_phase4.py and Phase 5 dashboard.
    """
    upgraded       = [c for c in chunks if c.get("score", {}).get("trend_upgraded")]
    rising         = [c for c in chunks if c.get("score", {}).get("rising_trend")]
    incident_ids   = set(
        c.get("score", {}).get("incident_id")
        for c in chunks
        if c.get("score", {}).get("incident_id")
    )

    return {
        "trend_upgraded_count": len(upgraded),
        "rising_trend_count":   len(rising),
        "total_incidents":      len(incident_ids),
        "incident_ids":         sorted(incident_ids),
        "upgraded_chunks":      [c.get("chunk_id") for c in upgraded],
    }