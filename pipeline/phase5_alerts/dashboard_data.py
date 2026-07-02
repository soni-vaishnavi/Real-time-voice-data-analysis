"""
pipeline/phase5_alerts/dashboard_data.py
=========================================
STAGE 7 — Dashboard Data Layer

Abstracts the data source for the Streamlit dashboard.
  "file" mode → reads output/decisions/all_decisions.json  (original, file pipeline)
  "api"  mode → polls GET /chunks from FastAPI             (live mic mode)

This means dashboard.py doesn't need to know where data comes from.
It just calls load_chunks() and gets a list of chunk dicts either way.

Usage in dashboard.py:
    from pipeline.phase5_alerts.dashboard_data import load_chunks, data_source_label

    chunks = load_chunks()   # works for both file and live mode
"""

import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
try:
    from pipeline.core.config import (
        DASHBOARD_DATA_SOURCE, DASHBOARD_API_URL, DECISIONS_DIR
    )
    _DATA_SOURCE = DASHBOARD_DATA_SOURCE    # "file" or "api"
    _API_URL     = DASHBOARD_API_URL
    _FILE_PATH   = str(DECISIONS_DIR / "all_decisions.json")
except Exception:
    _DATA_SOURCE = "file"
    _API_URL     = "http://localhost:8000"
    _FILE_PATH   = "output/decisions/all_decisions.json"


def data_source_label() -> str:
    """Human-readable label for the current data source."""
    if _DATA_SOURCE == "api":
        return f"🔴 LIVE  (API {_API_URL})"
    return "📁 FILE  (all_decisions.json)"


def load_chunks(limit: int = 1000) -> List[Dict]:
    """
    Load analyzed chunks from whichever source is configured.

    Returns:
        List of chunk dicts, newest last.
        Empty list on error.
    """
    if _DATA_SOURCE == "api":
        return _load_from_api(limit)
    return _load_from_file()


def post_action(chunk_id: str, action: str, operator_id: str = "dashboard",
                reason: str = "") -> Dict:
    """
    Post CONFIRM / REJECT action.
    In file mode: updates in-memory state only (no API to call).
    In API mode:  POST /alerts/{chunk_id}/action
    """
    if _DATA_SOURCE == "api":
        return _post_to_api(chunk_id, action, operator_id, reason)
    # File mode: just return success (dashboard manages state itself)
    return {"status": "ok", "action": action, "chunk_id": chunk_id,
            "note": "file mode — no API to notify"}


def load_action_log() -> List[Dict]:
    """Load alert action log from API or return empty for file mode."""
    if _DATA_SOURCE != "api":
        return []
    try:
        import requests
        r = requests.get(f"{_API_URL}/alerts/log", timeout=3)
        if r.status_code == 200:
            return r.json().get("actions", [])
    except Exception as e:
        logger.debug(f"Could not load action log from API: {e}")
    return []


def api_health() -> Optional[Dict]:
    """Return API health dict, or None if unreachable."""
    if _DATA_SOURCE != "api":
        return None
    try:
        import requests
        r = requests.get(f"{_API_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Internal loaders ───────────────────────────────────────────────────────────

def _load_from_file() -> List[Dict]:
    """Read all_decisions.json — original file-pipeline behavior."""
    if not os.path.exists(_FILE_PATH):
        return []
    try:
        with open(_FILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {_FILE_PATH}: {e}")
        return []


def _load_from_api(limit: int = 1000) -> List[Dict]:
    """
    Poll GET /chunks from FastAPI.
    Converts API chunk format to match the file-mode format dashboard expects.
    """
    try:
        import requests
        request_limit = min(limit, 500)
        # In live mode, prefer in-memory recent results so the dashboard shows
        # the current mic session instead of older DB history.
        for source in ("mem", "db"):
            r = requests.get(
                f"{_API_URL}/chunks",
                params={"source": source, "limit": request_limit},
                timeout=3,
            )
            if r.status_code != 200:
                logger.warning(f"API /chunks returned {r.status_code} for source={source}")
                continue

            data   = r.json()
            chunks = data.get("chunks", [])
            if chunks:
                break
        else:
            return []

        # Normalize API format → dashboard-expected format
        normalized = []
        for c in chunks:
            score = c.get("score", {})
            # Dashboard expects "chunk_start" not "chunk_start_sec"
            normalized.append({
                "chunk_id":    c.get("chunk_id"),
                "chunk_start": c.get("chunk_start", 0.0),
                "chunk_index": c.get("chunk_index", 0),
                "session_id":  c.get("session_id"),
                "text":        c.get("text", ""),
                "language_mix":c.get("language_mix", ""),
                "processed_at":c.get("processed_at"),
                "score": {
                    "final_score":        score.get("final_score", 0.0),
                    "zone":               score.get("zone", "GREEN"),
                    "zone_emoji":         score.get("zone_emoji", "🟢"),
                    "severity":           score.get("severity", "LOW"),
                    "auto_alert":         score.get("auto_alert", False),
                    "dominant_emotion":   score.get("dominant_emotion", "neutral"),
                    "emergency_category": score.get("emergency_category", "normal"),
                    "is_emergency":       score.get("is_emergency", False),
                    "incident_id":        score.get("incident_id"),
                    "trend_upgraded":     score.get("trend_upgraded", False),
                    "rising_trend":       score.get("rising_trend", False),
                    "components":         score.get("components", {}),
                },
                "keywords_found": c.get("keywords_found", []),
                "keyword_boost":  c.get("keyword_boost", 0.0),
            })
        return normalized

    except Exception as e:
        logger.error(f"Failed to load chunks from API: {e}")
        return []


def _post_to_api(chunk_id: str, action: str, operator_id: str, reason: str) -> Dict:
    try:
        import requests
        r = requests.post(
            f"{_API_URL}/alerts/{chunk_id}/action",
            json={"action": action, "operator_id": operator_id, "reason": reason},
            timeout=5,
        )
        return r.json() if r.status_code in (200, 409) else {"status": "error", "code": r.status_code}
    except Exception as e:
        return {"status": "error", "detail": str(e)}