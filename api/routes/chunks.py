"""
api/routes/chunks.py
======================
GET /chunks         — recent analyzed chunks (DB → in-memory fallback)
GET /chunks/summary — zone distribution stats
GET /chunks/session/{session_id} — all chunks for one session
"""

from fastapi import APIRouter, Request, Query, Depends
from typing import Optional

from db.database import get_db, SessionLocal

router = APIRouter()


@router.get("/")
def get_chunks(
    request: Request,
    zone:    Optional[str] = Query(default=None),
    limit:   int           = Query(default=50, ge=1, le=500),
    session: Optional[str] = Query(default=None),
    source:  str           = Query(default="db", description="db | memory"),
):
    """
    Return recent analyzed chunks.
    source=db  → read from SQLite (persistent, survives restarts)
    source=mem → read from in-memory ResultsStore (faster for live view)
    """
    if source == "db" or source == "both":
        chunks = _from_db(zone=zone, limit=limit, session=session)
        if chunks:
            return {"total": len(chunks), "source": "db", "chunks": chunks}

    # Fallback to in-memory store
    store = getattr(request.app.state, "results", None)
    if store is None:
        return {"chunks": [], "error": "results store not initialized"}

    all_c = store.get_by_zone(zone.upper()) if zone else store.get_all()
    if session:
        all_c = [c for c in all_c if c.get("session_id") == session]
    all_c = all_c[-limit:]

    return {
        "total":  store.count(),
        "source": "memory",
        "chunks": [_serialize(c) for c in all_c],
    }


@router.get("/summary")
def get_summary(request: Request):
    """Zone distribution and session stats."""
    # Try DB first
    try:
        with SessionLocal() as db:
            from db.models.chunk_model import Chunk
            from sqlalchemy import func
            rows = db.query(
                Chunk.zone, func.count(Chunk.id).label("cnt")
            ).group_by(Chunk.zone).all()

            dist  = {row.zone: row.cnt for row in rows}
            total = sum(dist.values())
            scores= db.query(func.avg(Chunk.final_score)).scalar() or 0.0

            return {
                "total_chunks":  total,
                "green":         dist.get("GREEN",  0),
                "yellow":        dist.get("YELLOW", 0),
                "red":           dist.get("RED",    0),
                "avg_score_pct": round(float(scores) * 100, 1),
                "source":        "db",
            }
    except Exception:
        pass

    # Fallback to in-memory
    store = getattr(request.app.state, "results", None)
    if not store:
        return {"error": "no data available"}
    all_c  = store.get_all()
    zones  = [c.get("score", {}).get("zone", "GREEN") for c in all_c]
    scores = [c.get("score", {}).get("final_score", 0.0) for c in all_c]
    avg    = round(sum(scores) / max(1, len(scores)) * 100, 1)
    return {
        "total_chunks":  len(all_c),
        "green":         zones.count("GREEN"),
        "yellow":        zones.count("YELLOW"),
        "red":           zones.count("RED"),
        "avg_score_pct": avg,
        "source":        "memory",
    }


@router.get("/session/{session_id}")
def get_session_chunks(session_id: str, limit: int = Query(default=200, ge=1, le=1000)):
    """All chunks for a specific session, from DB."""
    try:
        with SessionLocal() as db:
            from db.models.chunk_model import Chunk
            rows = (
                db.query(Chunk)
                .filter(Chunk.session_id == session_id)
                .order_by(Chunk.chunk_index)
                .limit(limit)
                .all()
            )
            return {"session_id": session_id, "total": len(rows),
                    "chunks": [r.to_dict() for r in rows]}
    except Exception as e:
        return {"error": str(e), "chunks": []}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _from_db(zone=None, limit=50, session=None):
    """Read chunks from SQLite."""
    try:
        with SessionLocal() as db:
            from db.models.chunk_model import Chunk
            q = db.query(Chunk)
            if zone:
                q = q.filter(Chunk.zone == zone.upper())
            if session:
                q = q.filter(Chunk.session_id == session)
            rows = q.order_by(Chunk.created_at.desc()).limit(limit).all()
            return [r.to_dict() for r in reversed(rows)]
    except Exception:
        return []


def _serialize(chunk: dict) -> dict:
    """Slim down a raw worker result dict for API response."""
    score = chunk.get("score", {})
    kw    = chunk.get("keyword_analysis", {})
    return {
        "chunk_id":      chunk.get("chunk_id"),
        "session_id":    chunk.get("session_id"),
        "chunk_index":   chunk.get("chunk_index"),
        "chunk_start":   chunk.get("chunk_start"),
        "text":          chunk.get("text", ""),
        "language_mix":  chunk.get("language_mix", ""),
        "processed_at":  chunk.get("processed_at"),
        "processing_ms": chunk.get("processing_ms"),
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
        "keywords_found": kw.get("keywords_list", []),
        "keyword_boost":  kw.get("total_boost", 0.0),
        "bart_used":      chunk.get("emergency_analysis", {}).get("bart_used", False),
    }