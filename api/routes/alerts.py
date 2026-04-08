"""
api/routes/alerts.py
======================
GET  /alerts           — open RED incidents
GET  /alerts/history   — all RED + YELLOW
GET  /alerts/log       — operator action audit log
POST /alerts/{id}/action — CONFIRM (fires alerts) or REJECT (false alarm)

Stage 5: CONFIRM/REJECT now fires AlertDispatcher instead of in-memory state.
Stage 5: All actions persisted to DB via AlertDispatcher._log_action().
"""

import threading
from datetime import datetime
from typing import Literal, Optional, List

from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel

from db.database import SessionLocal

router = APIRouter()


class ActionRequest(BaseModel):
    action:      Literal["CONFIRMED", "REJECTED"]
    operator_id: str = "operator_1"
    reason:      str = ""


@router.get("/")
def get_active_alerts(
    request: Request,
    limit:   int = Query(default=20, ge=1, le=100),
):
    """Open RED incidents not yet confirmed or dismissed."""
    # Read from DB
    open_alerts = _get_open_incidents_from_db()
    if open_alerts is not None:
        return {
            "total_red":   len(open_alerts),
            "open_alerts": len(open_alerts),
            "source":      "db",
            "alerts":      open_alerts[-limit:],
        }

    # Fallback: in-memory RED chunks
    store = getattr(request.app.state, "results", None)
    if store is None:
        return {"alerts": []}

    dispatcher = getattr(request.app.state, "dispatcher", None)
    fired_set  = set(dispatcher._fired_set) if dispatcher else set()

    red_chunks = store.get_by_zone("RED")
    open_alerts = [
        _alert_summary(c)
        for c in red_chunks
        if c.get("score", {}).get("incident_id") not in fired_set
    ]
    return {
        "total_red":   len(red_chunks),
        "open_alerts": len(open_alerts),
        "source":      "memory",
        "alerts":      open_alerts[-limit:],
    }


@router.get("/history")
def get_alert_history(
    limit: int           = Query(default=100, ge=1, le=500),
    zone:  Optional[str] = Query(default=None),
):
    """All RED + YELLOW chunks — full incident history from DB."""
    try:
        with SessionLocal() as db:
            from db.models.chunk_model import Chunk
            q = db.query(Chunk).filter(Chunk.zone.in_(["RED", "YELLOW"]))
            if zone:
                q = q.filter(Chunk.zone == zone.upper())
            rows = q.order_by(Chunk.created_at.desc()).limit(limit).all()
            return {
                "total":   len(rows),
                "source":  "db",
                "history": [r.to_dict() for r in reversed(rows)],
            }
    except Exception as e:
        return {"error": str(e), "history": []}


@router.get("/log")
def get_action_log(limit: int = Query(default=50, ge=1, le=200), request: Request = None):
    """Operator action audit log."""
    # Try DB first
    try:
        with SessionLocal() as db:
            from db.models.alert_action import AlertAction
            rows = (
                db.query(AlertAction)
                .order_by(AlertAction.created_at.desc())
                .limit(limit)
                .all()
            )
            return {
                "source": "db",
                "actions": [
                    {
                        "id":          r.id,
                        "incident_id": r.incident_id,
                        "chunk_id":    r.chunk_id,
                        "action":      r.action,
                        "operator_id": r.operator_id,
                        "reason":      r.reason,
                        "category":    r.category,
                        "severity":    r.severity,
                        "final_score": r.final_score,
                        "sms_sent":    r.sms_sent,
                        "email_sent":  r.email_sent,
                        "text_preview":r.text_preview,
                        "timestamp":   r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ],
            }
    except Exception:
        pass

    # Fallback: in-memory
    dispatcher = getattr(request.app.state, "dispatcher", None) if request else None
    if dispatcher:
        return {"source": "memory", "actions": list(reversed(dispatcher.action_log[-limit:]))}
    return {"source": "none", "actions": []}


@router.post("/{chunk_id}/action")
def take_action(
    chunk_id: str,
    body:     ActionRequest,
    request:  Request,
):
    """
    Operator confirms or rejects an alert.

    CONFIRMED → fires SMS + email via dispatcher, marks incident confirmed
    REJECTED  → logs false alarm, stops alarm, marks incident dismissed
    """
    dispatcher = getattr(request.app.state, "dispatcher", None)
    store      = getattr(request.app.state, "results",    None)

    if dispatcher is None:
        raise HTTPException(status_code=503, detail="Dispatcher not initialized")

    # Find chunk
    chunk = None
    if store:
        all_c = store.get_all()
        chunk = next((c for c in all_c if c.get("chunk_id") == chunk_id), None)

    # Try DB if not in memory
    if chunk is None:
        try:
            with SessionLocal() as db:
                from db.models.chunk_model import Chunk
                row = db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()
                if row:
                    chunk = row.to_dict()
        except Exception:
            pass

    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")

    incident_id = chunk.get("score", {}).get("incident_id") or chunk_id

    # Check already actioned
    if incident_id in dispatcher._fired_set:
        raise HTTPException(status_code=409, detail=f"Incident {incident_id} already actioned")

    if body.action == "CONFIRMED":
        fired = dispatcher.confirm(
            chunk_result = chunk,
            operator_id  = body.operator_id,
            reason       = body.reason,
        )
        # Update incident status in DB
        _update_incident_status(incident_id, "confirmed")
        return {
            "status":      "ok",
            "action":      "CONFIRMED",
            "incident_id": incident_id,
            "alert_fired": fired,
            "operator":    body.operator_id,
            "timestamp":   datetime.utcnow().isoformat(),
        }
    else:
        dispatcher.reject(
            chunk_result = chunk,
            operator_id  = body.operator_id,
            reason       = body.reason,
        )
        _update_incident_status(incident_id, "dismissed")
        return {
            "status":      "ok",
            "action":      "REJECTED",
            "incident_id": incident_id,
            "operator":    body.operator_id,
            "timestamp":   datetime.utcnow().isoformat(),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_open_incidents_from_db():
    """Return list of open incident summaries from DB."""
    try:
        with SessionLocal() as db:
            from db.models.incident import Incident
            rows = (
                db.query(Incident)
                .filter(Incident.status == "open")
                .order_by(Incident.created_at.desc())
                .all()
            )
            return [
                {
                    "incident_id":  r.id,
                    "session_id":   r.session_id,
                    "category":     r.category,
                    "severity":     r.severity,
                    "peak_score":   r.peak_score,
                    "red_count":    r.red_chunk_count,
                    "status":       r.status,
                    "latest_text":  r.latest_text,
                    "created_at":   r.created_at.isoformat() if r.created_at else None,
                    "first_chunk_id": r.first_chunk_id,
                }
                for r in rows
            ]
    except Exception:
        return None


def _update_incident_status(incident_id: str, status: str) -> None:
    try:
        with SessionLocal() as db:
            from db.models.incident import Incident
            inc = db.query(Incident).filter(Incident.id == incident_id).first()
            if inc:
                inc.status      = status
                inc.resolved_at = datetime.utcnow()
                db.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Incident status update failed: {e}")


def _alert_summary(chunk: dict) -> dict:
    score = chunk.get("score", {})
    return {
        "chunk_id":          chunk.get("chunk_id"),
        "session_id":        chunk.get("session_id"),
        "chunk_start":       chunk.get("chunk_start"),
        "text":              chunk.get("text", ""),
        "processed_at":      chunk.get("processed_at"),
        "final_score":       score.get("final_score", 0.0),
        "zone":              score.get("zone", "?"),
        "zone_emoji":        score.get("zone_emoji", ""),
        "severity":          score.get("severity", "?"),
        "auto_alert":        score.get("auto_alert", False),
        "dominant_emotion":  score.get("dominant_emotion", "?"),
        "emergency_category":score.get("emergency_category", "?"),
        "incident_id":       score.get("incident_id"),
        "status":            "open",
    }