"""
pipeline/phase5_alerts/dispatcher.py
=======================================
STAGE 5 — Alert Dispatcher

Central controller for all alert firing logic.
Wired into the worker's on_red_alert callback and the /alerts/{id}/action route.

Alert rules:
  1 RED (CRITICAL/HIGH severity)   → fire immediately (SMS + email + sound)
  1 RED (MEDIUM severity)          → sound only + show on dashboard (operator has 60s)
  2 RED in same session < 120s     → force fire regardless of severity
  Manual CONFIRM via API           → fire immediately
  Manual REJECT via API            → log as false alarm, no alerts

Deduplication:
  Each incident_id is fired at most once per session.
  Confirmed/rejected incidents are tracked in the DB and cannot be re-actioned.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Delay import of alert functions — they may not be installed in all environments
_sms_available   = None
_email_available = None


def _get_sms():
    global _sms_available
    if _sms_available is None:
        try:
            from pipeline.phase5_alerts.sms_alert import send_emergency_sms
            _sms_available = send_emergency_sms
        except ImportError:
            _sms_available = False
    return _sms_available


def _get_email():
    global _email_available
    if _email_available is None:
        try:
            from pipeline.phase5_alerts.email_alert import send_emergency_email
            _email_available = send_emergency_email
        except ImportError:
            _email_available = False
    return _email_available


def _play_sound(zone: str):
    try:
        from pipeline.phase5_alerts.sound_alert import trigger_for_zone
        trigger_for_zone(zone)
    except Exception as e:
        logger.debug(f"Sound alert unavailable: {e}")


class AlertDispatcher:
    """
    Thread-safe alert dispatcher.

    One instance per FastAPI app lifetime.
    Created in api/main.py lifespan, stored in app.state.dispatcher.

    Args:
        db_factory: SQLAlchemy SessionLocal (for writing audit log to DB)
                    If None, audit log is only in-memory (session state).
    """

    # How close two RED chunks must be to count as "same incident" for auto-fire
    SAME_INCIDENT_SEC = 120

    def __init__(self, db_factory=None):
        self._db_factory   = db_factory
        self._lock         = threading.Lock()

        # Track incident state per session: session_id → {incident_id, first_red_time, count}
        self._session_state: dict = {}

        # Global dedup: set of incident_ids that have already been fired
        self._fired_set: set = set()

        # In-memory action log (mirrors DB for dashboard reads)
        self.action_log: list = []

    # ── Main entry points ──────────────────────────────────────────────────────

    def handle_red(self, chunk_result: dict) -> None:
        """
        Called by worker thread whenever a chunk scores RED.
        Decides whether to fire immediately or wait for operator.
        """
        score      = chunk_result.get("score", {})
        session_id = chunk_result.get("session_id", "default")
        severity   = score.get("severity",    "HIGH")
        auto_alert = score.get("auto_alert",  False)
        incident_id = self._assign_incident(chunk_result)

        # Always play alarm sound on RED
        _play_sound("RED")
        logger.warning(f"RED detected | incident={incident_id} | "
                       f"severity={severity} | auto_alert={auto_alert}")

        # Track RED count for this session incident
        with self._lock:
            state = self._session_state.setdefault(session_id, {})
            istate = state.setdefault(incident_id, {
                "first_red_time": datetime.utcnow(),
                "count": 0,
            })
            istate["count"] += 1
            count = istate["count"]

        # Fire rules
        if incident_id in self._fired_set:
            return   # already fired for this incident

        should_fire = False
        fire_reason = "AUTO"

        if auto_alert and severity in ("CRITICAL", "HIGH"):
            # Immediate fire — CRITICAL/HIGH always auto-alert
            should_fire = True
            fire_reason = "AUTO"

        elif count >= 2:
            # Second RED in same incident → force fire even for MEDIUM
            elapsed = (datetime.utcnow() - istate["first_red_time"]).total_seconds()
            if elapsed <= self.SAME_INCIDENT_SEC:
                should_fire = True
                fire_reason = "AUTO_2RED"

        if should_fire:
            self._fire(chunk_result, reason=fire_reason)

    def confirm(self, chunk_result: dict, operator_id: str = "operator", reason: str = "") -> bool:
        """
        Called by /alerts/{id}/action with action=CONFIRMED.
        Fires SMS + email immediately.
        Returns False if already fired.
        """
        incident_id = chunk_result.get("score", {}).get("incident_id") or chunk_result.get("chunk_id")
        if incident_id in self._fired_set:
            logger.info(f"Confirm ignored — already fired: {incident_id}")
            return False

        self._fire(chunk_result, reason="CONFIRMED", operator_id=operator_id, note=reason)
        return True

    def reject(self, chunk_result: dict, operator_id: str = "operator", reason: str = "") -> None:
        """
        Called by /alerts/{id}/action with action=REJECTED.
        Stops alarm and logs false alarm.
        """
        incident_id = chunk_result.get("score", {}).get("incident_id") or chunk_result.get("chunk_id")

        # Stop alarm
        try:
            from pipeline.phase5_alerts.sound_alert import stop_alarm
            stop_alarm()
        except Exception:
            pass

        self._log_action(
            chunk_result  = chunk_result,
            action        = "REJECTED",
            operator_id   = operator_id,
            reason        = reason or "false_alarm",
            sms_sent      = False,
            email_sent    = False,
            sound_played  = False,
        )
        # Mark as fired so it cannot be re-confirmed
        self._fired_set.add(incident_id)
        logger.info(f"Alert REJECTED by {operator_id} | incident={incident_id}")

    # ── Internal ────────────────────────────────────────────────────────────────

    def _fire(
        self,
        chunk_result: dict,
        reason:       str = "AUTO",
        operator_id:  str = "system",
        note:         str = "",
    ) -> None:
        """Fire SMS + email + log. Thread-safe, deduplicated."""
        incident_id = chunk_result.get("score", {}).get("incident_id") or chunk_result.get("chunk_id")

        with self._lock:
            if incident_id in self._fired_set:
                return
            self._fired_set.add(incident_id)

        sms_sent = email_sent = False

        # SMS
        sms_fn = _get_sms()
        if sms_fn:
            try:
                result = sms_fn(chunk_result)
                sms_sent = result.get("sent", False)
                logger.info(f"SMS {'sent' if sms_sent else 'dry-run'} | incident={incident_id}")
            except Exception as e:
                logger.error(f"SMS failed: {e}")

        # Email
        email_fn = _get_email()
        if email_fn:
            try:
                result = email_fn(chunk_result)
                email_sent = result.get("sent", False)
                logger.info(f"Email {'sent' if email_sent else 'dry-run'} | incident={incident_id}")
            except Exception as e:
                logger.error(f"Email failed: {e}")

        self._log_action(
            chunk_result = chunk_result,
            action       = f"ALERT_FIRED",
            operator_id  = operator_id,
            reason       = reason + (f" — {note}" if note else ""),
            sms_sent     = sms_sent,
            email_sent   = email_sent,
            sound_played = True,
        )

        logger.warning(
            f"🚨 Alert FIRED | incident={incident_id} | reason={reason} | "
            f"sms={sms_sent} | email={email_sent}"
        )

    def _assign_incident(self, chunk_result: dict) -> str:
        """
        Assign or create incident_id for a RED chunk.
        Groups RED chunks that arrive close together in the same session.
        Modifies chunk_result["score"]["incident_id"] in place.
        """
        score      = chunk_result.get("score", {})
        session_id = chunk_result.get("session_id", "default")

        # If already has an incident_id (from trend_analyzer in file mode), use it
        if score.get("incident_id"):
            return score["incident_id"]

        with self._lock:
            state = self._session_state.setdefault(session_id, {})
            # Find the latest open incident for this session
            latest = None
            for inc_id, inc_state in state.items():
                last_time = inc_state.get("last_time", inc_state.get("first_red_time"))
                if (datetime.utcnow() - last_time).total_seconds() <= self.SAME_INCIDENT_SEC:
                    latest = inc_id
                    break

            if latest:
                incident_id = latest
                state[latest]["last_time"] = datetime.utcnow()
            else:
                # New incident — find highest existing number
                existing_nums = []
                for inc_id in state:
                    try:
                        existing_nums.append(int(inc_id.replace("INC_", "")))
                    except Exception:
                        pass
                # Also check DB for global uniqueness
                if self._db_factory:
                    try:
                        with self._db_factory() as db:
                            from db.models.incident import Incident
                            rows = db.query(Incident.id).all()
                            for (r,) in rows:
                                try:
                                    existing_nums.append(int(r.replace("INC_", "")))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                next_num    = (max(existing_nums) + 1) if existing_nums else 1
                incident_id = f"INC_{next_num:03d}"
                state[incident_id] = {
                    "first_red_time": datetime.utcnow(),
                    "last_time":      datetime.utcnow(),
                    "count":          0,
                }

        score["incident_id"] = incident_id

        # Create or update incident in DB
        if self._db_factory:
            self._upsert_incident(chunk_result, incident_id)

        return incident_id

    def _upsert_incident(self, chunk_result: dict, incident_id: str) -> None:
        """Create or update incident record in DB."""
        try:
            with self._db_factory() as db:
                from db.models.incident import Incident
                inc = db.query(Incident).filter(Incident.id == incident_id).first()
                score = chunk_result.get("score", {})

                if inc is None:
                    inc = Incident(
                        id            = incident_id,
                        session_id    = chunk_result.get("session_id", "default"),
                        category      = score.get("emergency_category", "unknown"),
                        severity      = score.get("severity", "HIGH"),
                        peak_score    = score.get("final_score", 0.0),
                        first_chunk_id= chunk_result.get("chunk_id"),
                        latest_chunk_id=chunk_result.get("chunk_id"),
                        latest_text   = (chunk_result.get("text") or "")[:200],
                        status        = "open",
                    )
                    db.add(inc)
                else:
                    inc.red_chunk_count  = (inc.red_chunk_count or 0) + 1
                    inc.latest_chunk_id  = chunk_result.get("chunk_id")
                    inc.latest_text      = (chunk_result.get("text") or "")[:200]
                    if score.get("final_score", 0.0) > (inc.peak_score or 0.0):
                        inc.peak_score = score.get("final_score", 0.0)
                db.commit()
        except Exception as e:
            logger.error(f"DB incident upsert failed: {e}")

    def _log_action(
        self,
        chunk_result: dict,
        action:       str,
        operator_id:  str,
        reason:       str,
        sms_sent:     bool,
        email_sent:   bool,
        sound_played: bool,
    ) -> None:
        """Write action to in-memory log and DB."""
        score    = chunk_result.get("score", {})
        entry    = {
            "incident_id": score.get("incident_id") or chunk_result.get("chunk_id"),
            "chunk_id":    chunk_result.get("chunk_id"),
            "session_id":  chunk_result.get("session_id"),
            "action":      action,
            "operator_id": operator_id,
            "reason":      reason,
            "category":    score.get("emergency_category", "?"),
            "severity":    score.get("severity", "?"),
            "final_score": f"{score.get('final_score', 0.0):.3f}",
            "zone":        score.get("zone", "?"),
            "text_preview":(chunk_result.get("text") or "")[:120],
            "sms_sent":    sms_sent,
            "email_sent":  email_sent,
            "sound_played":sound_played,
            "timestamp":   datetime.utcnow().isoformat(),
        }

        # In-memory log (for fast reads by dashboard)
        self.action_log.append(entry)
        # Keep last 200 entries in memory
        if len(self.action_log) > 200:
            self.action_log = self.action_log[-200:]

        # Persist to DB
        if self._db_factory:
            try:
                with self._db_factory() as db:
                    from db.models.alert_action import AlertAction
                    db.add(AlertAction(**{k: v for k, v in entry.items()
                                         if k != "timestamp"}))
                    db.commit()
            except Exception as e:
                logger.error(f"DB alert_action write failed: {e}")