"""
api/routes/health.py
======================
GET /health — System health and model status.
"""

from fastapi import APIRouter, Request
from pipeline.phase3_analysis.emergency_detector import is_bart_ready

router = APIRouter()


@router.get("/")
def health_check(request: Request):
    """
    System health check.

    Returns:
        status:  "ok" | "degraded" | "starting"
        worker:  thread status + chunk count + heartbeat
        models:  which models are loaded and ready
        queue:   current queue depth and stats
    """
    worker = getattr(request.app.state, "worker",  None)
    q      = getattr(request.app.state, "queue",   None)
    store  = getattr(request.app.state, "results", None)

    worker_health = worker.health() if worker else {"running": False}
    queue_stats   = q.stats()       if q      else {}
    total_chunks  = store.count()   if store  else 0

    # Determine overall status
    if not worker_health.get("running"):
        status = "error"
    elif not worker_health.get("models_loaded"):
        status = "starting"    # worker alive but models still loading on first chunk
    elif worker_health.get("heartbeat_stale"):
        status = "degraded"
    else:
        status = "ok"

    return {
        "status":       status,
        "worker":       worker_health,
        "models": {
            "bart_ready":    is_bart_ready(),
            "whisper_ready": worker_health.get("models_loaded", False),
            "emotion_ready": worker_health.get("models_loaded", False),
            "sarcasm_rules": True,   # always ready — no model load needed
        },
        "queue":        queue_stats,
        "results_count": total_chunks,
    }