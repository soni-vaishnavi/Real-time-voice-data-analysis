"""
api/main.py
============
VoiceGuard FastAPI — Stage 4 + 5

Stage 4: DB initialized at startup, worker writes chunks to SQLite
Stage 5: AlertDispatcher initialized at startup, wired into worker

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.core.config       import validate_config
from pipeline.core.queue_manager import AudioQueue
from pipeline.core.worker       import PipelineWorker, ResultsStore
from pipeline.phase3_analysis.emergency_detector import start_background_loading
from pipeline.phase5_alerts.dispatcher import AlertDispatcher
from db.database import init_db, SessionLocal

from api.routes.audio   import router as audio_router
from api.routes.health  import router as health_router
from api.routes.chunks  import router as chunks_router
from api.routes.alerts  import router as alerts_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
      1. validate_config()          — fail loudly on misconfiguration
      2. init_db()                  — create SQLite tables if not exist (WAL mode)
      3. start_background_loading() — BART starts loading in daemon thread
      4. AlertDispatcher()          — alert coordinator with DB access
      5. ResultsStore()             — in-memory fast read cache
      6. AudioQueue()               — bounded chunk queue
      7. PipelineWorker.start()     — worker thread begins
    """
    logger.info("VoiceGuard API starting up...")

    # 1. Config
    validate_config()
    logger.info("Config validated ✅")

    # 2. Database
    init_db()

    # 3. BART background load
    start_background_loading()
    logger.info("BART background loading started ✅")

    # 4. Alert dispatcher (has DB access for audit log)
    dispatcher             = AlertDispatcher(db_factory=SessionLocal)
    app.state.dispatcher   = dispatcher

    # 5. Results store
    results_store          = ResultsStore(maxlen=500)
    app.state.results      = results_store

    # 6. Queue
    audio_queue            = AudioQueue()
    app.state.queue        = audio_queue

    # 7. Worker
    worker = PipelineWorker(
        audio_queue   = audio_queue,
        results_store = results_store,
        dispatcher    = dispatcher,
        db_factory    = SessionLocal,
    )
    worker.start()
    app.state.worker = worker

    logger.info("VoiceGuard API ready — http://localhost:8000/docs ✅")

    yield

    # Shutdown
    logger.info("Shutting down...")
    worker.stop()
    worker.join(timeout=10)
    logger.info("Worker stopped ✅")


app = FastAPI(
    title       = "VoiceGuard API",
    description = "Real-Time Voice Emergency Detection System",
    version     = "4.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(audio_router,  prefix="/audio",  tags=["Audio"])
app.include_router(chunks_router, prefix="/chunks", tags=["Results"])
app.include_router(alerts_router, prefix="/alerts", tags=["Alerts"])
app.include_router(health_router, prefix="/health", tags=["System"])


@app.get("/", tags=["System"])
def root():
    return {
        "system":  "VoiceGuard",
        "version": "4.0.0",
        "stage":   "Stage 4+5 — SQLite + Alert Dispatcher",
        "docs":    "/docs",
    }