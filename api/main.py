"""
api/main.py
============
VoiceGuard FastAPI — Stage 4 + 5 + 6

Stage 6 addition:
  - Worker created with enable_heavy=True (wav2vec2 on YELLOW/RED chunks)
  - model_size="small" (was "tiny") for better Hinglish accuracy
  - health endpoint now reports wav2vec2 load status

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.core.config       import validate_config, WHISPER_MODEL_SIZE
from pipeline.core.queue_manager import AudioQueue
from pipeline.core.worker       import PipelineWorker, ResultsStore
from pipeline.phase3_analysis.emergency_detector import start_background_loading
from pipeline.phase5_alerts.dispatcher import AlertDispatcher
from db.database import init_db, SessionLocal

from api.routes.audio   import router as audio_router
from api.routes.health  import router as health_router
from api.routes.chunks  import router as chunks_router
from api.routes.alerts  import router as alerts_router

os.makedirs("output", exist_ok=True)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

debug_handler = logging.FileHandler("output/debug.log", mode="a", encoding="utf-8")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
root_logger.addHandler(debug_handler)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VoiceGuard API starting...")

    validate_config()
    logger.info("Config validated ✅")

    init_db()
    logger.info("Database initialized ✅")

    start_background_loading()
    logger.info("BART background loading started ✅")

    dispatcher             = AlertDispatcher(db_factory=SessionLocal)
    app.state.dispatcher   = dispatcher

    results_store          = ResultsStore(maxlen=500)
    app.state.results      = results_store

    audio_queue            = AudioQueue()
    app.state.queue        = audio_queue

    worker = PipelineWorker(
        audio_queue   = audio_queue,
        results_store = results_store,
        dispatcher    = dispatcher,
        db_factory    = SessionLocal,
        model_size    = WHISPER_MODEL_SIZE,   # from config (default "small")
        enable_heavy  = True,                  # Stage 6: wav2vec2 on YELLOW/RED
    )
    worker.start()
    app.state.worker = worker

    logger.info("VoiceGuard API ready — http://localhost:8000/docs ✅")
    logger.info(f"Whisper model: {WHISPER_MODEL_SIZE} | Heavy pipeline: enabled")

    yield

    logger.info("Shutting down...")
    worker.stop()
    worker.join(timeout=10)
    logger.info("Worker stopped ✅")


app = FastAPI(
    title       = "VoiceGuard API",
    description = "Real-Time Voice Emergency Detection System",
    version     = "6.0.0",
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
        "version": "6.0.0",
        "stage":   "Stage 6 — Heavy Pipeline (wav2vec2) Active",
        "docs":    "/docs",
    }