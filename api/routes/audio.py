"""
api/routes/audio.py
=====================
POST /audio/submit — Submit an audio chunk for processing.

Accepts a WAV file upload, saves it to output/chunks/, and pushes
metadata to the pipeline queue. Returns immediately — processing is
asynchronous in the worker thread.
"""

import os
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from typing import Optional

from pipeline.core.config import CHUNKS_DIR

router = APIRouter()


@router.post("/submit")
async def submit_audio(
    request:     Request,
    file:        UploadFile  = File(...),
    session_id:  Optional[str] = Form(default=None),
    chunk_index: int           = Form(default=0),
    chunk_start: float         = Form(default=0.0),
):
    """
    Submit an audio chunk for pipeline processing.

    Args (multipart form):
        file:        WAV audio file (required)
        session_id:  Session identifier (optional, auto-generated if absent)
        chunk_index: Position of this chunk in the session (0-based)
        chunk_start: Start time of chunk in seconds from session start

    Returns:
        {
            "status":     "queued" | "duplicate" | "queue_full",
            "chunk_id":   "chunk_20260401_103045_0001",
            "session_id": "session_20260401_103040",
            "queue_size": 3
        }
    """
    # Validate file type
    filename = file.filename or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files accepted")

    audio_bytes = await file.read()
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio file too small")

    # Save to chunks directory
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_id   = f"chunk_{ts}_{chunk_index:04d}"
    chunk_path = str(CHUNKS_DIR / f"{chunk_id}.wav")

    os.makedirs(str(CHUNKS_DIR), exist_ok=True)
    with open(chunk_path, "wb") as f:
        f.write(audio_bytes)

    # Build queue item
    sid  = session_id or f"session_{ts}"
    item = {
        "chunk_id":    chunk_id,
        "audio_path":  chunk_path,
        "chunk_start": chunk_start,
        "chunk_index": chunk_index,
        "session_id":  sid,
    }

    # Push to queue
    q = getattr(request.app.state, "queue", None)
    if q is None:
        raise HTTPException(status_code=503, detail="Queue not initialized")

    queued = q.put(item)

    if not queued:
        # Duplicate or queue full
        stats = q.stats()
        if stats["current_size"] >= stats["maxsize"]:
            return {"status": "queue_full",  "chunk_id": chunk_id,
                    "session_id": sid, "queue_size": q.qsize()}
        return     {"status": "duplicate",   "chunk_id": chunk_id,
                    "session_id": sid, "queue_size": q.qsize()}

    return {"status": "queued", "chunk_id": chunk_id,
            "session_id": sid,  "queue_size": q.qsize()}


@router.get("/queue-status")
def queue_status(request: Request):
    """Quick queue size check — used by mic capture to detect back-pressure."""
    q = getattr(request.app.state, "queue", None)
    if q is None:
        return {"error": "queue not initialized"}
    return q.stats()