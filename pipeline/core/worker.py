"""
pipeline/core/worker.py
=========================
STAGE 4 + 5 — Pipeline Worker (Final Light Pipeline Version)

Stage 4 additions over Stage 3:
  - Writes each processed chunk to SQLite (db.models.chunk_model.Chunk)
  - Creates/updates Session row for the current session
  - DB write happens after scoring, before alert dispatch

Stage 5 additions over Stage 4:
  - on_red_alert → AlertDispatcher.handle_red()
  - AlertDispatcher fires SMS + email + sound on RED
  - Deduplication: each incident fires at most once

One thread only. Sequential. Models cached after first chunk.
"""

import json
import queue
import threading
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ResultsStore:
    """Thread-safe in-memory results cache. Fast reads for API routes."""

    def __init__(self, maxlen: int = 500):
        self._results: List[Dict] = []
        self._lock   = threading.Lock()
        self._maxlen = maxlen

    def append(self, result: Dict) -> None:
        with self._lock:
            self._results.append(result)
            if len(self._results) > self._maxlen:
                self._results = self._results[-self._maxlen:]

    def get_all(self) -> List[Dict]:
        with self._lock:
            return list(self._results)

    def get_by_zone(self, zone: str) -> List[Dict]:
        with self._lock:
            return [r for r in self._results if r.get("score", {}).get("zone") == zone]

    def get_recent(self, n: int) -> List[Dict]:
        with self._lock:
            return list(self._results[-n:])

    def clear(self) -> None:
        with self._lock:
            self._results = []

    def count(self) -> int:
        with self._lock:
            return len(self._results)


class PipelineWorker(threading.Thread):
    """
    Daemon thread. Full light pipeline + DB writes + alert dispatch.

    Args:
        audio_queue:    AudioQueue instance
        results_store:  ResultsStore (shared with API routes for fast reads)
        dispatcher:     AlertDispatcher (Stage 5). None in Stage 3/4.
        db_factory:     SQLAlchemy SessionLocal. None in Stage 2/3.
    """

    def __init__(
        self,
        audio_queue:   object,
        results_store: ResultsStore,
        dispatcher:    Optional[object] = None,
        db_factory:    Optional[object] = None,
    ):
        super().__init__(daemon=True, name="PipelineWorker")
        self.queue         = audio_queue
        self.results_store = results_store
        self.dispatcher    = dispatcher
        self.db_factory    = db_factory

        self._stop_event     = threading.Event()
        self.chunks_processed = 0
        self.last_error: Optional[str]   = None
        self.last_beat:  datetime        = datetime.utcnow()

        # Models — cached after _load_models() on first chunk
        self._whisper       = None
        self._models_loaded = False

        # Session tracking for DB
        self._session_id:    Optional[str] = None
        self._session_start: Optional[datetime] = None

    # ── LIFECYCLE ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("Pipeline worker started")

        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except Exception:
                continue

            try:
                result = self._process_item(item)
                if result:
                    self.results_store.append(result)
                    self.chunks_processed += 1
                    self.last_beat = datetime.utcnow()
                    self._log_result(result)

                    # Stage 4: write to DB
                    if self.db_factory:
                        self._write_chunk_to_db(result)
                        self._update_session_stats(result)

                    # Stage 5: alert dispatch on RED
                    if (result.get("score", {}).get("zone") == "RED"
                            and self.dispatcher is not None):
                        try:
                            self.dispatcher.handle_red(result)
                        except Exception as e:
                            logger.error(f"Dispatcher error: {e}")

            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Worker error on {item.get('chunk_id','?')}: {e}")
                logger.debug(traceback.format_exc())
            finally:
                try:
                    self.queue.task_done()
                except Exception:
                    pass

        logger.info(f"Worker stopped | total={self.chunks_processed}")

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    def set_session(self, session_id: str) -> None:
        """Set active session. Called when mic capture starts a new session."""
        self._session_id    = session_id
        self._session_start = datetime.utcnow()
        if self.db_factory:
            self._create_session_row(session_id)

    def health(self) -> Dict:
        stale = (datetime.utcnow() - self.last_beat).total_seconds()
        return {
            "running":           self.is_running(),
            "chunks_processed":  self.chunks_processed,
            "last_beat_sec_ago": round(stale, 1),
            "heartbeat_stale":   stale > 30 and self.chunks_processed > 0,
            "last_error":        self.last_error,
            "queue_size":        self.queue.qsize(),
            "models_loaded":     self._models_loaded,
        }

    # ── PIPELINE ──────────────────────────────────────────────────────────────

    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Full light pipeline for one audio chunk."""
        import os
        chunk_id    = item.get("chunk_id",   "unknown")
        audio_path  = item.get("audio_path", "")
        chunk_start = item.get("chunk_start", 0.0)
        chunk_index = item.get("chunk_index", 0)
        session_id  = item.get("session_id", self._session_id or "default")

        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return None

        if not self._models_loaded:
            self._load_models()

        t_start = datetime.utcnow()

        # 1. Transcribe
        transcript  = self._transcribe(audio_path, chunk_id, chunk_start)
        text        = transcript.get("text", "")

        # 2. Keywords
        from pipeline.phase2_stt.keyword_normalizer import apply_keyword_normalization
        transcript  = apply_keyword_normalization(transcript)
        kw_analysis = transcript.get("keyword_analysis", {})

        # 3. Emotion + sarcasm
        from pipeline.phase3_analysis.emotion_detector import (
            detect_emotion, resolve_sarcasm_conflict
        )
        from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
        text_emotion   = detect_emotion(text)
        sarcasm_result = detect_sarcasm(text)
        sarcasm_res    = resolve_sarcasm_conflict(text_emotion, sarcasm_result)

        # 4. Emergency (BART if ready, keyword fallback if not)
        from pipeline.phase3_analysis.emergency_detector import detect_emergency
        from pipeline.core.config import MIN_WORDS_FOR_EMERGENCY, EMERGENCY_THRESHOLD

        word_count = len(text.split()) if text else 0
        if word_count >= MIN_WORDS_FOR_EMERGENCY:
            emergency = detect_emergency(text, wait_for_model=False)
            # Keyword boost
            kw_boost = kw_analysis.get("total_boost", 0.0)
            kw_cat   = kw_analysis.get("top_category", None)
            if kw_boost > 0 and kw_cat and kw_cat in emergency.get("all_scores", {}):
                boosted = min(emergency["all_scores"][kw_cat] + kw_boost, 1.0)
                emergency["all_scores"][kw_cat] = round(boosted, 4)
                new_top = max(emergency["all_scores"], key=emergency["all_scores"].get)
                emergency.update({
                    "top_category": new_top,
                    "top_score":    emergency["all_scores"][new_top],
                    "is_emergency": (new_top != "normal"
                                     and emergency["all_scores"][new_top] >= EMERGENCY_THRESHOLD),
                })
        else:
            emergency = {
                "top_category": "normal", "top_score": 0.0,
                "is_emergency": False, "all_scores": {}, "risk_level": 0.0,
                "bart_used": False, "skipped": f"too short ({word_count} words)",
            }

        # 5. Score + zone
        from pipeline.phase4_decision.scorer import compute_score
        from pipeline.phase4_decision.zone_classifier import classify_zone

        chunk_data = {
            "chunk_id":    chunk_id,
            "chunk_start": chunk_start,
            "chunk_index": chunk_index,
            "session_id":  session_id,
            "audio_path":  audio_path,
            "text":        text,
            "language_mix": transcript.get("language_mix", ""),
            "emotion_analysis": {
                "emotion":            text_emotion,
                "sarcasm":            sarcasm_result,
                "sarcasm_resolution": sarcasm_res,
            },
            "emergency_analysis": emergency,
            "keyword_analysis":   kw_analysis,
        }
        chunk_data["score"] = compute_score(chunk_data)
        classify_zone(chunk_data["score"])

        elapsed_ms = int((datetime.utcnow() - t_start).total_seconds() * 1000)
        chunk_data["processing_ms"] = elapsed_ms
        chunk_data["processed_at"]  = datetime.utcnow().isoformat()

        return chunk_data

    def _transcribe(self, audio_path: str, chunk_id: str, chunk_start: float) -> Dict:
        try:
            segments_gen, info = self._whisper.transcribe(
                audio_path,
                language                   = "en",
                beam_size                  = 3,
                word_timestamps            = True,
                vad_filter                 = True,
                condition_on_previous_text = False,
                no_speech_threshold        = 0.6,
            )
            segments  = list(segments_gen)
            full_text = " ".join(s.text.strip() for s in segments).strip()

            from pipeline.phase2_stt.whisper_transcriber import is_hallucination
            if is_hallucination(full_text):
                full_text = ""

            return {
                "chunk_id":             chunk_id,
                "chunk_start":          chunk_start,
                "text":                 full_text,
                "language_mix":         getattr(info, "language", "en"),
                "detection_confidence": round(getattr(info, "language_probability", 0.0), 3),
            }
        except Exception as e:
            logger.error(f"Transcription failed for {chunk_id}: {e}")
            return {"chunk_id": chunk_id, "chunk_start": chunk_start, "text": "", "language_mix": ""}

    def _load_models(self) -> None:
        """Load Whisper + emotion model once on first chunk."""
        from pipeline.core.config import WHISPER_MODEL_SIZE

        # Suppress HuggingFace load report warnings (cosmetic noise)
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass

        logger.info("Worker: loading Whisper (first-chunk model load)...")
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(
            WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
        )
        logger.info(f"Whisper {WHISPER_MODEL_SIZE} ready ✅")

        from pipeline.phase3_analysis.emotion_detector import get_emotion_model
        get_emotion_model()
        logger.info("Emotion model ready ✅")

        self._models_loaded = True
        logger.info("Worker: all light models loaded — pipeline operational ✅")

    def _log_result(self, result: Dict) -> None:
        score = result.get("score", {})
        logger.info(
            f"[Worker] {score.get('zone_emoji','')} {score.get('zone','?')} | "
            f"score={score.get('final_score',0):.3f} | "
            f"emotion={score.get('dominant_emotion','?')} | "
            f"cat={score.get('emergency_category','?')} | "
            f"time={result.get('processing_ms',0)}ms | "
            f"total={self.chunks_processed} | "
            f"text=\"{(result.get('text') or '')[:55]}\""
        )

    # ── DB WRITES (Stage 4) ────────────────────────────────────────────────────

    def _write_chunk_to_db(self, result: Dict) -> None:
        """Persist analyzed chunk to SQLite."""
        try:
            with self.db_factory() as db:
                from db.models.chunk_model import Chunk
                score = result.get("score", {})
                kw    = result.get("keyword_analysis", {})
                comp  = score.get("components", {})

                chunk = Chunk(
                    chunk_id          = result.get("chunk_id"),
                    session_id        = result.get("session_id", "default"),
                    chunk_index       = result.get("chunk_index", 0),
                    chunk_start_sec   = result.get("chunk_start", 0.0),
                    audio_path        = result.get("audio_path"),
                    transcript        = result.get("text", ""),
                    language_mix      = result.get("language_mix", ""),
                    final_score       = score.get("final_score", 0.0),
                    zone              = score.get("zone", "GREEN"),
                    severity          = score.get("severity", "LOW"),
                    auto_alert        = score.get("auto_alert", False),
                    dominant_emotion  = score.get("dominant_emotion"),
                    emergency_category= score.get("emergency_category"),
                    is_emergency      = score.get("is_emergency", False),
                    incident_id       = score.get("incident_id"),
                    emotion_component = comp.get("emotion_component", 0.0),
                    emergency_component=comp.get("emergency_component", 0.0),
                    keyword_component = comp.get("keyword_component", 0.0),
                    sarcasm_deduction = comp.get("sarcasm_deduction", 0.0),
                    keywords_found    = json.dumps(kw.get("keywords_list", [])),
                    keyword_boost     = kw.get("total_boost", 0.0),
                    bart_used         = result.get("emergency_analysis", {}).get("bart_used", False),
                    processing_ms     = result.get("processing_ms", 0),
                    trend_upgraded    = score.get("trend_upgraded", False),
                    rising_trend      = score.get("rising_trend", False),
                )
                db.add(chunk)
                db.commit()
        except Exception as e:
            logger.error(f"DB chunk write failed for {result.get('chunk_id','?')}: {e}")

    def _create_session_row(self, session_id: str) -> None:
        """Create session record in DB."""
        try:
            with self.db_factory() as db:
                from db.models.session_model import Session
                existing = db.query(Session).filter(Session.id == session_id).first()
                if existing is None:
                    db.add(Session(id=session_id, source_type="mic"))
                    db.commit()
        except Exception as e:
            logger.error(f"DB session create failed: {e}")

    def _update_session_stats(self, result: Dict) -> None:
        """Update session aggregate counts after each chunk."""
        session_id = result.get("session_id", "default")
        zone       = result.get("score", {}).get("zone", "GREEN")
        score_val  = result.get("score", {}).get("final_score", 0.0)

        try:
            with self.db_factory() as db:
                from db.models.session_model import Session
                from sqlalchemy import func

                sess = db.query(Session).filter(Session.id == session_id).first()
                if sess is None:
                    sess = Session(id=session_id, source_type="mic")
                    db.add(sess)

                sess.total_chunks = (sess.total_chunks or 0) + 1
                if zone == "RED":
                    sess.red_count = (sess.red_count or 0) + 1
                elif zone == "YELLOW":
                    sess.yellow_count = (sess.yellow_count or 0) + 1
                else:
                    sess.green_count = (sess.green_count or 0) + 1

                # Running average score
                old_avg = sess.avg_score or 0.0
                n       = sess.total_chunks
                sess.avg_score = round(((old_avg * (n - 1)) + score_val) / n, 4)
                db.commit()
        except Exception as e:
            logger.error(f"DB session update failed: {e}")