"""
pipeline/core/worker.py  —  Stage 4 + 5 + 6  (translate mode, timeout-safe)

Whisper fixes matching whisper_transcriber.py v6:
  - task="translate"  →  English output for all languages
  - NO initial_prompt →  was causing keyword hallucination on Hindi chunks
  - _transcribe_with_timeout() → prevents 80-second hangs
  - beam_size=3, temperature=[0,0.2,0.4,0.6]
  - compression_ratio_threshold=2.0 (stricter)
"""

import json, queue, threading, logging, traceback, wave
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ResultsStore:
    def __init__(self, maxlen: int = 500):
        self._results: List[Dict] = []
        self._lock = threading.Lock()
        self._maxlen = maxlen

    def append(self, r: Dict):
        with self._lock:
            self._results.append(r)
            if len(self._results) > self._maxlen:
                self._results = self._results[-self._maxlen:]

    def get_all(self) -> List[Dict]:
        with self._lock: return list(self._results)

    def get_by_zone(self, zone: str) -> List[Dict]:
        with self._lock:
            return [r for r in self._results if r.get("score", {}).get("zone") == zone]

    def get_recent(self, n: int) -> List[Dict]:
        with self._lock: return list(self._results[-n:])

    def clear(self):
        with self._lock: self._results = []

    def count(self) -> int:
        with self._lock: return len(self._results)


class PipelineWorker(threading.Thread):
    """
    Daemon thread: queue → preprocess → translate → keywords →
    emotion → emergency → score → [Stage 6 wav2vec2] → DB → alert
    """

    def __init__(self, audio_queue, results_store: ResultsStore,
                 dispatcher=None, db_factory=None,
                 model_size: str = "small", enable_heavy: bool = True):
        super().__init__(daemon=True, name="PipelineWorker")
        self.queue          = audio_queue
        self.results_store  = results_store
        self.dispatcher     = dispatcher
        self.db_factory     = db_factory
        self.model_size     = model_size
        self.enable_heavy   = enable_heavy
        self._stop_event    = threading.Event()
        self.chunks_processed = 0
        self.last_error: Optional[str]      = None
        self.last_beat:  datetime           = datetime.utcnow()
        self._whisper   = None
        self._models_loaded = False
        self._session_id: Optional[str]     = None
        self._session_start: Optional[datetime] = None

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def run(self):
        logger.info(f"Worker started | model={self.model_size} | heavy={'on' if self.enable_heavy else 'off'} | task=translate")
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except Exception:
                continue
            try:
                result = self._process(item)
                if result:
                    self.results_store.append(result)
                    self.chunks_processed += 1
                    self.last_beat = datetime.utcnow()
                    self._log(result)
                    if self.db_factory:
                        self._write_db(result)
                        self._update_session(result)
                    if result.get("score", {}).get("zone") == "RED" and self.dispatcher:
                        try:
                            self.dispatcher.handle_red(result)
                        except Exception as e:
                            logger.error(f"Dispatcher error: {e}")
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Worker error on {item.get('chunk_id','?')}: {e}")
                logger.debug(traceback.format_exc())
            finally:
                try: self.queue.task_done()
                except Exception: pass
        logger.info(f"Worker stopped | processed={self.chunks_processed}")

    def stop(self): self._stop_event.set()
    def is_running(self): return self.is_alive() and not self._stop_event.is_set()

    def set_session(self, session_id: str):
        self._session_id = session_id
        self._session_start = datetime.utcnow()
        if self.db_factory:
            self._create_session_row(session_id)

    def health(self) -> Dict:
        stale = (datetime.utcnow() - self.last_beat).total_seconds()
        try:
            from pipeline.core.heavy_pipeline import is_heavy_available
            heavy_ok = is_heavy_available()
        except Exception:
            heavy_ok = False
        return {
            "running":           self.is_running(),
            "chunks_processed":  self.chunks_processed,
            "last_beat_sec_ago": round(stale, 1),
            "heartbeat_stale":   stale > 30 and self.chunks_processed > 0,
            "last_error":        self.last_error,
            "queue_size":        self.queue.qsize(),
            "models_loaded":     self._models_loaded,
            "heavy_available":   heavy_ok,
        }

    # ── pipeline ───────────────────────────────────────────────────────────────
    def _process(self, item: Dict) -> Optional[Dict]:
        import os
        chunk_id    = item.get("chunk_id",    "unknown")
        audio_path  = item.get("audio_path",  "")
        chunk_start = item.get("chunk_start",  0.0)
        chunk_index = item.get("chunk_index",  0)
        session_id  = item.get("session_id",  self._session_id or "default")

        if not os.path.exists(audio_path):
            logger.warning(f"File not found: {audio_path}")
            return None
        if not self._models_loaded:
            self._load_models()

        t_start = datetime.utcnow()

        # 1. Preprocess (normalize live mic audio)
        proc_path = self._preprocess(audio_path, chunk_id)
        if proc_path is None:
            return None   # silent chunk

        # 2. Translate to English
        tr   = self._transcribe(proc_path, chunk_id, chunk_start)
        text = tr.get("text", "")

        # 3. Keywords
        from pipeline.phase2_stt.keyword_normalizer import apply_keyword_normalization
        tr = apply_keyword_normalization(tr)
        kw = tr.get("keyword_analysis", {})

        # 4. Emotion + sarcasm
        from pipeline.phase3_analysis.emotion_detector import detect_emotion, resolve_sarcasm_conflict
        from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
        emo    = detect_emotion(text)
        sarc   = detect_sarcasm(text)
        sarc_r = resolve_sarcasm_conflict(emo, sarc)

        # 5. Emergency
        from pipeline.phase3_analysis.emergency_detector import detect_emergency
        from pipeline.core.config import MIN_WORDS_FOR_EMERGENCY, EMERGENCY_THRESHOLD
        wc = len(text.split()) if text else 0
        if wc >= MIN_WORDS_FOR_EMERGENCY:
            emrg = detect_emergency(text, wait_for_model=False)
            kw_b = kw.get("total_boost", 0.0)
            kw_c = kw.get("top_category", None)
            if kw_b > 0 and kw_c:
                kw_cat = kw_c.lower()
                if kw_cat in emrg.get("all_scores", {}):
                    boosted = min(emrg["all_scores"][kw_cat] + kw_b, 1.0)
                    emrg["all_scores"][kw_cat] = round(boosted, 4)
                    nt = max(emrg["all_scores"], key=emrg["all_scores"].get)
                    emrg.update({"top_category": nt, "top_score": emrg["all_scores"][nt],
                                  "is_emergency": nt != "normal" and emrg["all_scores"][nt] >= EMERGENCY_THRESHOLD})
        else:
            emrg = {"top_category":"normal","top_score":0.0,"is_emergency":False,
                    "all_scores":{},"risk_level":0.0,"bart_used":False,"skipped":f"short ({wc}w)"}

        # 6. Score + zone
        from pipeline.phase4_decision.scorer import compute_score
        from pipeline.phase4_decision.zone_classifier import classify_zone
        chunk_data = {
            "chunk_id": chunk_id, "chunk_start": chunk_start,
            "chunk_index": chunk_index, "session_id": session_id,
            "audio_path": audio_path, "text": text,
            "language_mix": tr.get("language_mix", ""),
            "emotion_analysis": {"emotion": emo, "sarcasm": sarc, "sarcasm_resolution": sarc_r},
            "emergency_analysis": emrg, "keyword_analysis": kw,
        }
        chunk_data["score"] = compute_score(chunk_data)
        classify_zone(chunk_data["score"])

        # 7. Stage 6 — wav2vec2 on YELLOW/RED
        if self.enable_heavy:
            try:
                from pipeline.core.heavy_pipeline import run_heavy_pipeline_if_needed
                chunk_data = run_heavy_pipeline_if_needed(chunk_data, proc_path)
            except Exception as e:
                logger.warning(f"Stage 6 error (non-fatal): {e}")

        ms = int((datetime.utcnow() - t_start).total_seconds() * 1000)
        chunk_data["processing_ms"] = ms
        chunk_data["processed_at"]  = datetime.utcnow().isoformat()
        return chunk_data

    # ── preprocessing ──────────────────────────────────────────────────────────
    def _preprocess(self, audio_path: str, chunk_id: str) -> Optional[str]:
        import os
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                nc = wf.getnchannels()
                sw = wf.getsampwidth()
            if not frames: return None
            if sw == 2:
                s = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                s = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                return audio_path
            if nc == 2:
                s = s.reshape(-1, 2).mean(axis=1)
            rms = float(np.sqrt(np.mean(s ** 2)))
            if rms < 0.001:
                logger.debug(f"{chunk_id} | Silent (RMS={rms:.5f}) — skip")
                return None
            bn = os.path.basename(audio_path)
            parts = bn.replace(".wav","").split("_")
            is_live = (len(parts) >= 4 and parts[1].isdigit() and len(parts[1]) == 8)
            if not is_live:
                return audio_path
            target = 0.1
            gain = min(target / max(rms, 1e-6), 10.0)
            s = np.clip(s * gain, -1.0, 1.0)
            norm = audio_path.replace(".wav", "_norm.wav")
            with wave.open(norm, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes((s * 32767).astype(np.int16).tobytes())
            return norm
        except Exception as e:
            logger.warning(f"Preprocess failed {chunk_id}: {e}")
            return audio_path

    # ── transcription (translate → English) ───────────────────────────────────
    def _transcribe(self, audio_path: str, chunk_id: str, chunk_start: float) -> Dict:
        from pipeline.phase2_stt.whisper_transcriber import (
            is_hallucination, classify_language_mix, _transcribe_with_timeout
        )
        try:
            result = _transcribe_with_timeout(
                self._whisper, audio_path,
                timeout_sec = 45,
                # ── TRANSLATE mode — English output for all languages ────────
                language     = None,
                task         = "translate",
                beam_size    = 3,
                best_of      = 3,
                # NO initial_prompt — was causing keyword hallucination
                word_timestamps = True,
                vad_filter   = True,
                vad_parameters = dict(min_silence_duration_ms=200, min_speech_duration_ms=100),
                temperature  = [0.0, 0.2, 0.4, 0.6],
                condition_on_previous_text  = False,
                compression_ratio_threshold = 2.0,
                log_prob_threshold          = -0.8,
                no_speech_threshold         = 0.6,
            )
            if result is None:
                return {"chunk_id": chunk_id, "chunk_start": chunk_start, "text": "", "language_mix": ""}

            segs, info = result
            text = " ".join(s.text.strip() for s in segs).strip()
            if is_hallucination(text):
                logger.debug(f"{chunk_id} | Hallucination — clearing")
                text = ""
            dl = getattr(info, "language", "?")
            dp = getattr(info, "language_probability", 0.0)
            lm = classify_language_mix(dl, dp)
            logger.info(f"{chunk_id} | src={lm} ({dl}:{dp:.2f}) → EN | '{text[:70]}'")
            return {"chunk_id": chunk_id, "chunk_start": chunk_start,
                    "text": text, "language_mix": lm,
                    "detection_confidence": round(dp, 3)}
        except Exception as e:
            logger.error(f"Transcription failed {chunk_id}: {e}")
            return {"chunk_id": chunk_id, "chunk_start": chunk_start,
                    "text": "", "language_mix": ""}

    def _load_models(self):
        try:
            import transformers; transformers.logging.set_verbosity_error()
        except Exception: pass
        logger.info(f"Worker: loading Whisper '{self.model_size}' (translate mode)...")
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        logger.info(f"Whisper '{self.model_size}' ✅")
        from pipeline.phase3_analysis.emotion_detector import get_emotion_model
        get_emotion_model()
        logger.info("Emotion model ✅")
        from pipeline.phase3_analysis.emergency_detector import start_background_loading
        start_background_loading()
        logger.info("BART background loading started ✅")
        self._models_loaded = True
        logger.info("Worker: light models loaded — pipeline operational (translate→EN) ✅")

    def _log(self, r: Dict):
        s = r.get("score", {})
        h = "⚡" if r.get("heavy_pipeline") else ""
        logger.info(
            f"[Worker]{h}{s.get('zone_emoji','')}{s.get('zone','?')} | "
            f"score={s.get('final_score',0):.3f} | emo={s.get('dominant_emotion','?')} | "
            f"cat={s.get('emergency_category','?')} | {r.get('processing_ms',0)}ms | "
            f"#{self.chunks_processed} | \"{(r.get('text') or '')[:55]}\""
        )

    # ── DB writes ──────────────────────────────────────────────────────────────
    def _write_db(self, r: Dict):
        try:
            with self.db_factory() as db:
                from db.models.chunk_model import Chunk
                s = r.get("score", {})
                k = r.get("keyword_analysis", {})
                c = s.get("components", {})
                db.add(Chunk(
                    chunk_id=r.get("chunk_id"), session_id=r.get("session_id","default"),
                    chunk_index=r.get("chunk_index",0), chunk_start_sec=r.get("chunk_start",0.0),
                    audio_path=r.get("audio_path"), transcript=r.get("text",""),
                    language_mix=r.get("language_mix",""),
                    final_score=s.get("final_score",0.0), zone=s.get("zone","GREEN"),
                    severity=s.get("severity","LOW"), auto_alert=s.get("auto_alert",False),
                    dominant_emotion=s.get("dominant_emotion"),
                    emergency_category=s.get("emergency_category"),
                    is_emergency=s.get("is_emergency",False), incident_id=s.get("incident_id"),
                    emotion_component=c.get("emotion_component",0.0),
                    emergency_component=c.get("emergency_component",0.0),
                    keyword_component=c.get("keyword_component",0.0),
                    sarcasm_deduction=c.get("sarcasm_deduction",0.0),
                    keywords_found=json.dumps(k.get("keywords_list",[])),
                    keyword_boost=k.get("total_boost",0.0),
                    bart_used=r.get("emergency_analysis",{}).get("bart_used",False),
                    processing_ms=r.get("processing_ms",0),
                    trend_upgraded=s.get("trend_upgraded",False),
                    rising_trend=s.get("rising_trend",False),
                ))
                db.commit()
        except Exception as e:
            logger.error(f"DB write failed {r.get('chunk_id','?')}: {e}")

    def _create_session_row(self, sid: str):
        try:
            with self.db_factory() as db:
                from db.models.session_model import Session
                if not db.query(Session).filter(Session.id == sid).first():
                    db.add(Session(id=sid, source_type="mic")); db.commit()
        except Exception as e:
            logger.error(f"DB session create failed: {e}")

    def _update_session(self, r: Dict):
        sid   = r.get("session_id", "default")
        zone  = r.get("score", {}).get("zone", "GREEN")
        score = r.get("score", {}).get("final_score", 0.0)
        try:
            with self.db_factory() as db:
                from db.models.session_model import Session
                sess = db.query(Session).filter(Session.id == sid).first()
                if sess is None:
                    sess = Session(id=sid, source_type="mic"); db.add(sess)
                sess.total_chunks = (sess.total_chunks or 0) + 1
                if zone == "RED":    sess.red_count    = (sess.red_count    or 0) + 1
                elif zone == "YELLOW": sess.yellow_count = (sess.yellow_count or 0) + 1
                else:               sess.green_count  = (sess.green_count  or 0) + 1
                n = sess.total_chunks
                sess.avg_score = round(((sess.avg_score or 0.0) * (n-1) + score) / n, 4)
                db.commit()
        except Exception as e:
            logger.error(f"DB session update failed: {e}")