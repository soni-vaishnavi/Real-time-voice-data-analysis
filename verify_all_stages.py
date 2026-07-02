"""
verify_all_stages.py
=====================
Stage 8 — Full System Verification (Windows-safe, UTF-8 file reads)

Runs all stage checks in order. Use before pushing to Git.
No ML models loaded — all checks are import/logic/config only.

Usage:
    python verify_all_stages.py
    python verify_all_stages.py --models    # also verify model load (~2 min)
"""

import sys
import os
import ast
import json
import pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []


def read_src(path: str) -> str:
    """Read file with UTF-8 encoding — required on Windows for files with emojis."""
    return pathlib.Path(path).read_text(encoding="utf-8")


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  OK  {name}")
    else:
        msg = f"{name}" + (f" — {detail}" if detail else "")
        print(f"  !!  {msg}")
        errors.append(msg)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── STAGE 0: Config ────────────────────────────────────────────────────────────
section("STAGE 0 — Config Unification")

try:
    from pipeline.core.config import (
        WEIGHT_EMOTION, WEIGHT_EMERGENCY, WEIGHT_KEYWORD,
        ZONE_YELLOW, ZONE_RED, EMERGENCY_THRESHOLD,
        WHISPER_MODEL_SIZE, DASHBOARD_DATA_SOURCE,
        validate_config,
    )
    check("Weights sum to 1.0",
          abs(WEIGHT_EMOTION + WEIGHT_EMERGENCY + WEIGHT_KEYWORD - 1.0) < 0.001)
    check("EMERGENCY_THRESHOLD = 0.55",
          EMERGENCY_THRESHOLD == 0.55, f"got {EMERGENCY_THRESHOLD}")
    check("WHISPER_MODEL_SIZE = 'small'",
          WHISPER_MODEL_SIZE == "small", f"got {WHISPER_MODEL_SIZE}")
    check("ZONE_YELLOW < ZONE_RED < 1.0",
          0 < ZONE_YELLOW < ZONE_RED < 1.0)
    check("DASHBOARD_DATA_SOURCE defined",
          DASHBOARD_DATA_SOURCE in ("file", "api"))
    validate_config()
    check("validate_config() passes", True)
except Exception as e:
    check("Config import", False, str(e))

try:
    from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
    r  = detect_sarcasm("haan haan bilkul bore ho gaya")
    r2 = detect_sarcasm("bachao ambulance bulao")
    check("Sarcasm: detects sarcasm",              r["is_sarcastic"] == True)
    check("Sarcasm: genuine distress not sarcasm", r2["is_sarcastic"] == False)
except Exception as e:
    check("Sarcasm rules", False, str(e))

try:
    import pipeline.phase3_analysis.emergency_detector as ed
    from pipeline.core.config import EMERGENCY_THRESHOLD
    check("EMERGENCY_THRESHOLD runtime match",
          ed.EMERGENCY_THRESHOLD == EMERGENCY_THRESHOLD, f"got {ed.EMERGENCY_THRESHOLD}")
    check("No sarcasm model in emergency_detector",
          not hasattr(ed, '_sarcasm_pipeline'))
except Exception as e:
    check("Emergency detector", False, str(e))

try:
    import pipeline.phase4_decision.scorer as sc
    import pipeline.phase4_decision.zone_classifier as zc
    from pipeline.core.config import WEIGHT_EMOTION, ZONE_YELLOW, ZONE_RED
    check("Scorer imports from config",           sc.WEIGHT_EMOTION == WEIGHT_EMOTION)
    check("Zone classifier imports from config",  zc.ZONE_YELLOW == ZONE_YELLOW and zc.ZONE_RED == ZONE_RED)
except Exception as e:
    check("Scorer/Classifier imports", False, str(e))


# ── STAGE 1: Mic Capture ──────────────────────────────────────────────────────
section("STAGE 1 — Mic Capture")

try:
    from pipeline.core.mic_capture import MicCapture
    check("MicCapture importable", True)
    check("MicCapture is Thread subclass",
          issubclass(MicCapture, __import__('threading').Thread))
except Exception as e:
    check("MicCapture import", False, str(e))


# ── STAGE 2: Queue ────────────────────────────────────────────────────────────
section("STAGE 2 — Queue Manager")

try:
    from pipeline.core.queue_manager import AudioQueue
    q = AudioQueue(maxsize=3)
    q.put({"chunk_id":"t1","audio_path":"/x.wav","chunk_start":0,"chunk_index":0,"session_id":"s"})
    q.put({"chunk_id":"t2","audio_path":"/y.wav","chunk_start":5,"chunk_index":1,"session_id":"s"})
    check("Queue accepts items", q.qsize() == 2)
    check("Queue stats ok",      q.stats()["total_in"] == 2)
except Exception as e:
    check("AudioQueue", False, str(e))


# ── STAGE 3: Worker ───────────────────────────────────────────────────────────
section("STAGE 3 — Pipeline Worker")

try:
    worker_src = read_src("pipeline/core/worker.py")
    ast.parse(worker_src)
    check("worker.py parses",                    True)
    check("worker: task=translate",              '"translate"' in worker_src)
    check("worker: no initial_prompt kwarg",     "initial_prompt" not in worker_src or "NO initial_prompt" in worker_src)
    check("worker: _transcribe_with_timeout",    "_transcribe_with_timeout" in worker_src)
    check("worker: Stage 6 heavy_pipeline hook", "heavy_pipeline" in worker_src)
    check("worker: DB writes",                   "_write_db" in worker_src or "_write_chunk_to_db" in worker_src)
except Exception as e:
    check("Worker source", False, str(e))

try:
    from pipeline.core.worker import PipelineWorker, ResultsStore
    check("PipelineWorker importable", True)
    check("ResultsStore importable",   True)
except Exception as e:
    check("Worker import", False, str(e))


# ── STAGE 4: Database ─────────────────────────────────────────────────────────
section("STAGE 4 — SQLite Database")

try:
    ast.parse(read_src("db/database.py"))
    check("db/database.py parses", True)
    for mf in ["chunk_model.py", "session_model.py", "incident.py", "alert_action.py"]:
        p = pathlib.Path(f"db/models/{mf}")
        if p.exists():
            ast.parse(p.read_text(encoding="utf-8"))
            check(f"db/models/{mf} parses", True)
        else:
            check(f"db/models/{mf} exists", False, "not found")
except Exception as e:
    check("Database files", False, str(e))


# ── STAGE 5: Alerts ───────────────────────────────────────────────────────────
section("STAGE 5 — Alert Dispatcher")

try:
    disp_src = read_src("pipeline/phase5_alerts/dispatcher.py")
    ast.parse(disp_src)
    check("dispatcher.py parses",            True)
    check("AlertDispatcher.handle_red()",   "handle_red" in disp_src)
    check("AlertDispatcher.confirm()",      "def confirm" in disp_src)
    check("AlertDispatcher.reject()",       "def reject"  in disp_src)
    check("Deduplication _fired_set",       "_fired_set"  in disp_src)
except Exception as e:
    check("Dispatcher", False, str(e))


# ── STAGE 6: Heavy pipeline ───────────────────────────────────────────────────
section("STAGE 6 — Heavy Pipeline (wav2vec2)")

try:
    hp_src = read_src("pipeline/core/heavy_pipeline.py")
    ast.parse(hp_src)
    check("heavy_pipeline.py parses",                 True)
    check("GREEN chunks skipped",                     "GREEN" in hp_src)
    check("Lazy load _load_wav2vec2()",               "_load_wav2vec2" in hp_src)
    check("run_heavy_pipeline_if_needed()",           "run_heavy_pipeline_if_needed" in hp_src)
    check("Re-scores with fused emotion",             "compute_score" in hp_src)
except Exception as e:
    check("Heavy pipeline", False, str(e))


# ── STAGE 7: Dashboard data layer ─────────────────────────────────────────────
section("STAGE 7 — Dashboard Data Layer")

try:
    dd_src = read_src("pipeline/phase5_alerts/dashboard_data.py")
    ast.parse(dd_src)
    check("dashboard_data.py parses",  True)
    check("load_chunks() defined",     "def load_chunks" in dd_src)
    check("File mode loader",          "_load_from_file" in dd_src)
    check("API mode loader",           "_load_from_api"  in dd_src)
    check("post_action() defined",     "def post_action" in dd_src)
except Exception as e:
    check("Dashboard data layer", False, str(e))


# ── STAGE 8: Transcription ────────────────────────────────────────────────────
section("STAGE 8 — Transcription Accuracy (translate mode)")

try:
    tr_src = read_src("pipeline/phase2_stt/whisper_transcriber.py")
    ast.parse(tr_src)
    check("whisper_transcriber.py parses",        True)
    check("task=translate",                       '"translate"' in tr_src)
    check("HINGLISH_INITIAL_PROMPT = None",       "HINGLISH_INITIAL_PROMPT = None" in tr_src)
    check("_transcribe_with_timeout() defined",   "def _transcribe_with_timeout" in tr_src)
    check("Devanagari in hallucination filter",   "0964" in tr_src or "Devanagari" in tr_src or "danda" in tr_src.lower())
    check("compression_ratio_threshold = 2.0",   "2.0" in tr_src)
except Exception as e:
    check("Whisper transcriber source", False, str(e))

try:
    from pipeline.phase2_stt.whisper_transcriber import is_hallucination
    check("Hallucination: danda repetition",          is_hallucination("| | | |"))
    check("Hallucination: danda Devanagari",          is_hallucination("\u0964 \u0964 \u0964 \u0964"))
    check("Hallucination: Devanagari loop",           is_hallucination("\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c\u092c"))
    check("Hallucination: valid English OK",      not is_hallucination("save me call the ambulance"))
    check("Hallucination: valid Hindi-EN OK",     not is_hallucination("please help doctor chahiye"))
except Exception as e:
    check("Hallucination filter runtime", False, str(e))


# ── STAGE 9: Optimization ─────────────────────────────────────────────────────
section("STAGE 9 — Optimization")

try:
    from pipeline.core.config import QUEUE_MAXSIZE, AUTO_TRIGGER_SEC
    check("Queue bounded (QUEUE_MAXSIZE > 0)", QUEUE_MAXSIZE > 0)
    check("Auto-trigger timer configured",     AUTO_TRIGGER_SEC > 0)
except Exception as e:
    check("Optimization config", False, str(e))


# ── Scoring E2E ───────────────────────────────────────────────────────────────
section("SCORING END-TO-END (no ML models)")

try:
    from pipeline.phase4_decision.scorer import compute_score
    from pipeline.phase4_decision.zone_classifier import classify_zone

    genuine = {
        "emotion_analysis":  {
            "emotion":            {"dominant_emotion":"fear","dominant_score":0.92,
                                   "emergency_weight":1.0,"fear_score":0.92,"anger_score":0.05},
            "sarcasm_resolution": {"score_penalty":0.0},
        },
        "emergency_analysis": {"top_category":"medical","top_score":0.84,"is_emergency":True},
        "keyword_analysis":   {"total_boost":0.62},
    }
    genuine["score"] = compute_score(genuine)
    classify_zone(genuine["score"])
    check("Genuine emergency -> RED",  genuine["score"]["zone"] == "RED")
    check("auto_alert = True",         genuine["score"]["auto_alert"] == True)
    check("final_score > 0.72",        genuine["score"]["final_score"] > 0.72)

    sarcastic = {
        "emotion_analysis":  {
            "emotion":            {"dominant_emotion":"neutral","dominant_score":0.65,
                                   "emergency_weight":0.0,"fear_score":0.05,"anger_score":0.10},
            "sarcasm_resolution": {"score_penalty":0.70},
        },
        "emergency_analysis": {"top_category":"normal","top_score":0.25,"is_emergency":False},
        "keyword_analysis":   {"total_boost":0.47},
    }
    sarcastic["score"] = compute_score(sarcastic)
    classify_zone(sarcastic["score"])
    check("Sarcastic -> GREEN",   sarcastic["score"]["zone"] == "GREEN")

    print(f"    genuine={genuine['score']['final_score']:.3f} RED | "
          f"sarcastic={sarcastic['score']['final_score']:.3f} GREEN")
except Exception as e:
    check("Scoring E2E", False, str(e))
    import traceback; traceback.print_exc()


# ── MAIN.PY ───────────────────────────────────────────────────────────────────
section("MAIN.PY")

try:
    main_src = read_src("main.py")
    ast.parse(main_src)
    check("main.py parses",                        True)
    check("all_transcripts.json (not _final)",     "all_transcripts.json" in main_src and
                                                    "all_transcripts_final" not in main_src)
    check("DASHBOARD_DATA_SOURCE in live mode",    "DASHBOARD_DATA_SOURCE" in main_src)
    check("--model small default",                 "default='small'" in main_src or
                                                    'default="small"' in main_src)
except Exception as e:
    check("main.py", False, str(e))


# ── Optional model loading ─────────────────────────────────────────────────────
if "--models" in sys.argv:
    section("MODEL LOADING (slow — may take 2 minutes)")
    try:
        from pipeline.phase2_stt.whisper_transcriber import get_model
        m = get_model("small")
        check("Whisper 'small' loads", m is not None)
    except Exception as e:
        check("Whisper model load", False, str(e))

    try:
        from pipeline.phase3_analysis.emotion_detector import get_emotion_model
        em = get_emotion_model()
        check("Emotion model loads", em is not None)
    except Exception as e:
        check("Emotion model load", False, str(e))


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors:
    print("  ALL CHECKS PASSED - system is clean")
    print("  Ready to push to Git.")
else:
    print(f"  {len(errors)} CHECK(S) FAILED:")
    for e in errors:
        print(f"     - {e}")
    print("\n  Fix these before pushing.")
print("=" * 60)
sys.exit(0 if not errors else 1)