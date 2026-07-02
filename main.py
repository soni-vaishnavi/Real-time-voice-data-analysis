"""
main.py
========
VoiceGuard — Unified Entry Point (Stage 0 → 7)

Two modes:
  --mode file  (default) Phases 1-6 on a pre-recorded audio file
  --mode live  Live mic → FastAPI → queue → worker → DB → alerts → dashboard

Accuracy fixes in this version:
  - File mode: uses Whisper "small" model by default (--model tiny to override)
  - File mode: transcripts_path fixed to all_transcripts.json (not _final.json)
  - Live mode: worker normalizes mic audio before Whisper
  - Both modes: wav2vec2 LOAD REPORT warnings suppressed

Stage 7 in this version:
  - Live mode: sets DASHBOARD_DATA_SOURCE=api so dashboard reads from FastAPI
  - File mode: sets DASHBOARD_DATA_SOURCE=file (original JSON behavior)

Usage:
    python main.py                             # file mode, default audio
    python main.py input/audio/my.wav          # file mode, custom audio
    python main.py --model tiny                # file mode, fast tiny model
    python main.py --mode live                 # live mic + API + dashboard
    python main.py --mode live --no-dashboard  # live mic + API only
    python main.py --mode live --mic-device 1  # specific mic device
    python main.py --skip-phase1 --skip-phase2 # reuse existing output
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
import subprocess

os.makedirs("output", exist_ok=True)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Remove any pre-existing handlers and set new console + debug file handlers
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

debug_handler = logging.FileHandler("output/debug.log", mode="a", encoding="utf-8")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
root_logger.addHandler(debug_handler)

logger = logging.getLogger("main")

DEFAULT_AUDIO  = "input/audio/emergency_scenario.wav"
API_URL        = "http://localhost:8000"
API_PORT       = 8000
DASHBOARD_PORT = 8501


# ── Suppress noisy HuggingFace model weight warnings ─────────────────────────
def _suppress_hf_warnings():
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  FILE MODE — Phases 1-6
# ══════════════════════════════════════════════════════════════════════════════

def run_file_mode(audio_path: str, model_size: str, skip: dict, dashboard: bool = False) -> None:

    def _step(msg):  print(f"\n  ▶  {msg}")
    def _ok(msg, t=None): print(f"  ✅ {msg}" + (f" ({t:.1f}s)" if t else ""))

    print("\n" + "═" * 60)
    print("  VOICEGUARD — FILE MODE")
    print(f"  Audio: {audio_path}  |  Model: whisper-{model_size}")
    print("═" * 60)

    _suppress_hf_warnings()

    # Stage 7: file mode uses JSON file data source
    os.environ["DASHBOARD_DATA_SOURCE"] = "file"

    from pipeline.core.config import (
        validate_config, CHUNKS_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR, DECISIONS_DIR,
    )
    validate_config()
    _ok("Config validated")

    from pipeline.phase3_analysis.emergency_detector import start_background_loading
    start_background_loading()
    _ok("BART loading started in background")

    t_total = time.time()

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    meta_path = str(CHUNKS_DIR / "metadata.json")
    if not skip.get("phase1"):
        _step("Phase 1 — Audio preprocessing + VAD chunking")
        t0 = time.time()
        from pipeline.core.config import OUTPUT_DIR
        from pipeline.phase1_audio.preprocessor  import preprocess
        from pipeline.phase1_audio.noise_reducer  import process_noise_reduction
        from pipeline.phase1_audio.vad_chunker    import process_vad_chunking

        pre_path   = str(OUTPUT_DIR / "chunks" / "preprocessed.wav")
        noise_path = str(OUTPUT_DIR / "chunks" / "noise_reduced.wav")
        preprocess(audio_path, pre_path)
        process_noise_reduction(pre_path, noise_path)
        chunks = process_vad_chunking(
            input_path=noise_path, output_dir=str(CHUNKS_DIR),
            aggressiveness=2, window_sec=5.0, overlap_sec=2.0,
        )
        with open(meta_path, "w") as f:
            json.dump(chunks, f, indent=2)
        _ok(f"Phase 1 — {len(chunks)} chunks", time.time() - t0)
    else:
        print("  ⚠️  Skipping Phase 1")

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    # FIX: use all_transcripts.json (whisper_transcriber saves here)
    transcripts_path = str(TRANSCRIPTS_DIR / "all_transcripts.json")

    if not skip.get("phase2"):
        _step(f"Phase 2 — Whisper STT ({model_size})")
        t0 = time.time()
        from pipeline.phase2_stt.whisper_transcriber import transcribe_all_chunks
        transcripts = transcribe_all_chunks(
            metadata_path = meta_path,
            output_dir    = str(TRANSCRIPTS_DIR),
            model_size    = model_size,   # "small" by default now
        )
        _ok(f"Phase 2 — {len(transcripts)} transcripts", time.time() - t0)
    else:
        print("  ⚠️  Skipping Phase 2")

    # ── Phase 3 ────────────────────────────────────────────────────────────────
    analysis_path = str(ANALYSIS_DIR / "all_analysis.json")
    if not skip.get("phase3"):
        _step("Phase 3 — Emotion + Emergency analysis")
        t0 = time.time()
        from pipeline.phase3_analysis.analyzer import run_phase3
        run_phase3(transcripts_path=transcripts_path, output_dir=str(ANALYSIS_DIR))
        _ok("Phase 3 complete", time.time() - t0)
    else:
        print("  ⚠️  Skipping Phase 3")

    # ── Phase 4 ────────────────────────────────────────────────────────────────
    decisions_path = str(DECISIONS_DIR / "all_decisions.json")
    _step("Phase 4 — Decision engine")
    t0 = time.time()
    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)
    from pipeline.phase4_decision.scorer          import score_all_chunks
    from pipeline.phase4_decision.zone_classifier import classify_all_zones
    from pipeline.phase4_decision.trend_analyzer  import apply_trend_analysis
    final = apply_trend_analysis(classify_all_zones(score_all_chunks(analysis)))
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    zones = [c.get("score", {}).get("zone", "GREEN") for c in final]
    red   = zones.count("RED")
    _ok(f"Phase 4 — 🟢{zones.count('GREEN')} 🟡{zones.count('YELLOW')} 🔴{red}",
        time.time() - t0)

    # ── Phase 5 ────────────────────────────────────────────────────────────────
    if red > 0:
        _step(f"Phase 5 — Auto-alerts ({red} RED incidents)")
        from pipeline.phase5_alerts.sms_alert  import send_emergency_sms
        from pipeline.phase5_alerts.email_alert import send_emergency_email
        for chunk in final:
            s = chunk.get("score", {})
            if s.get("zone") == "RED" and s.get("auto_alert"):
                idx    = next((i for i, c in enumerate(final)
                               if c.get("chunk_id") == chunk.get("chunk_id")), 0)
                recent = final[max(0, idx - 4):idx + 1]
                send_emergency_sms(chunk)
                send_emergency_email(chunk, recent)
                print(f"    🚨 Alert: {s.get('incident_id','?')}")
    else:
        _ok("Phase 5 — No RED incidents")

    # ── Phase 6 ────────────────────────────────────────────────────────────────
    _step("Phase 6 — Session report")
    t0 = time.time()
    from pipeline.phase6_reports.session_report import generate_session_report
    pdf = generate_session_report(final, alert_log=[])
    _ok(f"Report: {pdf}", time.time() - t0)

    print("\n" + "═" * 60)
    print("  FILE MODE COMPLETE ✅")
    print(f"  Total: {time.time() - t_total:.1f}s")
    print(f"\n  To view dashboard (file mode, reads all_decisions.json):")
    print(f"  streamlit run pipeline/phase5_alerts/dashboard.py")
    if dashboard:
        print("\n  Launching dashboard...")
        dash = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             "pipeline/phase5_alerts/dashboard.py",
             "--server.port", str(DASHBOARD_PORT),
             "--server.headless", "true"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=os.environ,
        )
        time.sleep(2)
        if dash.poll() is None:
            print(f"  ✅ Dashboard started — http://localhost:{DASHBOARD_PORT}")
        else:
            print("  ❌ Failed to start dashboard. Run the Streamlit command manually.")
        print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE MODE — FastAPI + Mic + Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def run_live_mode(no_dashboard: bool = False, mic_device: int = None) -> None:

    _suppress_hf_warnings()

    # Stage 7: live mode uses API data source
    os.environ["DASHBOARD_DATA_SOURCE"] = "api"
    os.environ["DASHBOARD_API_URL"]     = API_URL

    print("\n" + "═" * 60)
    print("  VOICEGUARD — LIVE MODE (Stage 6+7)")
    print(f"  API:       http://localhost:{API_PORT}/docs")
    print(f"  Chunks:    http://localhost:{API_PORT}/chunks")
    print(f"  Alerts:    http://localhost:{API_PORT}/alerts")
    print(f"  Health:    http://localhost:{API_PORT}/health")
    print(f"  DB:        database/voiceguard.db")
    if not no_dashboard:
        print(f"  Dashboard: http://localhost:{DASHBOARD_PORT}  [reads from API]")
    print("  Press Ctrl+C to stop")
    print("═" * 60 + "\n")

    processes  = []
    mic        = None
    stop_event = threading.Event()

    def shutdown(sig=None, frame=None):
        print("\n\n  Shutting down VoiceGuard...")
        stop_event.set()
        if mic and mic.is_alive():
            mic.stop()
            mic.join(timeout=5)
        for p in processes:
            try: p.terminate()
            except Exception: pass
        print("  Stopped ✅")

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Start FastAPI ──────────────────────────────────────────────────────────
    print("  Starting FastAPI...")
    env_live = {**os.environ, "DASHBOARD_DATA_SOURCE": "api"}
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(API_PORT), "--log-level", "warning"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env_live,
    )
    processes.append(api_proc)
    time.sleep(3)

    if api_proc.poll() is not None:
        print("  ❌ FastAPI failed to start. Check logs.")
        return
    print(f"  ✅ FastAPI ready — http://localhost:{API_PORT}/docs")

    # ── Start dashboard ────────────────────────────────────────────────────────
    if not no_dashboard:
        dash = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             "pipeline/phase5_alerts/dashboard.py",
             "--server.port", str(DASHBOARD_PORT),
             "--server.headless", "true"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env={**env_live},
        )
        processes.append(dash)
        time.sleep(2)
        if dash.poll() is None:
            print(f"  ✅ Dashboard — http://localhost:{DASHBOARD_PORT}  (reads live from API)")

    # ── Start mic capture ──────────────────────────────────────────────────────
    print("\n  Starting mic capture...\n")

    import requests as req_lib
    from pipeline.core.mic_capture import MicCapture
    from pipeline.core.config import LIVE_CHUNK_WINDOW_SEC

    session_id = [time.strftime("session_%Y%m%d_%H%M%S")]

    def send_to_api(item: dict):
        try:
            with open(item["audio_path"], "rb") as f:
                resp = req_lib.post(
                    f"{API_URL}/audio/submit",
                    files = {"file": (os.path.basename(item["audio_path"]), f, "audio/wav")},
                    data  = {
                        "session_id":  session_id[0],
                        "chunk_index": str(item["chunk_index"]),
                        "chunk_start": str(item["chunk_start"]),
                    },
                    timeout = 5,
                )
            data   = resp.json()
            status = data.get("status", "?")
            q_size = data.get("queue_size", "?")
            logger.info(f"Submitted {item['chunk_id']} | status={status} | queue={q_size}")
        except Exception as e:
            logger.warning(f"Submit failed {item.get('chunk_id','?')}: {e}")

    mic = MicCapture(on_chunk=send_to_api, device=mic_device, chunk_sec=LIVE_CHUNK_WINDOW_SEC)
    mic.start()

    print("  System running.")
    print(f"  Live results: http://localhost:{API_PORT}/chunks")
    print(f"  Active alerts: http://localhost:{API_PORT}/alerts")
    print(f"  Health:        http://localhost:{API_PORT}/health\n")

    while not stop_event.is_set():
        time.sleep(1)

    shutdown()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs("output", exist_ok=True)

    parser = argparse.ArgumentParser(
        description = "VoiceGuard — Real-Time Voice Emergency Detection",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python main.py                               File mode, default audio (small model)
  python main.py --model tiny                  File mode, fast tiny model
  python main.py input/audio/test.wav          File mode, custom audio
  python main.py --mode live                   Live mic + API + dashboard
  python main.py --mode live --no-dashboard    Live mic + API only
  python main.py --mode live --mic-device 1    Use mic device index 1
  python main.py --skip-phase1 --skip-phase2   Reuse existing output
        """,
    )
    parser.add_argument("audio",          nargs="?",  default=DEFAULT_AUDIO)
    parser.add_argument("--mode",         choices=["file","live"], default="file")
    parser.add_argument("--model",        choices=["tiny","base","small","medium"],
                        default="small")   # CHANGED: was "tiny"
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--mic-device",   type=int,   default=None)
    parser.add_argument("--skip-phase1",  action="store_true")
    parser.add_argument("--skip-phase2",  action="store_true")
    parser.add_argument("--skip-phase3",  action="store_true")
    parser.add_argument("--dashboard",    action="store_true",
                        help="Start the Streamlit dashboard after file mode completes")
    args = parser.parse_args()

    if args.mode == "live":
        run_live_mode(
            no_dashboard = args.no_dashboard,
            mic_device   = args.mic_device,
        )
    else:
        if not os.path.exists(args.audio) and not args.skip_phase1:
            print(f"  ❌ Audio file not found: {args.audio}")
            sys.exit(1)
        run_file_mode(
            audio_path = args.audio,
            model_size = args.model,
            skip = {
                "phase1": args.skip_phase1,
                "phase2": args.skip_phase2,
                "phase3": args.skip_phase3,
            },
            dashboard = args.dashboard,
        )


if __name__ == "__main__":
    main()