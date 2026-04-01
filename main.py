"""
VoiceGuard — Full Pipeline Orchestrator
========================================
Runs all 6 phases end-to-end on a single audio file.

Usage:
    python main.py                              # uses default test audio
    python main.py audio/my_recording.wav       # custom audio file
    python main.py audio/my_recording.wav --model small   # better accuracy

What it does:
    Phase 1 → Preprocess + VAD chunk audio
    Phase 2 → Transcribe chunks (faster-whisper)
    Phase 3 → Emotion + emergency analysis
    Phase 4 → Decision engine (score + zone + trend)
    Phase 5 → Fire alerts if RED detected
    Phase 6 → Generate session PDF report
    Dashboard → Launch automatically when done

Output files:
    output/chunks/          → WAV chunk files
    output/transcripts/     → all_transcripts_final.json
    output/analysis/        → all_analysis.json
    output/decisions/       → all_decisions.json   ← dashboard reads this
    output/reports/         → session_report_*.pdf

Author: VoiceGuard Team — Poornima University BCA 2024-25
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/pipeline.log", mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger("main")

# ── Output directories ─────────────────────────────────────────────────────────
DIRS = {
    "chunks":      "output/chunks",
    "transcripts": "output/transcripts",
    "analysis":    "output/analysis",
    "decisions":   "output/decisions",
    "reports":     "output/reports",
}

# ── Default test audio ─────────────────────────────────────────────────────────
DEFAULT_AUDIO = "input/audio/Test_Normal.wav"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

def _step(msg: str):
    print(f"\n  ▶  {msg}")

def _ok(msg: str, elapsed: float = None):
    t = f"  ({elapsed:.1f}s)" if elapsed else ""
    print(f"  ✅ {msg}{t}")

def _warn(msg: str):
    print(f"  ⚠️  {msg}")

def _err(msg: str):
    print(f"  ❌ {msg}")

def _make_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs("output", exist_ok=True)

def _summary_line(chunks):
    total  = len(chunks)
    green  = sum(1 for c in chunks if c.get("score",{}).get("zone")=="GREEN")
    yellow = sum(1 for c in chunks if c.get("score",{}).get("zone")=="YELLOW")
    red    = sum(1 for c in chunks if c.get("score",{}).get("zone")=="RED")
    return f"Total={total}  🟢 GREEN={green}  🟡 YELLOW={yellow}  🔴 RED={red}"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — AUDIO PREPROCESSING + VAD CHUNKING
# ══════════════════════════════════════════════════════════════════════════════
def run_phase1(audio_path: str) -> str:
    """
    Returns path to metadata JSON (list of chunk dicts).
    """
    _banner("PHASE 1 — Audio Preprocessing & VAD Chunking")

    from pipeline.phase1_audio.preprocessor  import preprocess
    from pipeline.phase1_audio.noise_reducer  import process_noise_reduction
    from pipeline.phase1_audio.vad_chunker    import process_vad_chunking

    t0 = time.time()

    # Step 1 — Preprocess (convert to 16kHz mono WAV)
    _step("Preprocessing audio (convert to 16kHz mono WAV)...")
    clean_path = os.path.join(DIRS["chunks"], "preprocessed.wav")
    preprocessed = preprocess(audio_path, clean_path)
    _ok(f"Preprocessed → {preprocessed}")

    # Step 2 — Noise reduction
    _step("Reducing background noise...")
    noise_reduced_path = os.path.join(DIRS["chunks"], "noise_reduced.wav")
    noise_reduced = process_noise_reduction(preprocessed, noise_reduced_path)
    _ok(f"Noise reduced → {noise_reduced}")

    # Step 3 — VAD chunking
    _step("Chunking audio with VAD (5s window, 2s overlap)...")
    chunks = process_vad_chunking(
        input_path  = noise_reduced,
        output_dir  = DIRS["chunks"],
        aggressiveness = 2,
        window_sec  = 5.0,
        overlap_sec = 2.0,
    )

    # Save metadata
    metadata_path = os.path.join(DIRS["chunks"], "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    _ok(f"Phase 1 complete — {len(chunks)} chunks saved to {DIRS['chunks']}/", elapsed)
    return metadata_path


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — SPEECH TO TEXT
# ══════════════════════════════════════════════════════════════════════════════
def run_phase2(metadata_path: str, model_size: str = "tiny") -> str:
    """
    Returns path to all_transcripts_final.json.
    """
    _banner("PHASE 2 — Speech-to-Text (faster-whisper)")

    from pipeline.phase2_stt.whisper_transcriber import transcribe_all_chunks

    t0 = time.time()
    model_sizes = {"tiny": "75MB", "small": "244MB", "medium": "769MB"}
    _step(f"Loading Whisper model: {model_size} (first run downloads ~{model_sizes.get(model_size,'?')})")

    transcripts_path = os.path.join(DIRS["transcripts"], "all_transcripts_final.json")

    transcripts = transcribe_all_chunks(
        metadata_path = metadata_path,
        output_dir    = DIRS["transcripts"],
        model_size    = model_size,
    )

    elapsed = time.time() - t0
    _ok(f"Phase 2 complete — {len(transcripts)} chunks transcribed", elapsed)

    # Show a few samples
    print("\n  Sample transcripts:")
    for c in transcripts[:3]:
        txt = (c.get("text") or "—")[:70]
        print(f"    [{c.get('chunk_id','?')}] {txt}")

    return transcripts_path


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — EMOTION + EMERGENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_phase3(transcripts_path: str) -> str:
    """
    Returns path to all_analysis.json.
    """
    _banner("PHASE 3 — Emotion & Emergency Analysis")
    _warn("This loads 3 HuggingFace models (~3.7GB RAM). May take 2-5 min on first run.")

    from pipeline.phase3_analysis.analyzer import run_phase3 as phase3_run

    t0 = time.time()
    _step("Running dual-channel analysis (emotion + emergency + sarcasm)...")

    results = phase3_run(
        transcripts_path = transcripts_path,
        output_dir       = DIRS["analysis"],
    )

    analysis_path = os.path.join(DIRS["analysis"], "all_analysis.json")
    elapsed = time.time() - t0
    _ok(f"Phase 3 complete — {len(results)} chunks analyzed", elapsed)

    # Emotion summary
    from collections import Counter
    emo_counts = Counter(c.get("dominant_emotion","?") for c in results)
    print(f"\n  Emotion summary: {dict(emo_counts.most_common(4))}")

    return analysis_path


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_phase4(analysis_path: str) -> str:
    """
    Returns path to all_decisions.json.
    """
    _banner("PHASE 4 — Decision Engine (Score + Zone + Trend)")

    from pipeline.phase4_decision.scorer        import score_all_chunks
    from pipeline.phase4_decision.zone_classifier import classify_all_zones
    from pipeline.phase4_decision.trend_analyzer  import apply_trend_analysis

    t0 = time.time()

    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    _step("Computing combined scores...")
    scored = score_all_chunks(analysis)

    _step("Classifying zones (GREEN / YELLOW / RED)...")
    classified = classify_all_zones(scored)

    _step("Applying trend analysis (escalation detection)...")
    final = apply_trend_analysis(classified)

    # Save all_decisions.json
    decisions_path = os.path.join(DIRS["decisions"], "all_decisions.json")
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    _ok(f"Phase 4 complete — {_summary_line(final)}", elapsed)

    # Show RED chunks
    red = [c for c in final if c.get("score",{}).get("zone")=="RED"]
    if red:
        print(f"\n  🔴 RED incidents detected:")
        for c in red:
            s = c.get("score",{})
            print(f"    [{s.get('incident_id','?')}] {c.get('chunk_id','?')} "
                  f"score={round(s.get('final_score',0)*100)}% "
                  f"category={s.get('emergency_category','?')} "
                  f"severity={s.get('severity','?')}")
    else:
        print("  🟢 No RED incidents — all clear")

    return decisions_path


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — AUTO-ALERTS (for RED incidents)
# ══════════════════════════════════════════════════════════════════════════════
def run_phase5_alerts(decisions_path: str):
    """
    Fire SMS + email + sound for any RED auto-alert chunks.
    """
    _banner("PHASE 5 — Auto-Alerts")

    from pipeline.phase5_alerts.sms_alert   import send_emergency_sms
    from pipeline.phase5_alerts.email_alert  import send_emergency_email
    from pipeline.phase5_alerts.sound_alert  import trigger_for_zone

    with open(decisions_path, encoding="utf-8") as f:
        chunks = json.load(f)

    red_auto = [
        c for c in chunks
        if c.get("score",{}).get("zone") == "RED"
        and c.get("score",{}).get("auto_alert", False)
    ]

    if not red_auto:
        _ok("No auto-alert RED incidents — dashboard will handle manual confirmation")
        return []

    fired = []
    for chunk in red_auto:
        s   = chunk.get("score", {})
        iid = s.get("incident_id","?")
        _step(f"Firing alerts for {iid} (score={round(s.get('final_score',0)*100)}%)")

        idx    = next((i for i, c in enumerate(chunks) if c.get("chunk_id")==chunk.get("chunk_id")), 0)
        recent = chunks[max(0, idx-4): idx+1]

        sms_r = send_emergency_sms(chunk)
        eml_r = send_emergency_email(chunk, recent)
        trigger_for_zone("RED")

        _ok(f"SMS: {'sent' if sms_r['sent'] else 'dry-run'}  |  "
            f"Email: {'sent' if eml_r['sent'] else 'dry-run'}")
        fired.append(iid)

    return fired


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — SESSION REPORT
# ══════════════════════════════════════════════════════════════════════════════
def run_phase6_report(decisions_path: str, alert_log: list) -> str:
    """
    Generate session PDF report. Returns path to PDF.
    """
    _banner("PHASE 6 — Session Report Generation")

    from pipeline.phase6_reports.session_report import generate_session_report

    t0 = time.time()
    _step("Generating session PDF report...")

    with open(decisions_path, encoding="utf-8") as f:
        chunks = json.load(f)

    pdf_path = generate_session_report(chunks, alert_log=alert_log)
    elapsed  = time.time() - t0

    size_kb = round(os.path.getsize(pdf_path) / 1024, 1)
    _ok(f"Session report saved: {pdf_path} ({size_kb} KB)", elapsed)
    return pdf_path


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_final_summary(audio_path, decisions_path, pdf_path, total_time, fired_alerts):
    with open(decisions_path, encoding="utf-8") as f:
        chunks = json.load(f)

    total  = len(chunks)
    green  = sum(1 for c in chunks if c.get("score",{}).get("zone")=="GREEN")
    yellow = sum(1 for c in chunks if c.get("score",{}).get("zone")=="YELLOW")
    red    = sum(1 for c in chunks if c.get("score",{}).get("zone")=="RED")
    avg_sc = round(sum(c.get("score",{}).get("final_score",0) for c in chunks)/max(1,total)*100, 1)

    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE ✅")
    print("═" * 60)
    print(f"""
  Audio file    : {audio_path}
  Total time    : {total_time:.1f}s ({round(total_time/60, 1)} min)
  Chunks        : {total}
  
  Zone results  : 🟢 GREEN={green}  🟡 YELLOW={yellow}  🔴 RED={red}
  Average score : {avg_sc}%
  Alerts fired  : {len(fired_alerts)} {fired_alerts if fired_alerts else ''}

  Output files  :
    Decisions   → {decisions_path}
    PDF Report  → {pdf_path}
    Chunks      → {DIRS['chunks']}/
    Log         → output/pipeline.log

  Next step — Launch dashboard:
    streamlit run pipeline/phase5_alerts/dashboard.py
""")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="VoiceGuard — Full Pipeline (Phase 1 → 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # use default test audio
  python main.py audio/recording.wav          # custom audio
  python main.py audio/recording.wav --model small  # better accuracy
  python main.py --skip-phase1               # skip if chunks already exist
        """
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Path to audio file (default: {DEFAULT_AUDIO})"
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Whisper model size (tiny=fast, small=better, medium=best)"
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (use existing chunks/metadata.json)"
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (use existing transcripts)"
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3 (use existing analysis)"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not launch dashboard after pipeline completes"
    )
    args = parser.parse_args()

    # ── Validate audio file ────────────────────────────────────────────────
    if not args.skip_phase1 and not os.path.exists(args.audio):
        _err(f"Audio file not found: {args.audio}")
        _err(f"Usage: python main.py path/to/audio.wav")
        sys.exit(1)

    # ── Create output dirs ─────────────────────────────────────────────────
    _make_dirs()

    print("\n" + "═" * 60)
    print("  VOICEGUARD — REAL-TIME VOICE SURVEILLANCE")
    print("  Full Pipeline: Phase 1 → 2 → 3 → 4 → 5 → 6")
    print("═" * 60)
    print(f"""
  Audio   : {args.audio}
  Model   : whisper-{args.model}
  Started : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}
  Skips   : {'Phase1 ' if args.skip_phase1 else ''}{'Phase2 ' if args.skip_phase2 else ''}{'Phase3 ' if args.skip_phase3 else ''}(none if blank)
""")

    pipeline_start = time.time()

    try:
        # ── Phase 1 ───────────────────────────────────────────────────────
        metadata_path = os.path.join(DIRS["chunks"], "metadata.json")
        if args.skip_phase1:
            if not os.path.exists(metadata_path):
                _err(f"--skip-phase1 used but {metadata_path} not found. Run without skip first.")
                sys.exit(1)
            _warn("Skipping Phase 1 — using existing metadata.json")
        else:
            metadata_path = run_phase1(args.audio)

        # ── Phase 2 ───────────────────────────────────────────────────────
        transcripts_path = os.path.join(DIRS["transcripts"], "all_transcripts_final.json")
        if args.skip_phase2:
            if not os.path.exists(transcripts_path):
                _err(f"--skip-phase2 used but {transcripts_path} not found.")
                sys.exit(1)
            _warn("Skipping Phase 2 — using existing transcripts")
        else:
            transcripts_path = run_phase2(metadata_path, model_size=args.model)

        # ── Phase 3 ───────────────────────────────────────────────────────
        analysis_path = os.path.join(DIRS["analysis"], "all_analysis.json")
        if args.skip_phase3:
            if not os.path.exists(analysis_path):
                _err(f"--skip-phase3 used but {analysis_path} not found.")
                sys.exit(1)
            _warn("Skipping Phase 3 — using existing analysis")
        else:
            analysis_path = run_phase3(transcripts_path)

        # ── Phase 4 ───────────────────────────────────────────────────────
        decisions_path = run_phase4(analysis_path)

        # ── Phase 5 ───────────────────────────────────────────────────────
        fired_alerts = run_phase5_alerts(decisions_path)

        # ── Phase 6 ───────────────────────────────────────────────────────
        alert_log = [{"incident": iid, "reason": "AUTO",
                      "time": datetime.now().strftime("%H:%M:%S"),
                      "category":"?","severity":"HIGH",
                      "sms_sent":True,"email_sent":True,"score":0,"text":""}
                     for iid in fired_alerts]
        pdf_path = run_phase6_report(decisions_path, alert_log)

        # ── Final summary ─────────────────────────────────────────────────
        total_time = time.time() - pipeline_start
        print_final_summary(args.audio, decisions_path, pdf_path, total_time, fired_alerts)

        # ── Launch dashboard ──────────────────────────────────────────────
        if not args.no_dashboard:
            print("  Launching dashboard in 3 seconds...")
            print("  (Press Ctrl+C to cancel)\n")
            time.sleep(3)
            os.system("streamlit run pipeline/phase5_alerts/dashboard.py")

    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        _err(f"Pipeline failed: {e}")
        logger.exception("Pipeline error")
        raise


if __name__ == "__main__":
    main()