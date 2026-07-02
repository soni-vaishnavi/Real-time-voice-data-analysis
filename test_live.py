"""
test_live.py
=============
Live Mode Integration Test

Tests the complete live pipeline:
  Mic/File → API queue → Worker → Whisper translate →
  Emotion → BART → Score → Zone → DB → (Alerts if RED)

Usage:
    # Terminal 1: Start the API
    python main.py --mode live --no-dashboard

    # Terminal 2: Run this test
    python test_live.py                      # submit existing chunks to API
    python test_live.py --wait 120           # wait up to 120s for processing
    python test_live.py --check-only         # just check API health + current results
    python test_live.py --inject "bachao ambulance bulao"  # inject a text scenario

What this tests:
  1. API is running and accepting submissions
  2. Queue processes chunks without dropping
  3. Worker transcribes + scores correctly
  4. DB persists results (survives between calls)
  5. RED incidents fire alerts (dry-run or real)
  6. GET /chunks returns correct data
  7. GET /alerts shows open RED incidents
"""

import os
import sys
import json
import time
import wave
import struct
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_URL = "http://localhost:8000"


def _check_api(url: str) -> dict:
    """Check API health. Returns health dict or empty dict if unreachable."""
    try:
        import requests
        r = requests.get(f"{url}/health", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        return {"error": str(e)}


def _make_tone_wav(path: str, freq: int = 440, duration: float = 5.0, sample_rate: int = 16000):
    """
    Generate a tone WAV file.
    Used for testing when no real mic audio is available.
    """
    n = int(sample_rate * duration)
    samples = [int(3000 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def cmd_check(url: str):
    """Just check API health and print current results."""
    print(f"\n{'='*60}")
    print(f"  LIVE MODE STATUS CHECK")
    print(f"  API: {url}")
    print(f"{'='*60}")

    h = _check_api(url)
    if "error" in h:
        print(f"\n  ERROR: Cannot reach API — {h['error']}")
        print(f"  Start with: python main.py --mode live --no-dashboard")
        return False

    print(f"\n  API Status:    {h.get('status', '?')}")
    w = h.get("worker", {})
    print(f"  Worker:        {'running' if w.get('running') else 'STOPPED'}")
    print(f"  Models loaded: {w.get('models_loaded', False)}")
    print(f"  BART ready:    {h.get('models',{}).get('bart_ready', False)}")
    print(f"  wav2vec2:      {h.get('models',{}).get('wav2vec2_ready', False)}")
    print(f"  Chunks in DB:  {h.get('results_count', 0)}")
    q = h.get("queue", {})
    print(f"  Queue:         {q.get('current_size',0)}/{q.get('maxsize',10)} | total_in={q.get('total_in',0)}")
    print(f"  Processed:     {w.get('chunks_processed', 0)} chunks")

    # Current results
    try:
        import requests
        summary = requests.get(f"{url}/chunks/summary", timeout=3).json()
        print(f"\n  Zone distribution (all sessions):")
        print(f"    GREEN:  {summary.get('green',0)}")
        print(f"    YELLOW: {summary.get('yellow',0)}")
        print(f"    RED:    {summary.get('red',0)}")
        print(f"    Avg score: {summary.get('avg_score_pct',0)}%")
    except Exception:
        pass

    # Open alerts
    try:
        import requests
        alerts = requests.get(f"{url}/alerts", timeout=3).json()
        open_a = alerts.get("open_alerts", 0)
        total_r = alerts.get("total_red", 0)
        print(f"\n  RED incidents: {total_r} total | {open_a} open (awaiting confirmation)")
        if open_a > 0:
            for a in alerts.get("alerts", []):
                print(f"    INC: {a.get('incident_id','?')} | {a.get('emergency_category','?')} | "
                      f"score={a.get('final_score',0):.2f} | '{(a.get('text') or '')[:50]}'")
    except Exception:
        pass

    return True


def cmd_submit(url: str, wait_sec: int = 60):
    """Submit existing chunks from output/chunks/ to the API."""
    print(f"\n{'='*60}")
    print(f"  SUBMIT EXISTING CHUNKS TO LIVE API")
    print(f"  API: {url}")
    print(f"{'='*60}")

    if not _check_api(url):
        print(f"\n  API not running. Start with: python main.py --mode live --no-dashboard")
        return

    import requests

    meta_path = "output/chunks/metadata.json"
    if not os.path.exists(meta_path):
        print(f"\n  No chunks found at {meta_path}")
        print("  Run python main.py first to generate audio chunks.")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    chunks = [(m["file_path"], m["chunk_id"], m["start"])
              for m in metadata if os.path.exists(m.get("file_path",""))]

    if not chunks:
        print("\n  No WAV files found. Run python main.py --skip-phase2 to regenerate chunks.")
        return

    session_id = f"test_{int(time.time())}"
    print(f"\n  Session ID: {session_id}")
    print(f"  Submitting {len(chunks)} chunks...\n")

    submitted = []
    for i, (path, cid, start) in enumerate(chunks):
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    f"{url}/audio/submit",
                    files={"file": (os.path.basename(path), f, "audio/wav")},
                    data={"session_id": session_id, "chunk_index": str(i), "chunk_start": str(start)},
                    timeout=5,
                )
            data   = r.json()
            status = data.get("status", "?")
            qsize  = data.get("queue_size", "?")
            submitted.append(cid)
            print(f"  [{i+1:2}/{len(chunks)}] {cid} → {status:<10} queue={qsize}")
        except Exception as e:
            print(f"  [{i+1:2}/{len(chunks)}] {cid} → ERROR: {e}")

    print(f"\n  Submitted {len(submitted)}/{len(chunks)} chunks.")
    print(f"  Waiting up to {wait_sec}s for processing...")
    print(f"  (Each chunk ~5-15s with Whisper small + BART)\n")

    # Poll for results
    deadline   = time.time() + wait_sec
    prev_count = 0
    last_zones = {}

    while time.time() < deadline:
        time.sleep(8)
        try:
            r = requests.get(f"{url}/chunks",
                             params={"source": "db", "limit": 500}, timeout=5)
            all_chunks = r.json().get("chunks", [])
            ours       = [c for c in all_chunks if c.get("session_id") == session_id]
            count      = len(ours)

            if count != prev_count:
                zones  = [c.get("score",{}).get("zone","?") for c in ours]
                newest = ours[-1] if ours else {}
                ns     = newest.get("score", {})
                zone_s = f"G={zones.count('GREEN')} Y={zones.count('YELLOW')} R={zones.count('RED')}"
                text   = (newest.get("text") or "")[:40]
                print(f"  Processed: {count:2}/{len(submitted)} | {zone_s} | latest='{text}'")
                prev_count = count
                last_zones = {c.get("chunk_id"): c.get("score",{}).get("zone","?") for c in ours}

                if count >= len(submitted):
                    print(f"\n  All chunks processed!")
                    break
        except Exception as e:
            print(f"  Poll error: {e}")

    remaining = len(submitted) - prev_count
    if remaining > 0:
        print(f"\n  Timeout — {remaining} chunks still in queue.")
        print("  Increase --wait value or check if worker is still loading models.")

    # Final results
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  (session: {session_id})")
    print(f"{'='*60}")

    try:
        r = requests.get(f"{url}/chunks", params={"source": "db", "limit": 500}, timeout=10)
        all_chunks = r.json().get("chunks", [])
        ours       = [c for c in all_chunks if c.get("session_id") == session_id]

        zones  = [c.get("score",{}).get("zone","?") for c in ours]
        scores = [c.get("score",{}).get("final_score",0.0) for c in ours]
        avg_s  = round(sum(scores)/max(1,len(scores))*100, 1)

        print(f"\n  Processed: {len(ours)}/{len(submitted)}")
        print(f"  GREEN:     {zones.count('GREEN')}")
        print(f"  YELLOW:    {zones.count('YELLOW')}")
        print(f"  RED:       {zones.count('RED')}")
        print(f"  Avg score: {avg_s}%")

        print(f"\n  {'Chunk':<14} {'Zone':<8} {'Score':>6}  {'Category':<16} {'Text'}")
        print(f"  {'-'*70}")
        for c in sorted(ours, key=lambda x: x.get("chunk_start", 0)):
            s    = c.get("score", {})
            zone = s.get("zone", "?")
            sc   = s.get("final_score", 0.0)
            cat  = s.get("emergency_category", "?")
            text = (c.get("text") or "")[:40]
            cid  = c.get("chunk_id","?")[-10:]
            mark = ">>" if zone == "RED" else ("! " if zone == "YELLOW" else "  ")
            print(f"  {mark}{cid:<12} {zone:<8} {sc*100:5.1f}%  {cat:<16} '{text}'")

        # Alerts
        flagged = [c for c in ours if c.get("score",{}).get("zone") in ("RED","YELLOW")]
        if flagged:
            print(f"\n  FLAGGED ({len(flagged)}):")
            for c in flagged:
                s  = c.get("score", {})
                kw = c.get("keywords_found", [])
                print(f"    {c.get('chunk_id')} | {s.get('zone')} | {s.get('emergency_category')} | "
                      f"score={s.get('final_score',0):.2f} | kws={kw[:3]}")
                print(f"      text: '{(c.get('text') or '')[:70]}'")

    except Exception as e:
        print(f"  Could not fetch results: {e}")


def cmd_inject(url: str, text: str):
    """
    Inject a text scenario and print what the classification would be.
    Does NOT require the API — uses local pipeline directly.
    """
    print(f"\n{'='*60}")
    print(f"  TEXT SCENARIO INJECTION")
    print(f"  Text: '{text}'")
    print(f"{'='*60}\n")

    import logging
    logging.getLogger().setLevel(logging.WARNING)

    from pipeline.phase3_analysis.emotion_detector import detect_emotion, resolve_sarcasm_conflict, get_emotion_model
    from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
    from pipeline.phase3_analysis.emergency_detector import get_emergency_model, detect_emergency
    from pipeline.phase2_stt.keyword_normalizer import detect_keywords, get_keyword_summary
    from pipeline.phase4_decision.scorer import compute_score
    from pipeline.phase4_decision.zone_classifier import classify_zone

    print("  Loading models...")
    get_emotion_model()
    get_emergency_model(wait=True)
    print(f"  Analyzing: '{text}'\n")

    emotion  = detect_emotion(text)
    sarcasm  = detect_sarcasm(text)
    sarc_res = resolve_sarcasm_conflict(emotion, sarcasm)
    emergency= detect_emergency(text, wait_for_model=False)

    kw_found = detect_keywords(text)
    kw_sum   = get_keyword_summary(kw_found)
    kw_boost = kw_sum.get("total_boost", 0.0)
    kw_cat   = kw_sum.get("top_category", None)
    kw_list  = kw_sum.get("keywords_list", [])

    if kw_boost > 0 and kw_cat and kw_cat.lower() in emergency.get("all_scores", {}):
        boosted = min(emergency["all_scores"][kw_cat.lower()] + kw_boost, 1.0)
        emergency["all_scores"][kw_cat.lower()] = round(boosted, 4)
        nt = max(emergency["all_scores"], key=emergency["all_scores"].get)
        emergency.update({"top_category": nt, "top_score": emergency["all_scores"][nt],
                          "is_emergency": nt != "normal" and emergency["all_scores"][nt] >= 0.55})

    chunk_data = {
        "emotion_analysis":  {"emotion": emotion, "sarcasm": sarcasm, "sarcasm_resolution": sarc_res},
        "emergency_analysis": emergency,
        "keyword_analysis":  {"total_boost": kw_boost, "top_category": kw_cat, "keywords_list": kw_list},
    }
    score = compute_score(chunk_data)
    classify_zone(score)

    zone = score.get("zone", "?")
    sc   = score.get("final_score", 0.0)
    cat  = score.get("emergency_category", "?")
    sev  = score.get("severity", "?")
    emo  = score.get("dominant_emotion", "?")

    zone_icon = {"RED":"🔴","YELLOW":"🟡","GREEN":"🟢"}.get(zone, "?")

    print(f"  {zone_icon} ZONE:       {zone}")
    print(f"  Score:        {sc:.3f} ({round(sc*100)}%)")
    print(f"  Category:     {cat}")
    print(f"  Severity:     {sev}")
    print(f"  Emotion:      {emo}")
    print(f"  Keywords:     {kw_list}")
    print(f"  Is emergency: {emergency.get('is_emergency')}")
    print(f"  Auto-alert:   {score.get('auto_alert')}")

    if zone == "RED":
        print(f"\n  >> ALERT WOULD FIRE: SMS + Email (if configured)")
    elif zone == "YELLOW":
        print(f"\n  >> Dashboard warning shown, operator review needed")
    else:
        print(f"\n  >> Safe — no alert")

    # Show BART scores
    print(f"\n  BART scores:")
    for cat_name, sc_val in sorted(emergency.get("all_scores",{}).items(), key=lambda x: -x[1]):
        bar = "#" * int(sc_val * 30)
        print(f"    {cat_name:<15} {sc_val:.3f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Live Mode Tester")
    parser.add_argument("--check-only", action="store_true", help="Just check API health")
    parser.add_argument("--wait",  type=int, default=90,   help="Seconds to wait for processing")
    parser.add_argument("--api",   type=str, default=API_URL, help="API URL")
    parser.add_argument("--inject",type=str, default=None, help="Inject a text scenario")
    args = parser.parse_args()

    if args.inject:
        cmd_inject(args.api, args.inject)
    elif args.check_only:
        cmd_check(args.api)
    else:
        ok = cmd_check(args.api)
        if ok:
            cmd_submit(args.api, wait_sec=args.wait)


if __name__ == "__main__":
    main()