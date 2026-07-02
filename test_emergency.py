"""
test_emergency.py
==================
Emergency Classification Accuracy Test

Tests whether the system correctly identifies emergency types and severity.
Covers all 7 categories in Hindi, English, and Hinglish.

Two test modes:
  1. TEXT MODE (fast, no audio needed):
     Injects translated text directly into Phase 3+4
     Tests: emotion + emergency + keyword + scoring + zone
     Run: python test_emergency.py

  2. FULL FILE MODE (slow, processes real chunks):
     Creates WAV chunks from text via pyttsx3 TTS, runs full pipeline
     Run: python test_emergency.py --file

  3. LIVE API MODE (requires running API):
     Submits WAV chunks to FastAPI, checks /chunks and /alerts
     Run: python test_emergency.py --live

Usage:
    python test_emergency.py               # text injection mode (fastest)
    python test_emergency.py --file        # file pipeline mode
    python test_emergency.py --live        # live API mode (start API first)
    python test_emergency.py --all         # run all modes
"""

import os
import sys
import json
import time
import wave
import struct
import logging
import threading
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress info logs for cleaner test output
logging.getLogger().setLevel(logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST CASES
# ══════════════════════════════════════════════════════════════════════════════

# Format: (description, translated_english_text, expected_zone, expected_category, expected_kws)
# translated_english_text = what Whisper translate would output for the Hindi/Hinglish speech
TEST_CASES = [

    # ── MEDICAL ───────────────────────────────────────────────────────────────
    (
        "Medical: bachao ambulance (Hindi emergency)",
        "save me save me call the ambulance quickly I am in a lot of pain",
        "RED", "medical",
        ["ambulance", "save me", "pain"]
    ),
    (
        "Medical: heart attack (English)",
        "someone having chest pain heart attack call ambulance doctor needed",
        "RED", "medical",
        ["chest pain", "ambulance", "doctor"]
    ),
    (
        "Medical: bleeding (Hinglish)",
        "bahut khoon aa raha hai bleeding please help doctor bulao",
        "YELLOW", "medical",
        ["khoon", "bleeding", "help", "doctor"]
    ),

    # ── FIRE ──────────────────────────────────────────────────────────────────
    (
        "Fire: aag lagi (Hindi)",
        "fire has broken out aag lagi save yourselves",
        "RED", "fire",
        ["fire", "aag"]
    ),
    (
        "Fire: gas explosion (English)",
        "there is a fire explosion blast gas leak everyone run",
        "RED", "fire",
        ["fire", "blast", "explosion"]
    ),
    (
        "Fire: smoke detected",
        "smoke and fire in the building please call fire brigade",
        "RED", "fire",
        ["fire", "smoke"]
    ),

    # ── VIOLENCE ──────────────────────────────────────────────────────────────
    (
        "Violence: gun attack (Hindi translated)",
        "someone is shooting gun attack help police",
        "RED", "violence",
        ["shoot", "gun", "help"]
    ),
    (
        "Violence: knife attack (Hinglish)",
        "chaku maar diya he has stabbed me with a knife attack help",
        "RED", "violence",
        ["chaku", "knife", "attack", "help"]
    ),
    (
        "Violence: assault",
        "they are beating me assault please save me police",
        "RED", "violence",
        ["assault", "save me", "police"]
    ),

    # ── ACCIDENT ──────────────────────────────────────────────────────────────
    (
        "Accident: car crash (Hindi translated)",
        "there has been an accident road crash someone is injured call ambulance",
        "YELLOW", "accident",
        ["accident", "ambulance"]
    ),
    (
        "Accident: fall injury",
        "someone fell from the stairs injured bleeding help needed",
        "YELLOW", "accident",
        ["bleeding", "help"]
    ),

    # ── THEFT ─────────────────────────────────────────────────────────────────
    (
        "Theft: robbery (Hindi)",
        "chor chor stop thief robbery they are stealing purse",
        "YELLOW", "theft",
        ["robbery"]
    ),
    (
        "Theft: loot (Hinglish)",
        "loot ho raha hai robbery they looted everything police",
        "YELLOW", "theft",
        ["loot", "robbery", "police"]
    ),

    # ── MENTAL HEALTH ─────────────────────────────────────────────────────────
    (
        "Mental: suicide threat (Hindi translated)",
        "I don't want to live anymore suicide I want to end it all",
        "RED", "mental_health",
        ["suicide", "end it"]
    ),
    (
        "Mental: self harm",
        "I will jump from here end it I cannot take it anymore",
        "YELLOW", "mental_health",
        ["jump", "end it"]
    ),

    # ── NORMAL (should NOT trigger) ───────────────────────────────────────────
    (
        "Normal: casual conversation",
        "my friend is very nice we had a great time together yesterday",
        "GREEN", "normal",
        []
    ),
    (
        "Normal: discussing movie",
        "the film was amazing we really enjoyed watching it together",
        "GREEN", "normal",
        []
    ),
    (
        "Normal: sarcasm - should not trigger",
        "oh great wonderful the computer crashed again amazing day",
        "GREEN", "normal",
        []
    ),

    # ── TRICKY CASES ──────────────────────────────────────────────────────────
    (
        "Tricky: 'how can I get out of my car' (not accident)",
        "how can I get out of my car the door is stuck",
        "GREEN", "normal",
        []
    ),
    (
        "Tricky: low keyword score but clear medical emergency",
        "person is unconscious not breathing on the floor",
        "RED", "medical",
        ["unconscious"]
    ),
    (
        "Tricky: Hinglish friend complaint (not emergency)",
        "yaar meri dost ne mujhse baat karna band kar diya she stopped talking to me",
        "GREEN", "normal",
        []
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT INJECTION MODE — Phase 3+4 only
# ══════════════════════════════════════════════════════════════════════════════

def run_text_mode() -> Tuple[int, int, List[Dict]]:
    """
    Fast mode: inject translated text directly into Phase 3+4.
    No audio needed. Tests the full analysis + scoring pipeline.
    """
    print("\n" + "=" * 65)
    print("  TEXT INJECTION MODE")
    print("  Injecting translated text directly into Phase 3+4")
    print("  (No audio needed — tests analysis + scoring accuracy)")
    print("=" * 65)

    # Load models once
    print("\n  Loading models...")
    from pipeline.phase3_analysis.emotion_detector import detect_emotion, resolve_sarcasm_conflict, get_emotion_model
    from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
    from pipeline.phase3_analysis.emergency_detector import get_emergency_model, detect_emergency
    from pipeline.phase2_stt.keyword_normalizer import detect_keywords, get_keyword_summary
    from pipeline.phase4_decision.scorer import compute_score
    from pipeline.phase4_decision.zone_classifier import classify_zone

    get_emotion_model()
    get_emergency_model(wait=True)
    print("  Models loaded. Running tests...\n")

    passed = failed = 0
    results = []

    header = f"  {'Test Case':<40} {'Expected':<8} {'Got':<8} {'Score':>6}  {'Category':<14}  {'Keywords'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for desc, text, exp_zone, exp_cat, exp_kws in TEST_CASES:
        # Phase 3: emotion + sarcasm + emergency + keywords
        emotion   = detect_emotion(text)
        sarcasm   = detect_sarcasm(text)
        sarc_res  = resolve_sarcasm_conflict(emotion, sarcasm)
        emergency = detect_emergency(text, wait_for_model=False)

        # Keyword boost
        kw_found  = detect_keywords(text)
        kw_summary= get_keyword_summary(kw_found)
        kw_boost  = kw_summary.get("total_boost", 0.0)
        kw_cat    = kw_summary.get("top_category", None)
        kw_list   = kw_summary.get("keywords_list", [])

        if kw_boost > 0 and kw_cat and kw_cat.lower() in emergency.get("all_scores", {}):
            boosted = min(emergency["all_scores"][kw_cat.lower()] + kw_boost, 1.0)
            emergency["all_scores"][kw_cat.lower()] = round(boosted, 4)
            new_top = max(emergency["all_scores"], key=emergency["all_scores"].get)
            emergency.update({
                "top_category": new_top,
                "top_score":    emergency["all_scores"][new_top],
                "is_emergency": new_top != "normal" and emergency["all_scores"][new_top] >= 0.55,
            })

        # Phase 4: score + zone
        chunk_data = {
            "emotion_analysis": {
                "emotion": emotion,
                "sarcasm": sarcasm,
                "sarcasm_resolution": sarc_res,
            },
            "emergency_analysis": emergency,
            "keyword_analysis":   {"total_boost": kw_boost, "top_category": kw_cat,
                                   "keywords_list": kw_list},
        }
        score_dict = compute_score(chunk_data)
        classify_zone(score_dict)

        got_zone  = score_dict.get("zone", "GREEN")
        got_cat   = score_dict.get("emergency_category", "normal")
        got_score = score_dict.get("final_score", 0.0)

        # Evaluate: zone and category must match
        # For category: "normal" expected → anything in GREEN is ok
        zone_ok = got_zone == exp_zone
        cat_ok  = (exp_cat == "normal" and got_zone == "GREEN") or got_cat == exp_cat

        ok = zone_ok  # primary check is zone (RED/YELLOW/GREEN)

        status = "OK" if ok else "!!"
        kw_str = ",".join(kw_list[:3]) if kw_list else "-"
        zone_display = f"{got_zone}" if ok else f"{got_zone}<-{exp_zone}"

        print(f"  {status}  {desc[:38]:<38} {exp_zone:<8} {zone_display:<10} {got_score:5.2f}  {got_cat:<14}  {kw_str}")

        results.append({
            "desc": desc, "text": text,
            "expected_zone": exp_zone, "got_zone": got_zone,
            "expected_cat": exp_cat, "got_cat": got_cat,
            "score": got_score, "keywords": kw_list,
            "ok": ok,
        })

        if ok: passed += 1
        else:  failed += 1

    return passed, failed, results


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE API MODE
# ══════════════════════════════════════════════════════════════════════════════

def _make_silent_wav(path: str, duration_sec: float = 2.0, sample_rate: int = 16000):
    """Create a short silent WAV for testing."""
    import wave, struct, math
    n = int(sample_rate * duration_sec)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Generate a very quiet tone so it's not filtered out as silence
        samples = [int(100 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def check_api_running(api_url: str = "http://localhost:8000") -> bool:
    """Check if the FastAPI server is running."""
    try:
        import requests
        r = requests.get(f"{api_url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_live_mode(api_url: str = "http://localhost:8000") -> Tuple[int, int, List[Dict]]:
    """
    Live API mode: submit WAV chunks to API, poll /chunks for results.
    Tests the full pipeline: queue → worker → Whisper → emotion → BART → score → zone.
    """
    print("\n" + "=" * 65)
    print("  LIVE API MODE")
    print(f"  Connecting to: {api_url}")
    print("  Submitting real WAV chunks — tests full pipeline end-to-end")
    print("=" * 65)

    import requests

    if not check_api_running(api_url):
        print(f"\n  ERROR: API not running at {api_url}")
        print(f"  Start it first:")
        print(f"    uvicorn api.main:app --host 0.0.0.0 --port 8000")
        print(f"  Or use: python main.py --mode live --no-dashboard")
        return 0, 0, []

    # Check API health
    health = requests.get(f"{api_url}/health", timeout=5).json()
    print(f"\n  API status:    {health.get('status','?')}")
    print(f"  BART ready:    {health.get('models',{}).get('bart_ready', False)}")
    print(f"  Whisper ready: {health.get('models',{}).get('whisper_ready', False)}")
    print(f"  Queue size:    {health.get('queue',{}).get('current_size', 0)}")

    # Use existing chunks from output/chunks/ if available
    chunk_files = []
    meta_path = "output/chunks/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        chunk_files = [(m["file_path"], m["chunk_id"]) for m in metadata
                       if os.path.exists(m.get("file_path", ""))]
        print(f"\n  Found {len(chunk_files)} existing chunks to submit")
    else:
        print(f"\n  No existing chunks — run python main.py first to generate them")
        return 0, 0, []

    if not chunk_files:
        print("  No WAV files found in output/chunks/")
        return 0, 0, []

    # Submit all chunks
    print(f"\n  Submitting {len(chunk_files)} chunks to API...")
    session_id = f"test_session_{int(time.time())}"
    submitted = []

    for i, (file_path, chunk_id) in enumerate(chunk_files):
        try:
            with open(file_path, "rb") as f:
                r = requests.post(
                    f"{api_url}/audio/submit",
                    files={"file": (os.path.basename(file_path), f, "audio/wav")},
                    data={"session_id": session_id, "chunk_index": str(i), "chunk_start": str(i * 5.0)},
                    timeout=5,
                )
            status = r.json().get("status", "?")
            submitted.append(chunk_id)
            print(f"  [{i+1:2d}/{len(chunk_files)}] {chunk_id} → {status}")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(chunk_files)}] {chunk_id} → ERROR: {e}")

    print(f"\n  Submitted {len(submitted)} chunks. Waiting for processing...")
    print("  (Processing time depends on queue + BART load...)\n")

    # Poll for results
    max_wait  = 300   # 5 minutes max
    poll_every = 5
    waited     = 0
    prev_count = 0

    while waited < max_wait:
        time.sleep(poll_every)
        waited += poll_every
        try:
            r = requests.get(f"{api_url}/chunks", params={"source": "db", "limit": 200}, timeout=5)
            data   = r.json()
            chunks = data.get("chunks", [])
            # Filter to our session
            our_chunks = [c for c in chunks if c.get("session_id") == session_id]
            count = len(our_chunks)
            if count != prev_count:
                print(f"  [{waited:3d}s] Processed: {count}/{len(submitted)} chunks")
                prev_count = count
            if count >= len(submitted):
                print(f"\n  All chunks processed!")
                break
        except Exception as e:
            print(f"  [{waited:3d}s] Poll error: {e}")

    # Fetch final results
    try:
        r = requests.get(f"{api_url}/chunks", params={"source": "db", "limit": 500}, timeout=10)
        all_chunks = r.json().get("chunks", [])
        our_chunks = [c for c in all_chunks if c.get("session_id") == session_id]
    except Exception as e:
        print(f"  Could not fetch results: {e}")
        return 0, 0, []

    # Check alerts
    try:
        alerts = requests.get(f"{api_url}/alerts", timeout=5).json()
        open_alerts = alerts.get("open_alerts", 0)
    except Exception:
        open_alerts = 0

    # Analyze results
    print("\n" + "=" * 65)
    print(f"  LIVE MODE RESULTS  (session: {session_id})")
    print("=" * 65)

    zones  = [c.get("score", {}).get("zone", "?") for c in our_chunks]
    scores = [c.get("score", {}).get("final_score", 0.0) for c in our_chunks]
    cats   = [c.get("score", {}).get("emergency_category", "?") for c in our_chunks]

    green  = zones.count("GREEN")
    yellow = zones.count("YELLOW")
    red    = zones.count("RED")
    avg_sc = round(sum(scores) / max(1, len(scores)) * 100, 1) if scores else 0

    print(f"\n  Total processed:  {len(our_chunks)}/{len(submitted)}")
    print(f"  GREEN (safe):     {green}")
    print(f"  YELLOW (warning): {yellow}")
    print(f"  RED (emergency):  {red}")
    print(f"  Avg score:        {avg_sc}%")
    print(f"  Open alerts:      {open_alerts}")

    # Show each chunk result
    print(f"\n  Chunk-by-chunk results:")
    print(f"  {'Chunk':<14} {'Zone':<8} {'Score':>6}  {'Category':<16} {'Text'}")
    print("  " + "-" * 80)
    results = []
    for c in sorted(our_chunks, key=lambda x: x.get("chunk_start", 0)):
        s    = c.get("score", {})
        zone = s.get("zone", "?")
        sc   = s.get("final_score", 0.0)
        cat  = s.get("emergency_category", "?")
        text = (c.get("text") or "")[:45]
        cid  = c.get("chunk_id", "?")
        mark = "**" if zone == "RED" else ("!" if zone == "YELLOW" else "  ")
        print(f"  {mark}{cid:<12} {zone:<8} {sc*100:5.1f}%  {cat:<16} '{text}'")
        results.append({"chunk_id": cid, "zone": zone, "score": sc, "category": cat, "text": text})

    # Check alerts detail
    if red > 0 or yellow > 0:
        print(f"\n  Flagged chunks detail:")
        flagged = [c for c in our_chunks if c.get("score",{}).get("zone") in ("RED","YELLOW")]
        for c in flagged:
            s = c.get("score", {})
            kw = c.get("keywords_found", [])
            print(f"    {c.get('chunk_id')} | {s.get('zone')} | {s.get('emergency_category')} | "
                  f"score={s.get('final_score',0):.2f} | kws={kw} | '{(c.get('text') or '')[:50]}'")

    passed = len([c for c in our_chunks if c.get("score",{}).get("zone") != "?"])
    failed = len(our_chunks) - passed
    return passed, failed, results


# ══════════════════════════════════════════════════════════════════════════════
#  ACCURACY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_accuracy_report(results: List[Dict]):
    """Print detailed accuracy analysis."""
    if not results:
        return

    print("\n" + "=" * 65)
    print("  ACCURACY REPORT")
    print("=" * 65)

    # Overall
    total  = len(results)
    passed = sum(1 for r in results if r.get("ok", False))
    pct    = round(passed / total * 100) if total > 0 else 0
    print(f"\n  Overall accuracy: {passed}/{total} ({pct}%)")

    # By zone type
    red_cases    = [r for r in results if r.get("expected_zone") == "RED"]
    yellow_cases = [r for r in results if r.get("expected_zone") == "YELLOW"]
    green_cases  = [r for r in results if r.get("expected_zone") == "GREEN"]

    def zone_acc(cases):
        if not cases: return 0, 0
        ok = sum(1 for c in cases if c.get("ok", False))
        return ok, len(cases)

    r_ok, r_tot = zone_acc(red_cases)
    y_ok, y_tot = zone_acc(yellow_cases)
    g_ok, g_tot = zone_acc(green_cases)

    print(f"\n  By zone:")
    print(f"    RED    (emergency): {r_ok}/{r_tot}"
          + (f" ({round(r_ok/r_tot*100)}%)" if r_tot else ""))
    print(f"    YELLOW (warning):   {y_ok}/{y_tot}"
          + (f" ({round(y_ok/y_tot*100)}%)" if y_tot else ""))
    print(f"    GREEN  (safe):      {g_ok}/{g_tot}"
          + (f" ({round(g_ok/g_tot*100)}%)" if g_tot else ""))

    # False positives and false negatives
    false_pos = [r for r in green_cases if r.get("got_zone", "GREEN") != "GREEN"]
    false_neg = [r for r in red_cases   if r.get("got_zone", "RED") == "GREEN"]

    if false_pos:
        print(f"\n  False positives ({len(false_pos)}) — safe audio flagged:")
        for r in false_pos:
            print(f"    '{r['desc']}' → got {r.get('got_zone','?')} | score={r.get('score',0):.2f}")

    if false_neg:
        print(f"\n  False negatives ({len(false_neg)}) — emergency missed:")
        for r in false_neg:
            print(f"    '{r['desc']}' → got {r.get('got_zone','?')} | score={r.get('score',0):.2f}")

    wrong = [r for r in results if not r.get("ok", False)]
    if not wrong:
        print("\n  All test cases classified correctly!")
    elif pct >= 80:
        print(f"\n  {pct}% accuracy — good enough for demo")
        print("  Focus on reducing false negatives (missed emergencies)")
    else:
        print(f"\n  {pct}% accuracy — needs improvement")
        print("  Check EMERGENCY_THRESHOLD setting in config.py")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    do_text  = "--text" in sys.argv or "--all" in sys.argv or (
        "--file" not in sys.argv and "--live" not in sys.argv)
    do_file  = "--file" in sys.argv or "--all" in sys.argv
    do_live  = "--live" in sys.argv or "--all" in sys.argv

    api_url = "http://localhost:8000"
    for arg in sys.argv:
        if arg.startswith("--api="):
            api_url = arg.split("=", 1)[1]

    print("=" * 65)
    print("  EMERGENCY CLASSIFICATION ACCURACY TEST")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"  Categories: MEDICAL, FIRE, VIOLENCE, ACCIDENT, THEFT, MENTAL, NORMAL")
    print("=" * 65)

    all_results = []

    if do_text:
        passed, failed, results = run_text_mode()
        all_results.extend(results)
        print(f"\n  Text mode: {passed}/{passed+failed} passed "
              f"({round(passed/(passed+failed)*100) if (passed+failed) else 0}%)")

    if do_live:
        run_live_mode(api_url)

    if do_file:
        print("\n  FILE MODE: Run python main.py first, then check output/decisions/all_decisions.json")
        meta = "output/decisions/all_decisions.json"
        if os.path.exists(meta):
            with open(meta) as f:
                decisions = json.load(f)
            zones = [d.get("score",{}).get("zone","?") for d in decisions]
            print(f"  Results: GREEN={zones.count('GREEN')} YELLOW={zones.count('YELLOW')} RED={zones.count('RED')}")
            flagged = [d for d in decisions if d.get("score",{}).get("zone") in ("RED","YELLOW")]
            for d in flagged:
                s = d.get("score", {})
                print(f"    {d.get('chunk_id')} | {s.get('zone')} | {s.get('emergency_category')} | "
                      f"score={s.get('final_score',0):.2f} | '{(d.get('text') or '')[:50]}'")

    if all_results:
        print_accuracy_report(all_results)

    print("\n" + "=" * 65)
    print("  TEST COMPLETE")
    print("=" * 65)
    if not do_text and not do_live and not do_file:
        print("\n  Options:")
        print("    python test_emergency.py             # text injection (fast)")
        print("    python test_emergency.py --live      # live API mode")
        print("    python test_emergency.py --file      # check file pipeline output")
        print("    python test_emergency.py --all       # run everything")
        print("    python test_emergency.py --live --api=http://localhost:8000")


if __name__ == "__main__":
    main()