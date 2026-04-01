"""
Test Phase 4 — Decision Engine

Run from voice_surveillance/ folder:
    python test_phase4.py

Expects:
    output/analysis/all_analysis.json  (from Phase 3)

What Phase 4 does:
    scorer.py          → combine emotion + emergency + keyword scores
    zone_classifier.py → assign GREEN / YELLOW / RED
    trend_analyzer.py  → detect escalation patterns, group incidents

Output:
    output/decisions/all_decisions.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.phase4_decision.scorer         import score_all_chunks
from pipeline.phase4_decision.zone_classifier import classify_all_zones
from pipeline.phase4_decision.trend_analyzer  import apply_trend_analysis, get_trend_summary

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ANALYSIS_PATH = "output/analysis/all_analysis.json"
OUTPUT_DIR    = "output/decisions"
OUTPUT_FILE   = "output/decisions/all_decisions.json"


def inject_fake_red(chunks):
    """
    Inject one fake RED chunk so we can see the full decision flow.
    Remove this in production — real emergency audio will trigger naturally.
    """
    fake = {
        "chunk_id":    "chunk_FAKE_RED",
        "chunk_start": 999.0,
        "text":        "Bachao bachao! Koi ambulance bulao jaldi! Bahut dard ho raha hai!",
        "emotion_analysis": {
            "emotion": {
                "dominant_emotion": "fear",
                "dominant_score":   0.95,
                "emergency_weight": 1.0,
            },
            "sarcasm":             {"is_sarcastic": False, "sarcasm_score": 0.02},
            "conflict_resolution": {"resolution": "real", "score_penalty": 0.0},
        },
        "emergency_analysis": {
            "top_category": "medical",
            "top_score":    0.85,
            "is_emergency": True,
            "risk_level":   0.90,
        },
        "keyword_analysis": {
            "total_boost":  0.40,
            "top_category": "medical",
        },
    }
    print("WARNING: Injecting 1 fake RED chunk (bachao/ambulance scenario)")
    print("   Remove this block when testing with real emergency audio\n")
    return chunks + [fake]


def print_summary(chunks, trend_summary):
    print("\n" + "=" * 60)
    print("  PHASE 4 — DECISION ENGINE RESULTS")
    print("=" * 60)

    # Per-chunk table
    print(
        "\n{:<14} {:<9} {:<8} {:<12} {:<16} {}".format(
            "Chunk", "Zone", "Score", "Emotion", "Category", "Incident"
        )
    )
    print("-" * 80)

    green = yellow = red = 0

    for c in chunks:
        s      = c.get("score", {})
        zone   = s.get("zone", "?")
        emoji  = s.get("zone_emoji", "")
        score  = s.get("final_score", 0.0)
        emotion = s.get("dominant_emotion", "?")
        cat    = s.get("emergency_category", "?")
        inc    = s.get("incident_id") or "-"
        trend  = " TREND-UP" if s.get("trend_upgraded") else ""
        rising = " RISING"   if s.get("rising_trend")   else ""

        print(
            "{:<14} {}{:<8} {:<8.3f} {:<12} {:<16} {}{}{}".format(
                c["chunk_id"], emoji, zone, score, emotion, cat, inc, trend, rising
            )
        )

        if zone == "GREEN":   green += 1
        elif zone == "YELLOW": yellow += 1
        elif zone == "RED":    red += 1

    # Zone distribution
    total = len(chunks)
    print("\n" + "-" * 60)
    print("  ZONE DISTRIBUTION")
    print("-" * 60)

    def bar(n):
        return "█" * int((n / total) * 30) if total else ""

    print("  GREEN  {} {} ({:.0f}%)".format(bar(green),  green,  green  / total * 100))
    print("  YELLOW {} {} ({:.0f}%)".format(bar(yellow), yellow, yellow / total * 100))
    print("  RED    {} {} ({:.0f}%)".format(bar(red),    red,    red    / total * 100))

    # Trend summary
    print("\n" + "-" * 60)
    print("  TREND ANALYSIS")
    print("-" * 60)
    print("  Incidents detected    :", trend_summary["total_incidents"])
    print("  Incident IDs          :", trend_summary["incident_ids"] or "None")
    print("  Trend-upgraded chunks :", trend_summary["trend_upgraded_count"])
    print("  Rising trend chunks   :", trend_summary["rising_trend_count"])

    # RED chunk alert details
    red_chunks = [c for c in chunks if c.get("score", {}).get("zone") == "RED"]
    if red_chunks:
        print("\n" + "-" * 60)
        print("  RED CHUNKS — ALERT DETAILS")
        print("-" * 60)
        for c in red_chunks:
            s    = c.get("score", {})
            sev  = s.get("severity", "?")
            auto = "AUTO-ALERT" if s.get("auto_alert") else "NEEDS HUMAN CONFIRM"
            print("\n  Chunk    :", c["chunk_id"])
            print("  Category :", s.get("emergency_category", "?").upper())
            print("  Severity :", sev)
            print("  Score    :", round(s.get("final_score", 0), 3))
            print("  Action   :", auto)
            print("  Incident :", s.get("incident_id", "N/A"))
            if c.get("text"):
                print('  Text     : "{}"'.format(c["text"][:70]))

    # Score breakdown example
    print("\n" + "-" * 60)
    print("  SCORE BREAKDOWN EXAMPLE (first chunk with text)")
    print("-" * 60)
    for c in chunks:
        if c.get("text") and len(c.get("text", "")) > 10:
            s    = c.get("score", {})
            comp = s.get("components", {})
            print("  Chunk :", c["chunk_id"])
            print('  Text  : "{}"'.format(c["text"][:60]))
            print("  Emotion component   : {:.4f}  (weight 0.35)".format(comp.get("emotion_component", 0)))
            print("  Emergency component : {:.4f}  (weight 0.40)".format(comp.get("emergency_component", 0)))
            print("  Keyword component   : {:.4f}  (weight 0.25)".format(comp.get("keyword_component", 0)))
            print("  Sarcasm deduction   :-{:.4f}".format(comp.get("sarcasm_deduction", 0)))
            print("  " + "-" * 35)
            print("  FINAL SCORE         : {:.4f}  -> {} {}".format(
                s.get("final_score", 0), s.get("zone_emoji", ""), s.get("zone", "")
            ))
            break

    print("\n" + "=" * 60)
    print("  PHASE 4 COMPLETE")
    print("=" * 60)
    print("\n  Decisions saved ->", OUTPUT_FILE)
    print("  Ready for Phase 5 (Dashboard + Alerts)")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  TESTING PHASE 4: DECISION ENGINE")
    print("=" * 60)

    if not os.path.exists(ANALYSIS_PATH):
        print("\nERROR: Analysis not found at", ANALYSIS_PATH)
        print("   Run test_phase3.py first.")
        sys.exit(1)

    print("\nFound analysis:", ANALYSIS_PATH)

    with open(ANALYSIS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print("  ", len(chunks), "scored chunks loaded")

    # Inject fake RED for testing (remove when using real emergency audio)
    chunks = inject_fake_red(chunks)

    # Step 1: Score
    print("\n--- Step 1: Scoring ---")
    chunks = score_all_chunks(chunks)

    # Step 2: Zone Classification
    print("\n--- Step 2: Zone Classification ---")
    chunks = classify_all_zones(chunks)

    # Step 3: Trend Analysis
    print("\n--- Step 3: Trend Analysis ---")
    chunks = apply_trend_analysis(chunks)
    trend_summary = get_trend_summary(chunks)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Print summary
    print_summary(chunks, trend_summary)


if __name__ == "__main__":
    main()