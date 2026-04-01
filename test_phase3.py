"""
Test Phase 3 — Emotion + Emergency + Scoring

Run from voice_surveillance/ folder:
    python test_phase3.py

Expects:
    output/transcripts/all_transcripts_final.json  (from Phase 2)

Produces:
    output/analysis/all_analysis.json
    output/analysis/chunk_XXX_analysis.json (per chunk)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.phase3_analysis.analyzer import run_phase3

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRANSCRIPTS_PATH = "output/transcripts/all_transcripts_final.json"
OUTPUT_DIR       = "output/analysis"


def print_summary(results):
    """Print readable Phase 3 summary"""
    print("\n" + "=" * 60)
    print("  PHASE 3 ANALYSIS RESULTS")
    print("=" * 60)

    # Per-chunk results
    print(f"\n{'Chunk':<12} {'Zone':<8} {'Score':<8} {'Emotion':<12} {'Emergency':<16} {'Text Preview'}")
    print("-" * 90)

    green = yellow = red = 0
    emergencies = []

    for t in results:
        chunk_id  = t["chunk_id"]
        score     = t["score"]
        text      = t.get("text", "")[:45] + ("..." if len(t.get("text","")) > 45 else "")
        zone      = score["zone"]
        final     = score["final_score"]
        emotion   = score["dominant_emotion"]
        emergency = score["emergency_category"]
        emoji     = score["zone_emoji"]
        trend     = " ↑TREND" if score.get("trend_upgraded") else ""

        print(f"{chunk_id:<12} {emoji}{zone:<7} {final:<8.3f} {emotion:<12} {emergency:<16} {text}{trend}")

        if zone == "GREEN":   green += 1
        elif zone == "YELLOW": yellow += 1
        elif zone == "RED":    red += 1

        if score["is_emergency"]:
            emergencies.append({
                "chunk_id": chunk_id,
                "category": emergency,
                "score": final,
                "text": t.get("text", "")
            })

    # Zone distribution
    total = len(results)
    print("\n" + "-" * 60)
    print("  ZONE DISTRIBUTION")
    print("-" * 60)
    bar_g = "█" * int((green / total) * 30) if total else ""
    bar_y = "█" * int((yellow / total) * 30) if total else ""
    bar_r = "█" * int((red / total) * 30) if total else ""
    print(f"  🟢 GREEN   {bar_g:<30} {green} chunks ({green/total*100:.0f}%)")
    print(f"  🟡 YELLOW  {bar_y:<30} {yellow} chunks ({yellow/total*100:.0f}%)")
    print(f"  🔴 RED     {bar_r:<30} {red} chunks ({red/total*100:.0f}%)")

    # Emergencies detected
    print("\n" + "-" * 60)
    print("  EMERGENCIES DETECTED")
    print("-" * 60)
    if emergencies:
        for e in emergencies:
            print(f"  🚨 {e['chunk_id']} | {e['category'].upper()} | score={e['score']:.3f}")
            print(f"     \"{e['text'][:70]}\"")
    else:
        print("  ✅ No emergencies detected")
        print("     (Expected for normal conversation test audio)")

    # Score components example (first non-empty chunk)
    print("\n" + "-" * 60)
    print("  SCORE COMPONENTS EXAMPLE (first chunk with text)")
    print("-" * 60)
    for t in results:
        if t.get("text") and len(t.get("text", "")) > 10:
            s = t["score"]
            c = s["components"]
            print(f"  Chunk: {t['chunk_id']}")
            print(f"  Text:  \"{t['text'][:60]}\"")
            print(f"  Emotion component:   {c['emotion_component']:.4f}  (weight 0.35)")
            print(f"  Emergency component: {c['emergency_component']:.4f}  (weight 0.40)")
            print(f"  Keyword component:   {c['keyword_component']:.4f}  (weight 0.25)")
            print(f"  Sarcasm deduction:  -{c['sarcasm_deduction']:.4f}")
            print(f"  ─────────────────────────────")
            print(f"  FINAL SCORE:         {s['final_score']:.4f}  → {s['zone_emoji']} {s['zone']}")
            break

    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE ✅")
    print("=" * 60)
    print(f"\n  Analysis saved → {OUTPUT_DIR}/all_analysis.json")
    print(f"  Ready for Phase 4 (Decision Engine)")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  TESTING PHASE 3: ANALYSIS MODELS")
    print("=" * 60)

    # Check Phase 2 output exists
    if not os.path.exists(TRANSCRIPTS_PATH):
        print(f"\n❌ ERROR: Transcripts not found at {TRANSCRIPTS_PATH}")
        print("   Run test_phase2.py first to generate transcripts.")
        sys.exit(1)

    print(f"\n✅ Found transcripts: {TRANSCRIPTS_PATH}")

    # Load and show count
    with open(TRANSCRIPTS_PATH) as f:
        transcripts = json.load(f)
    print(f"   {len(transcripts)} chunks to analyze")

    print("\n⚠️  NOTE: First run will download models:")
    print("   - Emotion model (j-hartmann):  ~500MB")
    print("   - Sarcasm model (helinivan):   ~400MB")
    print("   - Emergency model (BART):      ~1.6GB")
    print("   Total: ~2.5GB — subsequent runs use cache\n")

    # Run Phase 3
    results = run_phase3(TRANSCRIPTS_PATH, OUTPUT_DIR)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()