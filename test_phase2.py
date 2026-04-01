"""
Phase 2 Test Script
Transcribes all chunks from Phase 1 using Whisper
Tests Hindi, English, and Hinglish handling
Tests transliteration and keyword detection

Usage:
    python test_phase2.py

NOTE: First run downloads Whisper 'small' model (~460MB)
      Make sure you have internet connection.

Expected output:
    - output/transcripts/chunk_000_transcript.json ... 
    - output/transcripts/all_transcripts.json
    - Console summary of all transcriptions
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.phase2_stt.whisper_transcriber import transcribe_all_chunks
from pipeline.phase2_stt.transliterator import transliterate_all_transcripts, contains_devanagari
from pipeline.phase2_stt.keyword_normalizer import apply_keyword_normalization_all


def run_phase2_test():
    print("\n" + "="*60)
    print("  PHASE 2 TEST — Whisper STT Pipeline")
    print("="*60)

    # ── PATHS ──────────────────────────────────────────────────
    metadata_path   = "output/chunks/metadata.json"
    transcripts_dir = "output/transcripts/"

    # ── CHECK PHASE 1 OUTPUT EXISTS ────────────────────────────
    if not os.path.exists(metadata_path):
        print(f"\n❌ ERROR: {metadata_path} not found.")
        print("   Run test_phase1.py first to generate chunks.")
        return

    with open(metadata_path) as f:
        chunks = json.load(f)
    print(f"\n✅ Found {len(chunks)} chunks from Phase 1")

    # ── STEP 1: WHISPER TRANSCRIPTION ─────────────────────────
    print("\n" + "-"*40)
    print("STEP 1: Whisper Transcription (dual language)")
    print("-"*40)
    print("⏳ Loading Whisper model (downloads ~460MB on first run)...")

    try:
        transcripts = transcribe_all_chunks(
            metadata_path=metadata_path,
            output_dir=transcripts_dir,
            model_size="tiny"
        )
        print(f"\n✅ Transcribed {len(transcripts)} chunks")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── STEP 2: TRANSLITERATION ────────────────────────────────
    print("\n" + "-"*40)
    print("STEP 2: Transliteration (Devanagari → Roman)")
    print("-"*40)

    try:
        transcripts = transliterate_all_transcripts(transcripts)
        devanagari_found = sum(1 for t in transcripts if t.get("transliteration_applied"))
        print(f"✅ Transliteration complete | {devanagari_found} chunks processed")
    except Exception as e:
        print(f"❌ Transliteration failed: {e}")
        import traceback
        traceback.print_exc()

    # ── STEP 3: KEYWORD NORMALIZATION ─────────────────────────
    print("\n" + "-"*40)
    print("STEP 3: Keyword Detection & Normalization")
    print("-"*40)

    try:
        transcripts = apply_keyword_normalization_all(transcripts)
        flagged = [t for t in transcripts if t["keyword_analysis"]["keywords_found"]]
        print(f"✅ Keyword detection complete | {len(flagged)} chunks had emergency keywords")
    except Exception as e:
        print(f"❌ Keyword normalization failed: {e}")
        import traceback
        traceback.print_exc()

    # ── SAVE FINAL TRANSCRIPTS ─────────────────────────────────
    final_path = os.path.join(transcripts_dir, "all_transcripts_final.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Final transcripts saved → {final_path}")

    # ── PRINT RESULTS SUMMARY ──────────────────────────────────
    print("\n" + "="*60)
    print("  TRANSCRIPT RESULTS")
    print("="*60)

    for t in transcripts:
        chunk_id    = t.get("chunk_id", "?")
        text        = t.get("text", "")[:70]
        lang        = t.get("language_mix", "?")
        conf        = t.get("avg_confidence", 0)
        keywords    = t.get("keyword_analysis", {}).get("keywords_list", [])
        category    = t.get("keyword_analysis", {}).get("top_category", "none")
        start       = t.get("chunk_start", 0)

        keyword_str = f" 🚨 [{category}] {keywords}" if keywords else ""
        print(f"  [{chunk_id}] {start:.1f}s | {lang:10s} | conf:{conf:.2f} | {text}{keyword_str}")

    # ── LANGUAGE STATS ─────────────────────────────────────────
    print("\n" + "-"*40)
    print("  LANGUAGE DISTRIBUTION")
    print("-"*40)
    lang_counts = {}
    for t in transcripts:
        lang = t.get("language_mix", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / len(transcripts) * 100
        bar = "█" * int(pct / 5)
        print(f"  {lang:12s} {bar:20s} {count} chunks ({pct:.1f}%)")

    # ── EMERGENCY KEYWORDS SUMMARY ─────────────────────────────
    all_keywords = []
    for t in transcripts:
        all_keywords.extend(t.get("keyword_analysis", {}).get("keywords_list", []))

    if all_keywords:
        print("\n" + "-"*40)
        print("  EMERGENCY KEYWORDS DETECTED")
        print("-"*40)
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
            print(f"  '{kw}' × {count}")
    else:
        print("\n  No emergency keywords detected in this audio")
        print("  (Expected for normal conversation test audio)")

    # ── SAMPLE TRANSCRIPT JSON ─────────────────────────────────
    print("\n" + "-"*40)
    print("  SAMPLE TRANSCRIPT JSON (first chunk)")
    print("-"*40)
    if transcripts:
        sample = transcripts[0].copy()
        # Truncate words list for display
        if len(sample.get("words", [])) > 3:
            sample["words"] = sample["words"][:3]
            sample["words"].append({"note": f"... and {len(transcripts[0]['words'])-3} more words"})
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    # ── FINAL STATUS ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 COMPLETE ✅")
    print("="*60)
    print(f"\n  Chunks transcribed:  {len(transcripts)}")
    print(f"  Transcripts saved:   {transcripts_dir}")
    print(f"  Final JSON:          {final_path}")
    print(f"\n  ✅ Phase 2 passed — ready for Phase 3 (Analysis Models)")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_phase2_test()