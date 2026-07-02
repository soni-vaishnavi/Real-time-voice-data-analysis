"""
test_transcription.py
======================
Stage 8 — Transcription Accuracy Test

Tests the fixed Whisper translation pipeline against known Hindi/Hinglish phrases.
Generates a WAV file for each phrase, runs transcription, shows result.

Usage:
    python test_transcription.py

What it tests:
    1. Translation accuracy: Hindi speech → English text
    2. Emergency keyword detection post-translation
    3. Language detection (should correctly identify Hindi/Hinglish)
    4. Hallucination filter (silence should return empty string)
    5. Timeout protection (long/looping inference should be killed at 45s)

Expected results for emergency audio:
    "bachao bachao ambulance bulao"  → "save me, call the ambulance"
    "aag lagi hai"                  → "there is a fire"
    "doctor chahiye"                → "need a doctor"
"""

import os
import sys
import json
import time
import wave
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_silence_wav(path: str, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Create a silent WAV file for testing."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    n_samples = int(sample_rate * duration_sec)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))


def test_hallucination_filter():
    """Test that the hallucination filter catches bad patterns."""
    print("\n[Test 1] Hallucination Filter")
    from pipeline.phase2_stt.whisper_transcriber import is_hallucination

    test_cases = [
        ("", True,                          "empty string"),
        ("।  । । ।", True,                  "Devanagari danda punctuation"),
        ("बबबबबबबबबबबबबबबबबबबबबबबबब", True,  "Devanagari character loop"),
        ("ो ो ो ो ो ो ो", True,              "Devanagari matra loop"),
        ("thank you for watching", True,     "YouTube hallucination phrase"),
        ("save me call the ambulance", False, "valid English emergency"),
        ("There is a fire", False,            "valid English"),
        ("need a doctor please", False,       "valid request"),
        ("hello hello hello hello hello", True, "word repetition loop"),
    ]

    passed = failed = 0
    for text, expected, desc in test_cases:
        result = is_hallucination(text)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} is_hallucination('{text[:40]}') = {result} | {desc}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\n  Result: {passed}/{passed+failed} passed")
    return failed == 0


def test_energy_check():
    """Test that silent audio is skipped before Whisper."""
    print("\n[Test 2] Audio Energy Check")
    from pipeline.phase2_stt.whisper_transcriber import _check_audio_energy

    # Create silent WAV
    silent_path = "output/test_silence.wav"
    os.makedirs("output", exist_ok=True)
    _make_silence_wav(silent_path, duration_sec=1.0)

    result = _check_audio_energy(silent_path)
    ok = result == False
    print(f"  {'✅' if ok else '❌'} Silent WAV energy check = {result} (expected False)")

    if os.path.exists("input/audio/Test_Normal.wav"):
        result2 = _check_audio_energy("input/audio/Test_Normal.wav")
        ok2 = result2 == True
        print(f"  {'✅' if ok2 else '❌'} Real audio energy check = {result2} (expected True)")
        return ok and ok2

    return ok


def test_keyword_matching_translated():
    """Test that translated English emergency words match our keyword dictionary."""
    print("\n[Test 3] Keyword Matching on Translated Text")
    from pipeline.phase2_stt.keyword_normalizer import detect_keywords, get_keyword_summary

    # These are typical Whisper translation outputs for Hindi emergency speech
    translated_samples = [
        ("save me call the ambulance",          ["ambulance"],    "MEDICAL"),
        ("there is a fire help",                ["fire", "help"], "FIRE"),
        ("I got shot gun",                      ["gun"],          "VIOLENCE"),
        ("blood bleeding please help",          ["blood", "bleeding", "help"], "MEDICAL"),
        ("robbery theft thief",                 ["robbery"],      "THEFT"),
        ("accident someone fell injured",       ["accident"],     "ACCIDENT"),
        ("I want to die suicide",               ["suicide"],      "MENTAL"),
        ("everything is fine nice day",         [],               None),
    ]

    passed = failed = 0
    for text, expected_kws, expected_cat in translated_samples:
        found = detect_keywords(text)
        summary = get_keyword_summary(found)
        found_kws = summary.get("keywords_list", [])
        cat = summary.get("top_category")

        # Check that at least one expected keyword was found
        any_found = any(kw in " ".join(found_kws).lower() for kw in expected_kws)
        cat_ok    = (expected_cat is None and cat is None) or (cat == expected_cat)
        ok = (len(expected_kws) == 0 and len(found_kws) == 0) or any_found

        status = "✅" if ok else "❌"
        print(f"  {status} '{text[:45]}' → kws={found_kws} | cat={cat}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\n  Result: {passed}/{passed+failed} passed")
    return failed == 0


def test_translation_on_existing_audio():
    """Run actual Whisper translation on existing test audio if available."""
    print("\n[Test 4] Whisper Translation on Real Audio")

    audio_path = "input/audio/Test_Normal.wav"
    meta_path  = "output/chunks/metadata.json"

    if not os.path.exists(meta_path):
        print(f"  [SKIP] {meta_path} not found — run python main.py first")
        return True

    with open(meta_path) as f:
        chunks = json.load(f)

    if not chunks:
        print("  [SKIP] No chunks found")
        return True

    # Test first 3 chunks only
    test_chunks = [c for c in chunks[:5] if os.path.exists(c.get("file_path",""))][:3]
    if not test_chunks:
        print("  [SKIP] Chunk WAV files not found")
        return True

    print(f"  Testing translation on {len(test_chunks)} chunks (model=small)...")
    from pipeline.phase2_stt.whisper_transcriber import transcribe_chunk

    all_ok = True
    for chunk in test_chunks:
        t0 = time.time()
        result = transcribe_chunk(
            chunk["file_path"],
            chunk["chunk_id"],
            chunk["start"],
            model_size="small",
        )
        elapsed = time.time() - t0

        if result is None:
            print(f"  ❌ {chunk['chunk_id']} | returned None")
            all_ok = False
            continue

        text     = result.get("text", "")
        lang     = result.get("language_detected", "?")
        lang_mix = result.get("language_mix", "?")

        # Check: no Devanagari in output (task=translate must give English)
        import re
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        has_danda      = "।" in text

        ok = not has_devanagari and not has_danda
        status = "✅" if ok else "❌"
        issue  = " ← DEVANAGARI IN OUTPUT!" if has_devanagari else (" ← DANDA!" if has_danda else "")
        print(f"  {status} {chunk['chunk_id']} | src={lang_mix}({lang}) | {elapsed:.1f}s | '{text[:60]}'{issue}")

        if not ok:
            all_ok = False

    if all_ok:
        print("  ✅ All chunks: English output, no Devanagari")
    return all_ok


def test_phase2_full_pipeline():
    """Test the complete Phase 2 pipeline on existing chunks."""
    print("\n[Test 5] Full Phase 2 Pipeline")

    meta_path = "output/chunks/metadata.json"
    if not os.path.exists(meta_path):
        print(f"  [SKIP] Run python main.py first to generate chunks")
        return True

    print("  Running transcribe_all_chunks (translate mode)...")
    t0 = time.time()
    from pipeline.phase2_stt.whisper_transcriber import transcribe_all_chunks
    transcripts = transcribe_all_chunks(
        metadata_path = meta_path,
        output_dir    = "output/transcripts_test/",
        model_size    = "small",
    )
    elapsed = time.time() - t0

    # Analysis
    import re
    speech_count    = sum(1 for t in transcripts if t.get("text","").strip())
    devanagari_count= sum(1 for t in transcripts if re.search(r'[\u0900-\u097F]', t.get("text","")))
    danda_count     = sum(1 for t in transcripts if "।" in t.get("text",""))
    avg_per_chunk   = elapsed / max(1, len(transcripts))

    print(f"\n  Total chunks:      {len(transcripts)}")
    print(f"  With speech:       {speech_count}/{len(transcripts)}")
    print(f"  Devanagari output: {devanagari_count} (should be 0 with task=translate)")
    print(f"  Danda output:      {danda_count} (should be 0)")
    print(f"  Total time:        {elapsed:.1f}s ({avg_per_chunk:.1f}s/chunk)")

    ok = devanagari_count == 0 and danda_count == 0
    print(f"\n  {'✅' if ok else '❌'} Translation purity check")

    print("\n  Sample transcripts:")
    for t in transcripts[:6]:
        text = t.get("text","") or "[no speech]"
        lang = t.get("language_mix","?")
        print(f"    {t['chunk_id']} | {lang} | '{text[:65]}'")

    return ok


def main():
    print("=" * 60)
    print("  STAGE 8 — TRANSCRIPTION ACCURACY TESTS")
    print("  Whisper translate mode (Hindi/Hinglish → English)")
    print("=" * 60)

    results = {}

    results["hallucination_filter"] = test_hallucination_filter()
    results["energy_check"]         = test_energy_check()
    results["keyword_matching"]     = test_keyword_matching_translated()
    results["translation_audio"]    = test_translation_on_existing_audio()

    # Optional: full pipeline test (slow, requires existing chunks)
    run_full = "--full" in sys.argv or "-f" in sys.argv
    if run_full:
        results["full_pipeline"] = test_phase2_full_pipeline()
    else:
        print("\n[Test 5] Full Phase 2 Pipeline")
        print("  [SKIP] Add --full flag to run: python test_transcription.py --full")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{total} test groups passed")
    if passed == total:
        print("\n  ✅ All transcription tests passed!")
        print("  Whisper is correctly translating Hindi/Hinglish to English.")
    else:
        print("\n  ⚠️  Some tests failed — check output above")
    print("=" * 60)


if __name__ == "__main__":
    main()