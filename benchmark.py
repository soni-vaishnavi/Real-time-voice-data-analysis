"""
benchmark.py
=============
Stage 9 — Performance Benchmarking

Measures actual processing times for each pipeline component.
Helps identify bottlenecks and verify optimization targets.

Usage:
    python benchmark.py                    # benchmark transcription only
    python benchmark.py --full             # benchmark all phases
    python benchmark.py --report           # print full report with suggestions

Expected targets (CPU, 8GB RAM):
    Phase 1 (preprocessing):  < 15s  for 100s audio
    Phase 2 (Whisper small):  8-15s  per chunk  (translate mode)
    Phase 3 (BART + emotion): 8-12s  per chunk
    Phase 4 (scoring):        < 0.1s per chunk  (pure Python)
    Full pipeline (35 chunks): 10-20 minutes total
"""

import os
import sys
import time
import json
import statistics
import threading
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Timer:
    def __init__(self, label: str):
        self.label   = label
        self.elapsed = 0.0
        self._start  = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start


def _print_bar(label: str, elapsed: float, target: float):
    pct = min(elapsed / target, 2.0)
    bar = "█" * int(pct * 20)
    color = "✅" if elapsed <= target else "⚠️ "
    print(f"  {color} {label:<30} {elapsed:6.1f}s  (target <{target}s)  {bar}")


# ── BENCHMARK 1: Whisper Translation ─────────────────────────────────────────

def benchmark_whisper(chunks_dir: str = "output/chunks",
                      meta_path:  str = "output/chunks/metadata.json",
                      model_size: str = "small",
                      n_chunks:   int = 5) -> List[float]:
    """Benchmark Whisper translation on real chunks."""
    print(f"\n[Whisper Benchmark] model={model_size}, n={n_chunks}")

    if not os.path.exists(meta_path):
        print(f"  [SKIP] {meta_path} not found — run python main.py first")
        return []

    with open(meta_path) as f:
        all_chunks = json.load(f)

    test_chunks = [c for c in all_chunks if os.path.exists(c.get("file_path",""))][:n_chunks]
    if not test_chunks:
        print("  [SKIP] No chunk WAV files found")
        return []

    from pipeline.phase2_stt.whisper_transcriber import transcribe_chunk, get_model
    get_model(model_size)  # pre-load

    times   = []
    results = []
    for chunk in test_chunks:
        t0 = time.time()
        result = transcribe_chunk(
            chunk["file_path"], chunk["chunk_id"], chunk["start"], model_size
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        text    = result.get("text","") if result else "[failed]"
        lang    = result.get("language_mix","?") if result else "?"
        print(f"  {chunk['chunk_id']} | {lang} | {elapsed:.1f}s | '{text[:50]}'")
        results.append({"chunk_id": chunk["chunk_id"], "elapsed": elapsed, "text": text, "lang": lang})

    if times:
        print(f"\n  Min: {min(times):.1f}s | Max: {max(times):.1f}s | "
              f"Avg: {statistics.mean(times):.1f}s | "
              f"Median: {statistics.median(times):.1f}s")
        slow = [r for r in results if r["elapsed"] > 20]
        if slow:
            print(f"  ⚠️  {len(slow)} chunks took >20s (translation loops?):")
            for r in slow:
                print(f"     {r['chunk_id']}: {r['elapsed']:.1f}s | '{r['text'][:40]}'")

    return times


# ── BENCHMARK 2: BART Emergency ───────────────────────────────────────────────

def benchmark_bart(test_texts: Optional[List[str]] = None) -> List[float]:
    """Benchmark BART emergency detection."""
    print("\n[BART Benchmark]")

    test_texts = test_texts or [
        "Save me call the ambulance",
        "There is a fire help",
        "Someone shot a gun robbery",
        "We are going to the market today",
        "The weather is nice and I feel good",
    ]

    from pipeline.phase3_analysis.emergency_detector import get_emergency_model, detect_emergency
    print("  Loading BART (may take 30-60s on first run)...")
    t0 = time.time()
    get_emergency_model(wait=True)
    load_time = time.time() - t0
    print(f"  BART loaded in {load_time:.1f}s")

    times = []
    for text in test_texts:
        t0      = time.time()
        result  = detect_emergency(text, wait_for_model=False)
        elapsed = time.time() - t0
        times.append(elapsed)
        cat   = result.get("top_category", "?")
        score = result.get("top_score", 0.0)
        print(f"  {elapsed:.2f}s | {cat} ({score:.2f}) | '{text[:45]}'")

    if times:
        print(f"\n  Avg: {statistics.mean(times):.2f}s | Max: {max(times):.2f}s")

    return times


# ── BENCHMARK 3: Emotion Detection ────────────────────────────────────────────

def benchmark_emotion(n_runs: int = 5) -> List[float]:
    """Benchmark distilRoBERTa emotion detection."""
    print(f"\n[Emotion Benchmark] n={n_runs}")

    test_texts = [
        "Save me the house is on fire",
        "I feel happy and joyful today",
        "Someone is attacking me please help",
        "I don't know what to do anymore",
        "Normal conversation about weather",
    ] * (n_runs // 5 + 1)
    test_texts = test_texts[:n_runs]

    from pipeline.phase3_analysis.emotion_detector import detect_emotion, get_emotion_model
    get_emotion_model()

    times = []
    for text in test_texts:
        t0      = time.time()
        result  = detect_emotion(text)
        elapsed = time.time() - t0
        times.append(elapsed)
        emo = result.get("dominant_emotion", "?")
        print(f"  {elapsed:.3f}s | {emo} | '{text[:45]}'")

    print(f"\n  Avg: {statistics.mean(times):.3f}s | Max: {max(times):.3f}s")
    return times


# ── BENCHMARK 4: Full Per-Chunk Pipeline ──────────────────────────────────────

def benchmark_full_chunk(meta_path: str = "output/chunks/metadata.json",
                         n_chunks: int = 3) -> List[float]:
    """Benchmark full pipeline per chunk (no DB, no alerts)."""
    print(f"\n[Full Pipeline Benchmark] n={n_chunks}")

    if not os.path.exists(meta_path):
        print(f"  [SKIP] {meta_path} not found")
        return []

    with open(meta_path) as f:
        all_chunks = json.load(f)

    test_chunks = [c for c in all_chunks if os.path.exists(c.get("file_path",""))][:n_chunks]
    if not test_chunks:
        print("  [SKIP] No WAV files found")
        return []

    from pipeline.phase2_stt.whisper_transcriber import transcribe_chunk, get_model
    from pipeline.phase2_stt.keyword_normalizer import apply_keyword_normalization
    from pipeline.phase3_analysis.emotion_detector import detect_emotion, get_emotion_model
    from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
    from pipeline.phase3_analysis.emergency_detector import detect_emergency, get_emergency_model
    from pipeline.phase4_decision.scorer import compute_score
    from pipeline.phase4_decision.zone_classifier import classify_zone

    get_model("small")
    get_emotion_model()
    get_emergency_model(wait=True)

    total_times = []
    for chunk in test_chunks:
        print(f"\n  {chunk['chunk_id']}:")
        timings = {}

        with Timer("whisper") as t:
            tr = transcribe_chunk(chunk["file_path"], chunk["chunk_id"], chunk["start"], "small")
            tr = apply_keyword_normalization(tr)
        timings["whisper+kw"] = t.elapsed

        text = tr.get("text","") if tr else ""

        with Timer("emotion") as t:
            emo  = detect_emotion(text)
            sarc = detect_sarcasm(text)
        timings["emotion+sarcasm"] = t.elapsed

        with Timer("bart") as t:
            emrg = detect_emergency(text, wait_for_model=False)
        timings["bart"] = t.elapsed

        with Timer("score") as t:
            chunk_data = {
                "text": text,
                "emotion_analysis": {"emotion": emo, "sarcasm": sarc,
                                     "sarcasm_resolution": {"score_penalty":0.0}},
                "emergency_analysis": emrg,
                "keyword_analysis": tr.get("keyword_analysis",{}) if tr else {},
            }
            s = compute_score(chunk_data)
            classify_zone(s)
        timings["scoring"] = t.elapsed

        total = sum(timings.values())
        total_times.append(total)

        for name, elapsed in timings.items():
            bar = "█" * int(elapsed * 2)
            print(f"    {name:<20} {elapsed:5.2f}s  {bar}")
        print(f"    {'TOTAL':<20} {total:5.2f}s  | zone={s.get('zone','?')} | '{text[:40]}'")

    if total_times:
        print(f"\n  Total avg: {statistics.mean(total_times):.1f}s/chunk | "
              f"Max: {max(total_times):.1f}s/chunk")
        rate = 3600 / statistics.mean(total_times)
        print(f"  Processing rate: ~{rate:.0f} chunks/hour  "
              f"(= ~{rate*5/60:.0f} minutes of audio/hour)")

    return total_times


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(whisper_times: List[float], bart_times: List[float],
                 emotion_times: List[float], full_times: List[float]):
    """Print performance report with optimization suggestions."""
    print("\n" + "=" * 60)
    print("  STAGE 9 — PERFORMANCE REPORT")
    print("=" * 60)

    if whisper_times:
        avg_w = statistics.mean(whisper_times)
        print(f"\n  Whisper (translate mode):")
        _print_bar("per chunk", avg_w, 15.0)
        if avg_w > 20:
            print("  💡 Tip: Some chunks are very slow (loops).")
            print("     The timeout=45s protection prevents infinite hangs.")
            print("     Consider --model base for faster inference.")

    if bart_times:
        avg_b = statistics.mean(bart_times)
        print(f"\n  BART Emergency Detection:")
        _print_bar("per chunk", avg_b, 5.0)
        if avg_b > 8:
            print("  💡 Tip: BART is slow. Consider:")
            print("     - facebook/bart-base-mnli (~400MB vs 1.6GB, ~3x faster)")
            print("     - Reduce candidate labels from 7 to 4 most common")

    if emotion_times:
        avg_e = statistics.mean(emotion_times)
        print(f"\n  Emotion Detection:")
        _print_bar("per chunk", avg_e, 0.5)

    if full_times:
        avg_f = statistics.mean(full_times)
        print(f"\n  Full Pipeline (Whisper+BART+Emotion):")
        _print_bar("per chunk", avg_f, 20.0)
        total_35 = avg_f * 35
        print(f"  Estimated for 35 chunks: {total_35/60:.1f} minutes")
        if total_35 > 600:
            print("  💡 Speed-up options (in priority order):")
            print("     1. --model base  (2x faster Whisper, small accuracy drop)")
            print("     2. --model tiny  (5x faster, more hallucinations)")
            print("     3. Disable wav2vec2: enable_heavy=False in api/main.py")
            print("     4. Reduce beam_size to 2 in whisper_transcriber.py")

    print("\n  Optimization Notes:")
    print("  - task='translate' is slower than 'transcribe' for same model")
    print("    but much more accurate for Hindi → English output")
    print("  - Temperature fallback [0,0.2,0.4,0.6] adds ~2s when loops detected")
    print("  - Timeout=45s prevents worst-case 80s hangs (was occurring with 'small')")
    print("  - For live mode: 5s chunks arrive every 5s, worker needs <5s/chunk")
    print("    Use --model tiny for live mode if latency matters more than accuracy")
    print("=" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAGE 9 — PERFORMANCE BENCHMARKING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    full   = "--full"   in sys.argv or "-f" in sys.argv
    report = "--report" in sys.argv or "-r" in sys.argv

    w_times = e_times = b_times = f_times = []

    # Always run Whisper benchmark (most important)
    w_times = benchmark_whisper(n_chunks=5)

    if full:
        e_times = benchmark_emotion(n_runs=5)
        b_times = benchmark_bart()
        f_times = benchmark_full_chunk(n_chunks=3)
    else:
        print("\n  Add --full to benchmark BART, emotion, and full pipeline")
        print("  Add --report for optimization suggestions")

    if report or full:
        print_report(w_times, b_times, e_times, f_times)


if __name__ == "__main__":
    main()