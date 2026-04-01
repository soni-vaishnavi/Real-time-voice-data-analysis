"""
Phase 1 Test Script
Run this to test the full Phase 1 pipeline on your Test_Normal.wav file

Usage:
    python test_phase1.py

Expected output:
    - output/preprocessed/Test_Normal_preprocessed.wav
    - output/noise_reduced/Test_Normal_noise_reduced.wav
    - output/chunks/chunk_000.wav, chunk_001.wav ...
    - output/chunks/metadata.json
"""

import os
import json
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.phase1_audio.preprocessor import preprocess
from pipeline.phase1_audio.noise_reducer import process_noise_reduction
from pipeline.phase1_audio.vad_chunker import process_vad_chunking


def run_phase1_test():
    print("\n" + "="*60)
    print("  PHASE 1 TEST — Audio Pipeline")
    print("="*60)

    # ── PATHS ──────────────────────────────────────────────────
    input_audio     = "input/audio/Test_Normal.wav"
    preprocessed    = "output/preprocessed/Test_Normal_preprocessed.wav"
    noise_reduced   = "output/noise_reduced/Test_Normal_noise_reduced.wav"
    chunks_dir      = "output/chunks/"

    # ── CHECK INPUT EXISTS ─────────────────────────────────────
    if not os.path.exists(input_audio):
        print(f"\n❌ ERROR: Input file not found: {input_audio}")
        print("   Make sure Test_Normal.wav is in input/audio/ folder")
        return

    print(f"\n✅ Input file found: {input_audio}")
    print(f"   Size: {os.path.getsize(input_audio) / (1024*1024):.2f} MB")

    # ── STEP 1: PREPROCESS ─────────────────────────────────────
    print("\n" + "-"*40)
    print("STEP 1: Preprocessing (format + normalize)")
    print("-"*40)
    try:
        os.makedirs("output/preprocessed", exist_ok=True)
        preprocess(input_audio, preprocessed)
        size_mb = os.path.getsize(preprocessed) / (1024*1024)
        print(f"✅ Preprocessed saved | Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    # ── STEP 2: NOISE REDUCTION ────────────────────────────────
    print("\n" + "-"*40)
    print("STEP 2: Noise Reduction")
    print("-"*40)
    try:
        os.makedirs("output/noise_reduced", exist_ok=True)
        process_noise_reduction(preprocessed, noise_reduced, strength=0.75)
        size_mb = os.path.getsize(noise_reduced) / (1024*1024)
        print(f"✅ Noise-reduced saved | Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ Noise reduction failed: {e}")
        return

    # ── STEP 3: VAD CHUNKING ───────────────────────────────────
    print("\n" + "-"*40)
    print("STEP 3: VAD Chunking (sliding window)")
    print("-"*40)
    try:
        metadata = process_vad_chunking(
            input_path=noise_reduced,
            output_dir=chunks_dir,
            aggressiveness=2,
            window_sec=5.0,
            overlap_sec=2.0
        )

        print(f"\n✅ VAD Chunking complete!")
        print(f"   Total chunks created: {len(metadata)}")
        print(f"\n   Chunk breakdown:")
        for chunk in metadata:
            print(f"   {chunk['chunk_id']} | {chunk['start']:.2f}s → {chunk['end']:.2f}s | duration: {chunk['duration']:.2f}s")

    except Exception as e:
        print(f"❌ VAD Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── FINAL SUMMARY ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 COMPLETE ✅")
    print("="*60)
    print(f"\n  Input:          {input_audio}")
    print(f"  Preprocessed:   {preprocessed}")
    print(f"  Noise-reduced:  {noise_reduced}")
    print(f"  Chunks folder:  {chunks_dir}")
    print(f"  Total chunks:   {len(metadata)}")
    print(f"\n  metadata.json preview:")

    # Show metadata.json content
    metadata_path = os.path.join(chunks_dir, "metadata.json")
    with open(metadata_path) as f:
        data = json.load(f)
    print(json.dumps(data[:3], indent=4))  # show first 3 chunks
    if len(data) > 3:
        print(f"  ... and {len(data)-3} more chunks")

    print("\n  ✅ Phase 1 passed — ready for Phase 2 (Whisper STT)")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_phase1_test()