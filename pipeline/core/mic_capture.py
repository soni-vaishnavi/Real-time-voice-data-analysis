"""
pipeline/core/mic_capture.py
=============================
STAGE 1 — Microphone Audio Capture

Continuously records audio from the default microphone in 5-second
fixed-window chunks and saves each as a WAV file.

Design decisions:
  - Fixed 5-second windows (not VAD-gated): captures everything including
    the first second of a sudden emergency shout — VAD-gated capture would
    miss the onset before the detector triggers.
  - Overlap is optional for file mode; for live mic, each window is independent.
  - No ML at all in this stage — pure audio I/O only.
  - Thread-safe: capture runs in its own thread, results go to a callback
    or optionally to an output queue (used by Stage 2).

Requirements:
    pip install sounddevice

Usage (standalone test):
    python pipeline/core/mic_capture.py

Usage (integrated with Stage 2 queue):
    from pipeline.core.mic_capture import MicCapture
    capture = MicCapture(on_chunk=queue.put_nowait)
    capture.start()
    ...
    capture.stop()
"""

import os
import wave
import uuid
import queue
import threading
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from pipeline.core.config import (
    SAMPLE_RATE,
    CHUNK_WINDOW_SEC,
    CHUNKS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MicCapture(threading.Thread):
    """
    Background thread that captures microphone audio in 5-second windows.

    Each completed chunk is:
      1. Saved to output/chunks/ as chunk_YYYYMMDD_HHMMSS_NNNN.wav
      2. Passed to on_chunk callback (if provided)

    Args:
        on_chunk:      Optional callback(item: dict) called with each saved chunk.
                       item = { "chunk_id", "audio_path", "chunk_start", "chunk_index" }
                       If None, chunks are only saved to disk (useful for Stage 1 testing).
        output_dir:    Directory to save WAV files. Defaults to config CHUNKS_DIR.
        device:        sounddevice device index or name. None = system default.
        chunk_sec:     Chunk duration in seconds. Defaults to config CHUNK_WINDOW_SEC.
        sample_rate:   Sample rate in Hz. Defaults to config SAMPLE_RATE.
    """

    def __init__(
        self,
        on_chunk:    Optional[Callable] = None,
        output_dir:  Optional[str]      = None,
        device:      Optional[int]      = None,
        chunk_sec:   int                = CHUNK_WINDOW_SEC,
        sample_rate: int                = SAMPLE_RATE,
    ):
        super().__init__(daemon=True, name="MicCapture")
        self.on_chunk    = on_chunk
        self.output_dir  = Path(output_dir) if output_dir else CHUNKS_DIR
        self.device      = device
        self.chunk_sec   = chunk_sec
        self.sample_rate = sample_rate

        self._stop_event   = threading.Event()
        self._chunk_index  = 0
        self._session_start = datetime.now()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MicCapture init | {sample_rate} Hz | {chunk_sec}s windows | "
                    f"output={self.output_dir}")

    def run(self) -> None:
        """Main capture loop. Blocks until stop() is called."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            return

        samples_per_chunk = self.sample_rate * self.chunk_sec
        logger.info(f"Mic capture started | device={self.device or 'default'} | "
                    f"{samples_per_chunk} samples per chunk")

        session_start_sec = 0.0

        while not self._stop_event.is_set():
            try:
                # Record exactly chunk_sec seconds — blocking call
                audio = sd.rec(
                    frames      = samples_per_chunk,
                    samplerate  = self.sample_rate,
                    channels    = 1,
                    dtype       = "int16",
                    device      = self.device,
                    blocking    = True,
                )
                # audio shape: (samples, 1) — flatten to (samples,)
                audio_flat = audio.flatten()

                chunk_path = self._save_wav(audio_flat)
                item = {
                    "chunk_id":    chunk_path.stem,
                    "audio_path":  str(chunk_path),
                    "chunk_start": session_start_sec,
                    "chunk_index": self._chunk_index,
                    "session_id":  self._session_start.strftime("%Y%m%d_%H%M%S"),
                }

                logger.info(f"Chunk saved | {chunk_path.name} | "
                            f"start={session_start_sec:.1f}s | index={self._chunk_index}")

                # Fire callback if provided (e.g., push to Stage 2 queue)
                if self.on_chunk is not None:
                    try:
                        self.on_chunk(item)
                    except Exception as e:
                        logger.warning(f"on_chunk callback error: {e}")

                session_start_sec += self.chunk_sec
                self._chunk_index += 1

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Mic capture error: {e}")
                    # Brief pause before retrying to avoid tight error loop
                    self._stop_event.wait(timeout=1.0)

        logger.info(f"Mic capture stopped after {self._chunk_index} chunks")

    def stop(self) -> None:
        """Signal the capture loop to stop after the current chunk finishes."""
        logger.info("Stopping mic capture...")
        self._stop_event.set()

    def is_running(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    def _save_wav(self, audio: np.ndarray) -> Path:
        """Save int16 numpy array as 16-bit mono WAV."""
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_name = f"chunk_{timestamp}_{self._chunk_index:04d}.wav"
        path       = self.output_dir / chunk_name

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)        # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        return path


# ── STANDALONE TEST ────────────────────────────────────────────────────────────

def test_mic_standalone(n_chunks: int = 5) -> None:
    """
    Stage 1 test: record N chunks from mic, save to disk, print results.
    No ML. Verifies sounddevice works and WAV files are created correctly.
    """
    print("\n" + "=" * 55)
    print("  STAGE 1 — MIC CAPTURE TEST")
    print("=" * 55)
    print(f"  Recording {n_chunks} chunks of {CHUNK_WINDOW_SEC}s each...")
    print(f"  Output:   {CHUNKS_DIR}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"\n  Speak into your microphone. Recording starts NOW.\n")

    saved = []

    def on_chunk(item: dict):
        saved.append(item)
        size_kb = round(os.path.getsize(item["audio_path"]) / 1024, 1)
        print(f"  [{len(saved)}/{n_chunks}] {item['chunk_id']}.wav | "
              f"start={item['chunk_start']:.1f}s | {size_kb} KB")
        if len(saved) >= n_chunks:
            capture.stop()

    capture = MicCapture(on_chunk=on_chunk)
    capture.start()
    capture.join(timeout=n_chunks * CHUNK_WINDOW_SEC + 5)

    print(f"\n  Done. {len(saved)} chunks saved to {CHUNKS_DIR}/")
    print()

    for item in saved:
        path = item["audio_path"]
        if os.path.exists(path):
            size_kb = round(os.path.getsize(path) / 1024, 1)
            print(f"  ✅ {os.path.basename(path)} — {size_kb} KB")
        else:
            print(f"  ❌ {os.path.basename(path)} — NOT FOUND")

    print()
    if len(saved) == n_chunks:
        print(f"  ✅ Stage 1 PASSED — {n_chunks} chunks recorded successfully")
        print("  Output files are ready. Proceed to Stage 2.")
    else:
        print(f"  ❌ Stage 1 FAILED — expected {n_chunks}, got {len(saved)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VoiceGuard Stage 1 — Mic Capture Test")
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks to record")
    parser.add_argument("--device", type=int, default=None, help="Mic device index (default: system default)")
    args = parser.parse_args()

    if args.device is not None:
        # Override default device for the test
        import sounddevice as sd
        print(f"\nAvailable input devices:")
        print(sd.query_devices())

    test_mic_standalone(n_chunks=args.chunks)