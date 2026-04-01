"""
Phase 1 - Step 3: VAD Chunker
Uses WebRTC VAD to detect speech and split audio into smart chunks
with sliding window overlap so no speech gets cut mid-sentence
"""

import os
import json
import wave
import struct
import webrtcvad
import numpy as np
from pydub import AudioSegment
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def read_wav_frames(file_path: str):
    """Read raw frames from WAV file for WebRTC VAD"""
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    return frames, sample_rate, num_channels, sample_width


def frame_generator(frame_duration_ms: int, audio_bytes: bytes, sample_rate: int):
    """
    Generate fixed-size audio frames for VAD processing.
    
    WebRTC VAD processes audio in small fixed frames:
    - 10ms, 20ms, or 30ms frames only (VAD requirement)
    - We use 30ms frames — best balance of accuracy and speed
    
    Each frame = sample_rate * (frame_duration_ms/1000) * 2 bytes
    At 16kHz with 30ms: 16000 * 0.030 * 2 = 960 bytes per frame
    """
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # *2 for 16-bit
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0

    while offset + frame_size <= len(audio_bytes):
        yield {
            "bytes": audio_bytes[offset:offset + frame_size],
            "timestamp": timestamp,
            "duration": duration
        }
        offset += frame_size
        timestamp += duration


def vad_detect_speech(
    frames: list,
    sample_rate: int,
    aggressiveness: int = 2,
    padding_duration_ms: int = 300
) -> List[Dict]:
    """
    Detect speech segments using WebRTC VAD.
    
    How it works:
    - VAD looks at each 30ms frame
    - Marks each frame as SPEECH or SILENCE
    - We use a padding buffer so speech doesn't get cut too early
      (300ms padding = if silence < 300ms, keep it as speech)
    
    Args:
        aggressiveness: 0-3
            0 = least aggressive (keeps more audio, more false positives)
            1 = gentle
            2 = balanced ← recommended for surveillance
            3 = most aggressive (strips more, may cut quiet speech)
    
    Returns:
        List of speech segments with start/end times
    """
    vad = webrtcvad.Vad(aggressiveness)

    # Classify each frame as speech or not
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame["bytes"], sample_rate)
        voiced_frames.append({
            "bytes": frame["bytes"],
            "timestamp": frame["timestamp"],
            "duration": frame["duration"],
            "is_speech": is_speech
        })

    # Group consecutive speech frames into segments
    # Using padding to avoid cutting speech too early
    num_padding_frames = int(padding_duration_ms / 30)  # 30ms frames
    ring_buffer = []
    triggered = False
    speech_segments = []
    current_segment_frames = []
    segment_start = 0.0

    for frame in voiced_frames:
        if not triggered:
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)

            # Count speech frames in buffer
            num_voiced = sum(1 for f in ring_buffer if f["is_speech"])

            # If >90% of buffer is speech → start segment
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                segment_start = ring_buffer[0]["timestamp"]
                current_segment_frames.extend(ring_buffer)
                ring_buffer = []
        else:
            current_segment_frames.append(frame)
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)

            # Count unvoiced frames in buffer
            num_unvoiced = sum(1 for f in ring_buffer if not f["is_speech"])

            # If >90% of buffer is silence → end segment
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                segment_end = current_segment_frames[-1]["timestamp"] + current_segment_frames[-1]["duration"]

                # Only keep segments longer than 0.5 seconds
                if segment_end - segment_start > 0.5:
                    speech_segments.append({
                        "start": round(segment_start, 3),
                        "end": round(segment_end, 3),
                        "duration": round(segment_end - segment_start, 3),
                        "frames": current_segment_frames
                    })

                current_segment_frames = []
                ring_buffer = []

    # Handle last segment if still in triggered state
    if triggered and current_segment_frames:
        segment_end = current_segment_frames[-1]["timestamp"] + current_segment_frames[-1]["duration"]
        if segment_end - segment_start > 0.5:
            speech_segments.append({
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "duration": round(segment_end - segment_start, 3),
                "frames": current_segment_frames
            })

    logger.info(f"VAD detected {len(speech_segments)} speech segments")
    return speech_segments


def apply_sliding_window(
    speech_segments: List[Dict],
    window_sec: float = 5.0,
    overlap_sec: float = 2.0
) -> List[Dict]:
    """
    Apply sliding window on top of VAD segments.
    
    Why sliding window on top of VAD?
    - VAD gives us rough speech boundaries
    - But a single speech segment might be 30+ seconds long
    - Whisper works best on 5-10 second chunks
    - Sliding window breaks long segments into overlapping pieces
    - Overlap ensures words at boundaries aren't missed
    
    Example:
    VAD segment: 0s → 25s (long continuous speech)
    
    After sliding window (5s window, 2s overlap):
    Chunk 1: 0s  → 5s
    Chunk 2: 3s  → 8s   (overlaps by 2s)
    Chunk 3: 6s  → 11s
    Chunk 4: 9s  → 14s
    ... and so on
    """
    chunks = []
    chunk_id = 0
    step_sec = window_sec - overlap_sec  # how much to advance each time

    for segment in speech_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_duration = segment["duration"]

        if seg_duration <= window_sec:
            # Short segment — keep as single chunk
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:03d}",
                "start": seg_start,
                "end": seg_end,
                "duration": round(seg_duration, 3),
                "frames": segment["frames"]
            })
            chunk_id += 1
        else:
            # Long segment — apply sliding window
            window_start = seg_start
            while window_start < seg_end:
                window_end = min(window_start + window_sec, seg_end)
                duration = window_end - window_start

                # Only keep if at least 1 second long
                if duration >= 1.0:
                    # Find frames that belong to this window
                    window_frames = [
                        f for f in segment["frames"]
                        if window_start <= f["timestamp"] < window_end
                    ]

                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id:03d}",
                        "start": round(window_start, 3),
                        "end": round(window_end, 3),
                        "duration": round(duration, 3),
                        "frames": window_frames
                    })
                    chunk_id += 1

                window_start += step_sec

                # Stop if remaining audio is less than 1 second
                if seg_end - window_start < 1.0:
                    break

    logger.info(f"Sliding window produced {len(chunks)} chunks from {len(speech_segments)} segments")
    return chunks


def save_chunks(
    chunks: List[Dict],
    output_dir: str,
    sample_rate: int = 16000
) -> List[Dict]:
    """
    Save each chunk as individual WAV file.
    Also saves metadata.json with all chunk info.
    
    Returns:
        List of chunk metadata (without raw frames — just paths + timestamps)
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        # Reconstruct audio from frames
        if not chunk["frames"]:
            logger.warning(f"Skipping {chunk_id} — no frames")
            continue

        raw_bytes = b"".join(f["bytes"] for f in chunk["frames"])

        # Save as WAV
        chunk_path = os.path.join(output_dir, f"{chunk_id}.wav")
        with wave.open(chunk_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)        # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(raw_bytes)

        chunk_meta = {
            "chunk_id": chunk_id,
            "file_path": chunk_path,
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": chunk["duration"]
        }
        metadata.append(chunk_meta)
        logger.info(f"Saved {chunk_id} | {chunk['start']:.2f}s → {chunk['end']:.2f}s | {chunk['duration']:.2f}s")

    # Save metadata.json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved → {metadata_path}")
    return metadata


def process_vad_chunking(
    input_path: str,
    output_dir: str,
    aggressiveness: int = 2,
    window_sec: float = 5.0,
    overlap_sec: float = 2.0
) -> List[Dict]:
    """
    Full VAD chunking pipeline:
    Load → Frame → VAD detect → Sliding window → Save chunks
    
    Args:
        input_path:     noise-reduced WAV file
        output_dir:     directory to save chunks
        aggressiveness: VAD aggressiveness 0-3
        window_sec:     sliding window size in seconds
        overlap_sec:    overlap between windows in seconds
    
    Returns:
        List of chunk metadata dicts
    """
    logger.info("=" * 50)
    logger.info("PHASE 1 - STEP 3: VAD CHUNKING STARTED")
    logger.info("=" * 50)

    # Read WAV frames
    audio_bytes, sample_rate, channels, sample_width = read_wav_frames(input_path)
    logger.info(f"Read WAV | Sample rate: {sample_rate}Hz | Channels: {channels}")

    # Validate format
    if sample_rate not in [8000, 16000, 32000]:
        raise ValueError(f"VAD requires 8000/16000/32000 Hz, got {sample_rate}Hz. Run preprocessor first.")
    if channels != 1:
        raise ValueError(f"VAD requires mono audio, got {channels} channels. Run preprocessor first.")

    # Generate 30ms frames
    frames = list(frame_generator(30, audio_bytes, sample_rate))
    logger.info(f"Generated {len(frames)} frames (30ms each)")

    # VAD speech detection
    speech_segments = vad_detect_speech(frames, sample_rate, aggressiveness)

    # Apply sliding window
    chunks = apply_sliding_window(speech_segments, window_sec, overlap_sec)

    # Save chunks
    metadata = save_chunks(chunks, output_dir, sample_rate)

    logger.info(f"VAD CHUNKING COMPLETE ✅ | {len(metadata)} chunks saved to {output_dir}")
    return metadata