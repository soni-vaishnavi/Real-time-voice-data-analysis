"""
Phase 1 - Step 4: Speaker Diarization
Identifies WHO is speaking and WHEN in multi-speaker audio.
Uses pyannote-audio from HuggingFace.

Why diarization matters for surveillance:
- Multiple people speak simultaneously in emergencies
- System needs to track which speaker is in distress
- Speaker 2 saying "bachao" is different from Speaker 1 saying it calmly

Setup required (one time):
1. Accept pyannote terms at https://huggingface.co/pyannote/speaker-diarization-3.1
2. Get HuggingFace token from https://huggingface.co/settings/tokens
3. Set token in config.py: HF_TOKEN = "your_token_here"

Without token: diarization is skipped, speaker_id = "UNKNOWN"
"""

import os
import json
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── DIARIZATION PIPELINE ──────────────────────────────────────────────────────

_diarization_pipeline = None


def get_diarization_pipeline(hf_token: Optional[str] = None):
    """
    Load pyannote diarization pipeline.
    Requires HuggingFace token and model acceptance.
    Returns None if unavailable — system continues without diarization.
    """
    global _diarization_pipeline

    if _diarization_pipeline is not None:
        return _diarization_pipeline

    if not hf_token:
        logger.warning("No HuggingFace token provided — diarization disabled")
        logger.warning("Set HF_TOKEN in config.py to enable speaker diarization")
        return None

    try:
        from pyannote.audio import Pipeline
        import torch

        logger.info("Loading pyannote speaker diarization model...")
        logger.info("First run will download model (~1GB) — please wait...")

        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        logger.info("Speaker diarization model loaded ✅")
        return _diarization_pipeline

    except ImportError:
        logger.warning("pyannote.audio not installed")
        logger.warning("Install with: pip install pyannote-audio")
        return None

    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}")
        logger.warning("Continuing without diarization — speaker_id = UNKNOWN")
        return None


# ── DIARIZE AUDIO ─────────────────────────────────────────────────────────────

def diarize_audio(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None
) -> List[Dict]:
    """
    Run speaker diarization on full audio file.
    
    Returns list of speaker segments:
    [
        {"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2},
        {"speaker": "SPEAKER_01", "start": 4.5, "end": 8.1},
        {"speaker": "SPEAKER_00", "start": 8.3, "end": 12.0},
        ...
    ]
    
    Args:
        audio_path:   path to audio file (WAV, 16kHz mono)
        hf_token:     HuggingFace token for model access
        num_speakers: hint for expected number of speakers (optional)
                      None = auto-detect
    """
    pipeline = get_diarization_pipeline(hf_token)

    if pipeline is None:
        logger.warning("Diarization unavailable — returning empty result")
        return []

    logger.info(f"Running diarization on: {audio_path}")
    if num_speakers:
        logger.info(f"Expected speakers: {num_speakers}")

    try:
        # Run diarization
        diarization_result = pipeline(
            audio_path,
            num_speakers=num_speakers  # None = auto-detect
        )

        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3)
            })

        # Count unique speakers
        unique_speakers = list(set(s["speaker"] for s in segments))
        logger.info(f"Diarization complete | {len(unique_speakers)} speakers | {len(segments)} segments")
        for spk in sorted(unique_speakers):
            spk_time = sum(s["duration"] for s in segments if s["speaker"] == spk)
            logger.info(f"  {spk}: {spk_time:.1f}s total speaking time")

        return segments

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return []


# ── ASSIGN SPEAKERS TO CHUNKS ─────────────────────────────────────────────────

def assign_speakers_to_chunks(
    chunks_metadata: List[Dict],
    diarization_segments: List[Dict]
) -> Dict[str, Dict]:
    """
    Match each audio chunk to its dominant speaker.
    
    Strategy:
    For each chunk time range [start → end]:
    - Find all diarization segments that overlap with this chunk
    - Calculate how much each speaker spoke in this chunk
    - Assign the speaker with most speaking time as dominant speaker
    
    Returns:
        {
            "chunk_000": {"speaker_id": "SPEAKER_00", "confidence": 0.85, "all_speakers": {...}},
            "chunk_001": {"speaker_id": "SPEAKER_01", "confidence": 0.92, "all_speakers": {...}},
            ...
        }
    """
    if not diarization_segments:
        logger.warning("No diarization segments — all chunks assigned UNKNOWN")
        return {
            chunk["chunk_id"]: {
                "speaker_id": "UNKNOWN",
                "confidence": 0.0,
                "all_speakers": {}
            }
            for chunk in chunks_metadata
        }

    chunk_speaker_map = {}

    for chunk in chunks_metadata:
        chunk_id    = chunk["chunk_id"]
        chunk_start = chunk["start"]
        chunk_end   = chunk["end"]

        # Calculate overlap time per speaker in this chunk
        speaker_times = {}

        for seg in diarization_segments:
            seg_start = seg["start"]
            seg_end   = seg["end"]
            speaker   = seg["speaker"]

            # Calculate overlap
            overlap_start = max(chunk_start, seg_start)
            overlap_end   = min(chunk_end, seg_end)
            overlap       = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap

        if not speaker_times:
            # No speaker found for this chunk
            chunk_speaker_map[chunk_id] = {
                "speaker_id": "UNKNOWN",
                "confidence": 0.0,
                "all_speakers": {}
            }
            continue

        # Find dominant speaker
        dominant_speaker = max(speaker_times, key=speaker_times.get)
        total_time = sum(speaker_times.values())
        confidence = speaker_times[dominant_speaker] / total_time if total_time > 0 else 0.0

        chunk_speaker_map[chunk_id] = {
            "speaker_id": dominant_speaker,
            "confidence": round(confidence, 3),
            "all_speakers": {
                spk: round(time, 3)
                for spk, time in speaker_times.items()
            }
        }

        logger.info(
            f"{chunk_id} → {dominant_speaker} "
            f"({confidence*100:.0f}% of chunk) | "
            f"speakers: {dict(speaker_times)}"
        )

    return chunk_speaker_map


# ── FULL DIARIZATION PIPELINE ─────────────────────────────────────────────────

def process_diarization(
    audio_path: str,
    chunks_metadata: List[Dict],
    output_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None
) -> Dict:
    """
    Full diarization pipeline:
    1. Run diarization on full audio
    2. Assign speakers to each chunk
    3. Save results to JSON
    
    Args:
        audio_path:       preprocessed audio file path
        chunks_metadata:  list from Phase 1 metadata.json
        output_path:      where to save diarization results
        hf_token:         HuggingFace token
        num_speakers:     expected number of speakers (optional)
    
    Returns:
        chunk_speaker_map dict
    """
    logger.info("=" * 50)
    logger.info("PHASE 1 - STEP 4: SPEAKER DIARIZATION")
    logger.info("=" * 50)

    # Run diarization on full audio
    segments = diarize_audio(audio_path, hf_token, num_speakers)

    # Assign speakers to chunks
    chunk_speaker_map = assign_speakers_to_chunks(chunks_metadata, segments)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "diarization_segments": segments,
        "chunk_speaker_map": chunk_speaker_map,
        "total_speakers": len(set(
            v["speaker_id"] for v in chunk_speaker_map.values()
            if v["speaker_id"] != "UNKNOWN"
        ))
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Diarization results saved → {output_path}")
    logger.info(f"Total unique speakers: {result['total_speakers']}")
    logger.info("DIARIZATION COMPLETE ✅")

    return chunk_speaker_map


# ── FALLBACK — NO DIARIZATION ─────────────────────────────────────────────────

def create_unknown_speaker_map(chunks_metadata: List[Dict]) -> Dict:
    """
    Create speaker map with UNKNOWN for all chunks.
    Used when diarization is not available.
    System continues working — just without speaker tracking.
    """
    return {
        chunk["chunk_id"]: {
            "speaker_id": "UNKNOWN",
            "confidence": 0.0,
            "all_speakers": {}
        }
        for chunk in chunks_metadata
    }
