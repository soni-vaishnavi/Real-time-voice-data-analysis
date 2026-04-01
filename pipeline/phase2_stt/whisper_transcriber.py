"""
Phase 2 - Whisper Transcriber (v4 - optimized for speed)

Key optimizations:
1. Single model pass — no separate language detection
2. Language info extracted FREE from transcription result
3. tiny model by default on CPU (3-5 sec/chunk vs 15-25 sec)
4. Always English mode — no language switching overhead
"""

import os
import json
import logging
from typing import List, Dict, Optional
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model: Optional[WhisperModel] = None


def get_model(model_size: str = "tiny") -> WhisperModel:
    """
    Load faster-whisper model once and cache.
    
    Model size guide for CPU (8GB RAM):
    tiny  → 39M params  → 3-5 sec/chunk  → good for keywords ✅ (default)
    base  → 74M params  → 5-8 sec/chunk  → better accuracy
    small → 244M params → 15-25 sec/chunk → best accuracy, slow on CPU
    
    For production demo: use small
    For development/testing: use tiny
    """
    global _model
    if _model is None:
        logger.info(f"Loading faster-whisper '{model_size}' model...")
        _model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.info(f"faster-whisper '{model_size}' ready ✅")
    return _model


def is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination patterns"""
    if not text or len(text.strip()) < 2:
        return True

    bad_phrases = [
        "thank you for watching", "thanks for watching",
        "please subscribe", "subtitles by", "www.", "http",
        "♪", "<|hi|>", "<|en|>", "<|zh|>",
    ]
    text_lower = text.lower().strip()
    for phrase in bad_phrases:
        if phrase in text_lower:
            return True

    # Excessive repetition
    words = text.split()
    if len(words) >= 4:
        for word in set(words):
            if words.count(word) > 3 and len(word) > 2:
                logger.warning(f"Repetition hallucination: '{word}' x{words.count(word)}")
                return True

    # Corrupted unicode or wrong script
    if "\ufffd" in text:
        return True

    import re
    # CJK or Cyrillic = hallucination
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]', text):
        return True

    return False


def classify_language_mix(detected_lang: str, lang_prob: float) -> str:
    """Classify language mix from transcription result info"""
    if detected_lang == "hi" and lang_prob > 0.60:
        return "hindi"
    elif detected_lang == "en" and lang_prob > 0.60:
        return "english"
    else:
        return "hinglish"


def transcribe_chunk(
    chunk_path: str,
    chunk_id: str,
    chunk_start: float,
    model_size: str = "tiny",
    speaker_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Transcribe one chunk — SINGLE MODEL PASS only.
    Language info extracted from transcription result (free, no extra pass).
    Always English mode for speed + Hinglish compatibility.
    """
    if not os.path.exists(chunk_path):
        logger.error(f"Chunk not found: {chunk_path}")
        return None

    model = get_model(model_size)

    # ── SINGLE TRANSCRIPTION PASS ─────────────────────────────
    # language="en" always → handles Hindi/Hinglish phonetically
    # info object gives us language detection FREE
    segments_gen, info = model.transcribe(
        chunk_path,
        language="en",
        task="transcribe",
        beam_size=3,                        # reduced from 5 → faster
        word_timestamps=True,
        vad_filter=True,                    # skip silence built-in
        vad_parameters=dict(
            min_silence_duration_ms=300,
            min_speech_duration_ms=100,     # skip very short bursts
        ),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    # Consume generator
    segments = list(segments_gen)

    # ── EXTRACT LANGUAGE INFO FROM RESULT (free) ───────────────
    # info.language = what Whisper detected the audio as
    # info.language_probability = confidence of that detection
    # We get this FREE without an extra model pass
    detected_lang = info.language if hasattr(info, 'language') else "en"
    lang_prob = info.language_probability if hasattr(info, 'language_probability') else 0.0
    lang_mix = classify_language_mix(detected_lang, lang_prob)

    logger.info(
        f"{chunk_id} | detected={lang_mix} ({detected_lang}:{lang_prob:.2f})"
        + (f" | speaker={speaker_id}" if speaker_id else "")
    )

    # ── EXTRACT TEXT AND WORDS ─────────────────────────────────
    full_text = " ".join(seg.text.strip() for seg in segments).strip()
    words = []

    for segment in segments:
        if segment.no_speech_prob > 0.6:
            continue
        if segment.words:
            for word in segment.words:
                if word.probability < 0.40:
                    continue
                words.append({
                    "word": word.word.strip(),
                    "start": round(chunk_start + word.start, 3),
                    "end": round(chunk_start + word.end, 3),
                    "confidence": round(word.probability, 3),
                })

    # ── HALLUCINATION CHECK ───────────────────────────────────
    if is_hallucination(full_text):
        logger.warning(f"{chunk_id} | Hallucination — clearing")
        full_text = ""
        words = []

    avg_conf = (
        round(sum(w["confidence"] for w in words) / len(words), 3)
        if words else 0.0
    )

    transcript = {
        "chunk_id": chunk_id,
        "chunk_start": chunk_start,
        "language_detected": detected_lang,
        "detection_confidence": round(lang_prob, 3),
        "language_mix": lang_mix,
        "transcription_language_used": "en",
        "speaker_id": speaker_id if speaker_id else "UNKNOWN",
        "text": full_text,
        "words": words,
        "avg_confidence": avg_conf,
        "no_speech": full_text == ""
    }

    display = full_text[:80] + ("..." if len(full_text) > 80 else "")
    logger.info(f"{chunk_id} | '{display}'")
    return transcript


def transcribe_all_chunks(
    metadata_path: str,
    output_dir: str,
    model_size: str = "tiny",
    diarization_path: Optional[str] = None
) -> List[Dict]:
    """
    Transcribe all Phase 1 chunks.
    Model loaded ONCE. Single pass per chunk.
    """
    logger.info("=" * 50)
    logger.info("PHASE 2: TRANSCRIPTION (faster-whisper optimized)")
    logger.info("=" * 50)

    with open(metadata_path, "r") as f:
        chunks_metadata = json.load(f)

    logger.info(f"Loaded {len(chunks_metadata)} chunks")
    os.makedirs(output_dir, exist_ok=True)

    # Load diarization if available
    diarization = {}
    if diarization_path and os.path.exists(diarization_path):
        with open(diarization_path) as f:
            raw = json.load(f)
            diarization = raw.get("chunk_speaker_map", {})
        logger.info(f"Diarization loaded for {len(diarization)} chunks")

    # Load model once
    get_model(model_size)
    all_transcripts = []

    for i, chunk in enumerate(chunks_metadata):
        chunk_id    = chunk["chunk_id"]
        chunk_path  = chunk["file_path"]
        chunk_start = chunk["start"]
        speaker_id  = diarization.get(chunk_id, {}).get("speaker_id", None)

        logger.info(f"[{i+1}/{len(chunks_metadata)}] {chunk_id}")

        try:
            transcript = transcribe_chunk(
                chunk_path, chunk_id, chunk_start, model_size, speaker_id
            )
            if transcript:
                out_path = os.path.join(output_dir, f"{chunk_id}_transcript.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(transcript, f, ensure_ascii=False, indent=2)
                all_transcripts.append(transcript)
        except Exception as e:
            logger.error(f"Failed on {chunk_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    combined_path = os.path.join(output_dir, "all_transcripts.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_transcripts)} transcripts → {combined_path}")
    logger.info("PHASE 2 COMPLETE ✅")
    return all_transcripts
