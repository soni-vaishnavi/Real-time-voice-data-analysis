"""
pipeline/phase2_stt/whisper_transcriber.py
==========================================
Phase 2 - Whisper Transcriber (v6 — Translation mode)

ROOT CAUSE FIXES over v5:
  1. task="translate" instead of "transcribe"
     - Whisper translates ALL languages directly to English
     - "bachao ambulance bulao" → "save me call the ambulance"
     - No Devanagari output, no transliteration needed
     - Consistent English output for all chunks — keyword matching works perfectly

  2. initial_prompt REMOVED
     - v5's prompt was causing chunk_022 to output exactly:
       "jaldi, nahi, karo, jaldi, please, save, attack, chor, robbery"
     - Whisper leaks prompt words when confused on Hinglish audio
     - Remove it entirely — auto-detect + translate is enough

  3. Devanagari / repetition loop protection
     - Added timeout via threading (caps inference at 60s/chunk)
     - Expanded hallucination filter catches Devanagari loops
     - Catches। (danda) repetitions, ो ौ loops, etc.

  4. beam_size=3 for translate mode
     - Translation beam_size=5 causes much slower loops
     - beam_size=3 is fast enough + temperature fallback handles quality

  5. Energy check before Whisper
     - Silent chunks are skipped before expensive inference

Translation accuracy for Hindi/Hinglish emergency words:
  Hindi → English (Whisper translate):
    bachao         → save / help me
    ambulance bulao → call the ambulance
    aag lagi       → fire has broken out
    goli maari     → shot was fired
    khoon           → blood / bleeding
    dard ho raha    → feeling pain
    doctor chahiye  → need a doctor
"""

import os
import json
import logging
import threading
import numpy as np
from typing import List, Dict, Optional
from faster_whisper import WhisperModel

from pipeline.phase2_stt.keyword_normalizer import apply_keyword_normalization_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model: Optional[WhisperModel] = None

# ── NO INITIAL PROMPT ──────────────────────────────────────────────────────────
# Removed: initial_prompt caused Whisper to output prompt keywords as transcript
# on Hindi chunks (chunk_022 was literally outputting "jaldi, nahi, karo...")
# task="translate" handles domain context internally.
HINGLISH_INITIAL_PROMPT = None   # kept for worker.py import compatibility


def get_model(model_size: str = "small") -> WhisperModel:
    """
    Load faster-whisper model once and cache.
    "small" is the sweet spot for Hinglish translation accuracy.
    """
    global _model
    if _model is None:
        logger.info(f"Loading faster-whisper '{model_size}' model...")
        _model = WhisperModel(
            model_size,
            device       = "cpu",
            compute_type = "int8",
        )
        logger.info(f"faster-whisper '{model_size}' ready ✅")
    return _model


def _check_audio_energy(audio_path: str, min_rms: float = 0.001) -> bool:
    """Return False if audio is too quiet to contain speech."""
    try:
        import wave
        with wave.open(audio_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        if not frames:
            return False
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return rms >= min_rms
    except Exception:
        return True


def is_hallucination(text: str) -> bool:
    """
    Detect Whisper hallucination / garbage output.
    Expanded to catch Devanagari loops and punctuation-only output.
    """
    if not text or len(text.strip()) < 2:
        return True

    text_stripped = text.strip()

    # Devanagari danda punctuation repetition: '। । । ।'
    # This happens when Whisper transcribes silence in Hindi mode
    import re
    if re.match(r'^[\u0964\u0965\s]+$', text_stripped):
        return True

    # Pure Devanagari vowel marks repetition: 'ो ो ो ो' or 'ौ ौ ौ'
    if re.match(r'^[\u0900-\u097F\s।॥]+$', text_stripped):
        deva_chars = re.sub(r'\s', '', text_stripped)
        # If more than 5 chars and all Devanagari — it's a loop
        if len(deva_chars) > 5:
            return True

    # Common hallucination phrases
    bad_phrases = [
        "thank you for watching", "thanks for watching",
        "please subscribe", "subtitles by", "www.", "http",
        "♪", "by amara.org", "♫",
        "<|hi|>", "<|en|>", "<|zh|>",
    ]
    text_lower = text.lower().strip()
    for phrase in bad_phrases:
        if phrase in text_lower:
            return True

    # Excessive single-word repetition (hallucination loop)
    words = text.split()
    if len(words) >= 4:
        for word in set(words):
            if len(word) > 2 and words.count(word) > 3:
                logger.warning(f"Repetition hallucination: '{word}' x{words.count(word)}")
                return True

    # Corrupted unicode
    if "\ufffd" in text:
        return True

    # CJK or Cyrillic
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]', text):
        return True

    # Only punctuation / symbols — no alphanumeric
    clean = re.sub(r'[^a-zA-Z\u0900-\u097F0-9]', '', text)
    if len(clean) < 2:
        return True

    return False


def classify_language_mix(detected_lang: str, lang_prob: float) -> str:
    """Classify detected language."""
    if detected_lang == "hi" and lang_prob > 0.55:
        return "hindi"
    elif detected_lang == "en" and lang_prob > 0.60:
        return "english"
    else:
        return "hinglish"


def _transcribe_with_timeout(
    model: WhisperModel,
    audio_path: str,
    timeout_sec: int = 45,
    **kwargs
) -> Optional[tuple]:
    """
    Run model.transcribe() with a timeout.
    Prevents 80-second hangs when Whisper loops on Hindi audio.

    Returns (segments_list, info) or None on timeout.
    """
    result_holder = [None]
    error_holder  = [None]

    def _run():
        try:
            gen, info = model.transcribe(audio_path, **kwargs)
            result_holder[0] = (list(gen), info)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        # Thread still running = timeout
        logger.warning(f"Whisper inference timed out after {timeout_sec}s — skipping chunk")
        return None

    if error_holder[0]:
        raise error_holder[0]

    return result_holder[0]


def transcribe_chunk(
    chunk_path:  str,
    chunk_id:    str,
    chunk_start: float,
    model_size:  str = "small",
    speaker_id:  Optional[str] = None,
) -> Optional[Dict]:
    """
    Transcribe (translate to English) one 5-second audio chunk.

    Key settings:
    - task="translate":   ALL languages → English output
    - language=None:      auto-detect source language
    - beam_size=3:        fast + temperature fallback for quality
    - NO initial_prompt:  was causing keyword hallucination in Hindi chunks
    - timeout=45s:        prevents infinite Devanagari loops
    """
    if not os.path.exists(chunk_path):
        logger.error(f"Chunk not found: {chunk_path}")
        return None

    # Skip silent chunks before expensive Whisper inference
    if not _check_audio_energy(chunk_path):
        logger.info(f"{chunk_id} | Silent chunk — skipping")
        return _empty_transcript(chunk_id, chunk_start, speaker_id)

    model = get_model(model_size)

    result = _transcribe_with_timeout(
        model,
        chunk_path,
        timeout_sec = 45,
        # ── KEY SETTINGS ──────────────────────────────────────────
        language         = None,         # auto-detect source language
        task             = "translate",  # → always outputs English
        beam_size        = 3,            # beam_size=5 causes slower loops
        best_of          = 3,
        # NO initial_prompt — it caused keyword hallucination
        word_timestamps  = True,
        vad_filter       = True,
        vad_parameters   = dict(
            min_silence_duration_ms = 200,
            min_speech_duration_ms  = 100,
        ),
        temperature      = [0.0, 0.2, 0.4, 0.6],  # more fallbacks
        condition_on_previous_text  = False,
        compression_ratio_threshold = 2.0,   # stricter than default 2.4
        log_prob_threshold          = -0.8,  # stricter than default -1.0
        no_speech_threshold         = 0.6,
    )

    if result is None:
        # Timeout — return empty
        return _empty_transcript(chunk_id, chunk_start, speaker_id)

    segments, info = result
    full_text = " ".join(seg.text.strip() for seg in segments).strip()

    # Hallucination check
    if is_hallucination(full_text):
        logger.warning(f"{chunk_id} | Hallucination detected — clearing")
        full_text = ""
        segments  = []

    # Extract words
    words = []
    for seg in segments:
        if seg.no_speech_prob > 0.6:
            continue
        if seg.words:
            for w in seg.words:
                if w.probability < 0.30:
                    continue
                words.append({
                    "word":       w.word.strip(),
                    "start":      round(chunk_start + w.start, 3),
                    "end":        round(chunk_start + w.end, 3),
                    "confidence": round(w.probability, 3),
                })

    detected_lang = getattr(info, "language", "?")
    lang_prob     = getattr(info, "language_probability", 0.0)
    lang_mix      = classify_language_mix(detected_lang, lang_prob)
    avg_conf      = round(sum(w["confidence"] for w in words) / len(words), 3) if words else 0.0

    logger.info(f"{chunk_id} | {lang_mix} ({detected_lang}:{lang_prob:.2f}) "
                + (f"| spk={speaker_id}" if speaker_id else ""))
    logger.info(f"{chunk_id} | '{full_text[:80]}'")

    return {
        "chunk_id":             chunk_id,
        "chunk_start":          chunk_start,
        "language_detected":    detected_lang,
        "detection_confidence": round(lang_prob, 3),
        "language_mix":         lang_mix,
        "transcription_task":   "translate",
        "speaker_id":           speaker_id or "UNKNOWN",
        "text":                 full_text,
        "words":                words,
        "avg_confidence":       avg_conf,
        "no_speech":            full_text == "",
    }


def _empty_transcript(chunk_id: str, chunk_start: float, speaker_id: Optional[str]) -> Dict:
    return {
        "chunk_id":             chunk_id,
        "chunk_start":          chunk_start,
        "language_detected":    "?",
        "detection_confidence": 0.0,
        "language_mix":         "unknown",
        "transcription_task":   "translate",
        "speaker_id":           speaker_id or "UNKNOWN",
        "text":                 "",
        "words":                [],
        "avg_confidence":       0.0,
        "no_speech":            True,
    }


def transcribe_all_chunks(
    metadata_path:    str,
    output_dir:       str,
    model_size:       str = "small",
    diarization_path: Optional[str] = None,
) -> List[Dict]:
    """
    Transcribe (translate to English) all Phase 1 chunks.
    Saves output_dir/all_transcripts.json
    """
    logger.info("=" * 55)
    logger.info(f"PHASE 2: TRANSCRIPTION → TRANSLATION TO ENGLISH")
    logger.info(f"         Model: {model_size} | task=translate | auto language detect")
    logger.info("=" * 55)

    with open(metadata_path) as f:
        chunks_metadata = json.load(f)

    logger.info(f"Loaded {len(chunks_metadata)} chunks")
    os.makedirs(output_dir, exist_ok=True)

    # Load diarization if available
    diarization = {}
    if diarization_path and os.path.exists(diarization_path):
        with open(diarization_path) as f:
            raw = json.load(f)
            diarization = raw.get("chunk_speaker_map", {})

    get_model(model_size)   # pre-load
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

    # Phase 2 should attach keyword metadata before downstream analysis.
    all_transcripts = apply_keyword_normalization_all(all_transcripts)

    combined_path = os.path.join(output_dir, "all_transcripts.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, ensure_ascii=False, indent=2)

    combined_final_path = os.path.join(output_dir, "all_transcripts_final.json")
    with open(combined_final_path, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, ensure_ascii=False, indent=2)

    speech_count = sum(1 for t in all_transcripts if not t.get("no_speech"))
    logger.info(f"Saved {len(all_transcripts)} transcripts → {combined_path}")
    logger.info(f"Speech in {speech_count}/{len(all_transcripts)} chunks (translated to English)")
    logger.info("PHASE 2 COMPLETE ✅")
    return all_transcripts