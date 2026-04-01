"""
Phase 3 - Audio Emotion Recognizer
Detects emotion directly from voice audio using wav2vec2.

Why this matters:
- Text says "I'm fine" but voice is trembling = real distress
- Text says "I'm going to die" but voice is calm = sarcasm
- Voice pitch, speed, tremor cannot be faked easily

Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
- Input: WAV audio chunk (directly from Phase 1 output)
- Output: 8 emotion probabilities
- Size: ~1.2GB, CPU compatible
"""

import os
import logging
import numpy as np
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_audio_emotion_pipeline = None

# Map wav2vec2 model labels to our standard emotion names
AUDIO_EMOTION_LABEL_MAP = {
    "angry":     "anger",
    "anger":     "anger",
    "disgust":   "disgust",
    "fearful":   "fear",
    "fear":      "fear",
    "happy":     "joy",
    "joy":       "joy",
    "neutral":   "neutral",
    "sad":       "sadness",
    "sadness":   "sadness",
    "surprised": "surprise",
    "surprise":  "surprise",
    "calm":      "neutral",
}

# Same emergency weights as text model
EMOTION_EMERGENCY_WEIGHT = {
    "fear":     1.0,
    "anger":    0.7,
    "sadness":  0.5,
    "surprise": 0.3,
    "disgust":  0.2,
    "joy":      0.0,
    "neutral":  0.0,
}


def get_audio_emotion_model():
    """Load wav2vec2 audio emotion model once and cache"""
    global _audio_emotion_pipeline
    if _audio_emotion_pipeline is None:
        import torch
        from transformers import pipeline
        logger.info("Loading audio emotion model (wav2vec2)...")
        logger.info("First run downloads ~1.2GB — please wait...")
        _audio_emotion_pipeline = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            framework="pt",
            device=-1,          # CPU
        )
        logger.info("Audio emotion model ready ✅")
    return _audio_emotion_pipeline


def detect_audio_emotion(audio_path: str) -> Dict:
    """
    Detect emotion from raw audio file.
    
    Args:
        audio_path: path to WAV chunk from Phase 1
    
    Returns:
        {
            dominant_emotion: "fear",
            dominant_score: 0.78,
            all_scores: {fear: 0.78, neutral: 0.12, ...},
            emergency_weight: 1.0,
            source: "audio"
        }
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return _empty_audio_emotion_result()

    try:
        model = get_audio_emotion_model()

        # Run audio emotion classification
        # pipeline handles audio loading internally
        results = model(audio_path, top_k=None)

        # Normalize labels to our standard names
        all_scores = {}
        for item in results:
            label = AUDIO_EMOTION_LABEL_MAP.get(
                item["label"].lower(), item["label"].lower()
            )
            # Accumulate if same label from multiple raw labels
            all_scores[label] = all_scores.get(label, 0) + item["score"]

        # Normalize scores to sum to 1.0
        total = sum(all_scores.values())
        if total > 0:
            all_scores = {k: round(v / total, 4) for k, v in all_scores.items()}

        # Find dominant
        dominant = max(all_scores, key=all_scores.get)
        dominant_score = all_scores[dominant]
        emergency_weight = EMOTION_EMERGENCY_WEIGHT.get(dominant, 0.0)

        return {
            "dominant_emotion": dominant,
            "dominant_score": dominant_score,
            "all_scores": all_scores,
            "emergency_weight": emergency_weight,
            "fear_score": all_scores.get("fear", 0.0),
            "anger_score": all_scores.get("anger", 0.0),
            "source": "audio"
        }

    except Exception as e:
        logger.error(f"Audio emotion detection failed: {e}")
        return _empty_audio_emotion_result()


def _empty_audio_emotion_result() -> Dict:
    return {
        "dominant_emotion": "neutral",
        "dominant_score": 0.0,
        "all_scores": {},
        "emergency_weight": 0.0,
        "fear_score": 0.0,
        "anger_score": 0.0,
        "source": "audio_unavailable"
    }


# ── AUDIO FEATURE EXTRACTION ──────────────────────────────────────────────────

def extract_voice_features(audio_path: str) -> Dict:
    """
    Extract low-level voice features that help detect emotional state.
    These features complement the ML model with interpretable signals.

    Features extracted:
    - pitch_mean:     average fundamental frequency (Hz)
                      high pitch = stress/fear, low = calm
    - pitch_std:      pitch variation (tremor indicator)
                      high variation = emotional instability
    - speech_rate:    words per second estimate from energy
                      fast = anxiety/anger, slow = sadness/calm
    - energy_mean:    average loudness
                      high = shouting/anger, low = whispering/fear
    - energy_std:     loudness variation
                      high = emotional speech
    - zcr_mean:       zero crossing rate (voice quality)
                      high = tense/harsh voice

    Returns dict of features or empty dict if extraction fails.
    """
    try:
        import librosa
        import numpy as np

        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        if len(y) < sr * 0.5:   # skip very short clips
            return {}

        # Pitch (fundamental frequency)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])

        pitch_mean = float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_std  = float(np.nanstd(f0_voiced))  if len(f0_voiced) > 0 else 0.0

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std  = float(np.std(rms))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        # Speech rate estimate (energy peaks = syllable approximation)
        # Rough estimate: count energy peaks per second
        peaks = np.where(np.diff(np.sign(np.diff(rms))))[0]
        speech_rate = len(peaks) / (len(y) / sr) if len(y) > 0 else 0.0

        features = {
            "pitch_mean":   round(pitch_mean, 2),
            "pitch_std":    round(pitch_std, 2),
            "energy_mean":  round(energy_mean, 4),
            "energy_std":   round(energy_std, 4),
            "zcr_mean":     round(zcr_mean, 4),
            "speech_rate":  round(speech_rate, 2),
        }

        # Interpret features into emotion hints
        hints = []
        if pitch_mean > 220 and pitch_std > 40:
            hints.append("high_pitch_variation → stress/fear signal")
        if energy_mean > 0.08:
            hints.append("high_energy → shouting/anger signal")
        if speech_rate > 8:
            hints.append("fast_speech → anxiety/anger signal")
        if pitch_mean < 120 and energy_mean < 0.02:
            hints.append("low_pitch_energy → sadness/depression signal")

        features["interpretation_hints"] = hints
        return features

    except ImportError:
        logger.warning("librosa not installed — voice features unavailable")
        logger.warning("Install with: pip install librosa")
        return {}
    except Exception as e:
        logger.error(f"Voice feature extraction failed: {e}")
        return {}


# ── FULL AUDIO ANALYSIS ───────────────────────────────────────────────────────

def analyze_audio_emotion(chunk_path: str) -> Dict:
    """
    Run full audio emotion analysis on one chunk.
    Combines wav2vec2 model + voice feature extraction.
    """
    if not chunk_path or not os.path.exists(chunk_path):
        return {
            "audio_emotion": _empty_audio_emotion_result(),
            "voice_features": {},
            "available": False
        }

    audio_emotion = detect_audio_emotion(chunk_path)
    voice_features = extract_voice_features(chunk_path)

    return {
        "audio_emotion": audio_emotion,
        "voice_features": voice_features,
        "available": True
    }
