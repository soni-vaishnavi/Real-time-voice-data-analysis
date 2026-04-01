"""
Phase 1 - Step 1: Audio Preprocessor
Converts any audio format to 16kHz mono WAV + normalizes volume
"""

import os
from pydub import AudioSegment
from pydub import effects
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_audio(file_path: str) -> AudioSegment:
    """
    Load any audio file (wav, mp3, m4a, ogg etc.)
    Returns: pydub AudioSegment
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower().replace(".", "")

    logger.info(f"Loading audio file: {file_path}")
    audio = AudioSegment.from_file(file_path, format=ext if ext != "wav" else "wav")
    logger.info(f"Loaded | Duration: {len(audio)/1000:.2f}s | Channels: {audio.channels} | Sample rate: {audio.frame_rate}Hz")

    return audio


def standardize_audio(audio: AudioSegment, target_sample_rate: int = 16000) -> AudioSegment:
    """
    Convert to 16kHz mono WAV — required format for Whisper and VAD.
    
    Why 16kHz mono?
    - Whisper was trained on 16kHz audio
    - WebRTC VAD only accepts 8kHz, 16kHz, or 32kHz
    - Mono = single channel, simpler processing
    """
    # Convert stereo to mono
    if audio.channels > 1:
        logger.info(f"Converting {audio.channels} channels → mono")
        audio = audio.set_channels(1)

    # Convert sample rate
    if audio.frame_rate != target_sample_rate:
        logger.info(f"Resampling {audio.frame_rate}Hz → {target_sample_rate}Hz")
        audio = audio.set_frame_rate(target_sample_rate)

    # Ensure 16-bit depth (required by WebRTC VAD)
    audio = audio.set_sample_width(2)

    logger.info(f"Standardized | Duration: {len(audio)/1000:.2f}s | {target_sample_rate}Hz | Mono | 16-bit")
    return audio


def normalize_volume(audio: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:
    """
    Normalize audio volume to consistent level.
    
    Why normalize?
    - CCTV/phone mics have very different volume levels
    - VAD and Whisper work best at consistent volume
    - target_dBFS=-20.0 is a safe standard loudness level
    """
    current_dBFS = audio.dBFS

    if current_dBFS == float('-inf'):
        logger.warning("Audio appears to be silent — skipping normalization")
        return audio

    change_in_dBFS = target_dBFS - current_dBFS
    logger.info(f"Normalizing volume: {current_dBFS:.1f}dBFS → {target_dBFS}dBFS (change: {change_in_dBFS:+.1f}dB)")

    normalized = audio.apply_gain(change_in_dBFS)
    return normalized


def save_audio(audio: AudioSegment, output_path: str) -> str:
    """Save processed audio to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio.export(output_path, format="wav")
    logger.info(f"Saved preprocessed audio → {output_path}")
    return output_path


def preprocess(input_path: str, output_path: str, target_sample_rate: int = 16000) -> str:
    """
    Full preprocessing pipeline:
    Load → Standardize → Normalize → Save
    
    Args:
        input_path:  path to raw audio file
        output_path: path to save processed audio
        target_sample_rate: default 16000 Hz
    
    Returns:
        output_path of saved file
    """
    logger.info("=" * 50)
    logger.info("PHASE 1 - STEP 1: PREPROCESSING STARTED")
    logger.info("=" * 50)

    audio = load_audio(input_path)
    audio = standardize_audio(audio, target_sample_rate)
    audio = normalize_volume(audio)
    output_path = save_audio(audio, output_path)

    logger.info("PREPROCESSING COMPLETE ✅")
    return output_path