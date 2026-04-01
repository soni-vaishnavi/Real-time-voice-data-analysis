"""
Phase 1 - Step 2: Noise Reducer
Removes background noise (fan, crowd, mic hiss) from audio
"""

import os
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audiosegment_to_numpy(audio: AudioSegment) -> np.ndarray:
    """Convert pydub AudioSegment → numpy array for noisereduce"""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # Normalize to -1.0 to 1.0 range
    samples = samples / (2 ** 15)
    return samples


def numpy_to_audiosegment(samples: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert numpy array → pydub AudioSegment"""
    # Convert back to int16
    samples_int16 = (samples * (2 ** 15)).astype(np.int16)
    audio = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,   # 16-bit = 2 bytes
        channels=1
    )
    return audio


def reduce_noise(audio: AudioSegment, strength: float = 0.75) -> AudioSegment:
    """
    Remove background noise from audio.
    
    How it works:
    - noisereduce analyzes the audio for consistent background noise
    - Builds a noise profile (what does silence/background sound like?)
    - Subtracts that noise profile from the full audio
    - Result: only speech remains
    
    Args:
        audio:    preprocessed AudioSegment (must be 16kHz mono)
        strength: noise reduction strength 0.0-1.0
                  0.5 = gentle (preserves more audio quality)
                  0.75 = balanced (recommended)
                  1.0 = aggressive (may distort speech)
    
    Returns:
        AudioSegment with noise reduced
    """
    logger.info(f"Applying noise reduction (strength={strength})")

    sample_rate = audio.frame_rate
    samples = audiosegment_to_numpy(audio)

    # Apply noise reduction
    # stationary=True means we assume background noise is consistent
    # (fan, AC, crowd hum — all consistent noises)
    reduced_samples = nr.reduce_noise(
        y=samples,
        sr=sample_rate,
        stationary=True,
        prop_decrease=strength    # how much to reduce noise
    )

    result_audio = numpy_to_audiosegment(reduced_samples, sample_rate)

    logger.info(f"Noise reduction complete | Duration preserved: {len(result_audio)/1000:.2f}s")
    return result_audio


def process_noise_reduction(input_path: str, output_path: str, strength: float = 0.75) -> str:
    """
    Load preprocessed audio → reduce noise → save
    
    Args:
        input_path:  preprocessed WAV file path
        output_path: path to save noise-reduced audio
        strength:    noise reduction strength (default 0.75)
    
    Returns:
        output_path of saved file
    """
    logger.info("=" * 50)
    logger.info("PHASE 1 - STEP 2: NOISE REDUCTION STARTED")
    logger.info("=" * 50)

    # Load audio
    audio = AudioSegment.from_wav(input_path)
    logger.info(f"Loaded: {input_path} | {len(audio)/1000:.2f}s")

    # Reduce noise
    audio = reduce_noise(audio, strength=strength)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio.export(output_path, format="wav")
    logger.info(f"Saved noise-reduced audio → {output_path}")

    logger.info("NOISE REDUCTION COMPLETE ✅")
    return output_path