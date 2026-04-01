"""
Phase 5 - Sound Alert
Local sound alarm using pygame.

Logic:
    GREEN  zone → silence
    YELLOW zone → soft beep (one time)
    RED    zone → loud continuous alarm until dismissed

Assets needed (place in voice_surveillance/assets/):
    beep.wav   → short soft beep  (~0.5 sec)
    alarm.wav  → loud alarm sound (~3-5 sec, will loop)

If assets don't exist → generates simple tones using pygame itself.
No external sound files required for testing.
"""

import os
import threading
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── STATE ──────────────────────────────────────────────────────────────────────
_pygame_ready   = False
_alarm_active   = False
_alarm_thread   = None
_stop_alarm_flag = threading.Event()

# ── ASSET PATHS ────────────────────────────────────────────────────────────────
ASSETS_DIR  = "assets"
BEEP_PATH   = os.path.join(ASSETS_DIR, "beep.wav")
ALARM_PATH  = os.path.join(ASSETS_DIR, "alarm.wav")


def _init_pygame():
    """Initialize pygame mixer once. Safe to call multiple times."""
    global _pygame_ready
    if _pygame_ready:
        return True
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        _pygame_ready = True
        logger.info("pygame mixer initialized ✅")
        return True
    except Exception as e:
        logger.warning(f"pygame init failed: {e} — sound alerts disabled")
        return False


def _generate_tone(frequency: int, duration_ms: int, volume: float = 0.5) -> "pygame.mixer.Sound":
    """
    Generate a simple sine wave tone using numpy + pygame.
    Used when .wav files are not present.
    """
    import pygame
    import numpy as np

    sample_rate = 44100
    n_samples   = int(sample_rate * duration_ms / 1000)
    t           = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave        = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    wave        = np.column_stack([wave, wave])   # stereo
    sound       = pygame.sndarray.make_sound(wave)
    return sound


def _get_sound(path: str, fallback_freq: int, fallback_dur_ms: int):
    """Load .wav file if exists, otherwise generate tone."""
    import pygame
    if os.path.exists(path):
        try:
            return pygame.mixer.Sound(path)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e} — using generated tone")
    return _generate_tone(fallback_freq, fallback_dur_ms)


# ── PUBLIC FUNCTIONS ───────────────────────────────────────────────────────────

def play_yellow_beep():
    """
    Play a single soft beep for YELLOW zone.
    Non-blocking — returns immediately.
    """
    if not _init_pygame():
        logger.info("[SOUND] YELLOW beep (pygame unavailable — silent)")
        return

    import pygame
    try:
        sound = _get_sound(BEEP_PATH, fallback_freq=880, fallback_dur_ms=400)
        sound.set_volume(0.4)
        sound.play()
        logger.info("[SOUND] YELLOW beep played")
    except Exception as e:
        logger.warning(f"[SOUND] Beep failed: {e}")


def start_red_alarm():
    """
    Start continuous RED alarm in background thread.
    Loops until stop_alarm() is called.
    """
    global _alarm_active, _alarm_thread, _stop_alarm_flag

    if _alarm_active:
        logger.info("[SOUND] Alarm already running")
        return

    if not _init_pygame():
        logger.warning("[SOUND] RED ALARM TRIGGERED (pygame unavailable — silent)")
        _alarm_active = True
        return

    _stop_alarm_flag.clear()
    _alarm_active = True

    def _alarm_loop():
        import pygame
        try:
            sound = _get_sound(ALARM_PATH, fallback_freq=1200, fallback_dur_ms=800)
            sound.set_volume(0.9)
            logger.warning("[SOUND] 🔴 RED ALARM STARTED — call stop_alarm() to dismiss")
            while not _stop_alarm_flag.is_set():
                sound.play()
                time.sleep(sound.get_length() + 0.05)
        except Exception as e:
            logger.error(f"[SOUND] Alarm loop error: {e}")
        finally:
            logger.info("[SOUND] Alarm stopped")

    _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
    _alarm_thread.start()


def stop_alarm():
    """Stop the RED alarm if it's running."""
    global _alarm_active
    if not _alarm_active:
        return
    _stop_alarm_flag.set()
    _alarm_active = False
    try:
        import pygame
        pygame.mixer.stop()
    except Exception:
        pass
    logger.info("[SOUND] Alarm dismissed ✅")


def is_alarm_active() -> bool:
    return _alarm_active


def trigger_for_zone(zone: str):
    """
    Single call to trigger correct sound for a zone.
    Called by dashboard.py when a new chunk arrives.
    """
    if zone == "RED":
        start_red_alarm()
    elif zone == "YELLOW":
        play_yellow_beep()
    # GREEN → silence, do nothing