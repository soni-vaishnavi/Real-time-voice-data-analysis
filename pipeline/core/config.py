"""
pipeline/core/config.py
========================
SINGLE SOURCE OF TRUTH for all VoiceGuard constants.
Every module imports from here. No module defines its own weights or thresholds.

Usage in any module:
    from pipeline.core.config import WEIGHT_EMOTION, ZONE_RED, EMERGENCY_THRESHOLD
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent.parent   # voice_surveillance/
OUTPUT_DIR      = BASE_DIR / "output"
CHUNKS_DIR      = OUTPUT_DIR / "chunks"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
ANALYSIS_DIR    = OUTPUT_DIR / "analysis"
DECISIONS_DIR   = OUTPUT_DIR / "decisions"
REPORTS_DIR     = OUTPUT_DIR / "reports"
DATABASE_PATH   = BASE_DIR / "database" / "voiceguard.db"
DATABASE_URL    = f"sqlite:///{DATABASE_PATH}"

# ── PHASE 1: AUDIO ────────────────────────────────────────────────────────────
SAMPLE_RATE           = 16000
CHUNK_WINDOW_SEC      = 5
CHUNK_OVERLAP_SEC     = 2
VAD_AGGRESSIVENESS    = 2
NOISE_REDUCE_STRENGTH = 0.75

# ── PHASE 2: STT ──────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE  = "tiny"   # tiny=75MB, base=145MB, small=460MB
WHISPER_DEVICE      = "cpu"
WHISPER_BEAM_SIZE   = 3
WORD_CONFIDENCE_MIN = 0.40

# ── PHASE 3: MODELS ───────────────────────────────────────────────────────────
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"
EMERGENCY_MODEL = "facebook/bart-large-mnli"

EMERGENCY_LABELS = [
    "medical emergency, someone needs a doctor or ambulance",
    "fire emergency, there is a fire or smoke",
    "violence or physical assault, someone is being attacked or hurt",
    "accident or injury, someone has fallen or been injured",
    "theft or robbery, someone is stealing or being robbed",
    "mental health crisis, someone wants to harm themselves",
    "normal conversation, no emergency",
]

EMERGENCY_LABEL_SHORT = {
    "medical emergency, someone needs a doctor or ambulance":           "medical",
    "fire emergency, there is a fire or smoke":                         "fire",
    "violence or physical assault, someone is being attacked or hurt":  "violence",
    "accident or injury, someone has fallen or been injured":           "accident",
    "theft or robbery, someone is stealing or being robbed":            "theft",
    "mental health crisis, someone wants to harm themselves":           "mental_health",
    "normal conversation, no emergency":                                "normal",
}

# EMERGENCY_THRESHOLD = 0.55 (was 0.35 in old emergency_detector, 0.55 in old analyzer — unified here)
EMERGENCY_THRESHOLD     = 0.55
MIN_WORDS_FOR_EMERGENCY = 4

# ── PHASE 4: SCORING ──────────────────────────────────────────────────────────
# Weights MUST sum to 1.0 — enforced by validate_config()
WEIGHT_EMOTION   = 0.35
WEIGHT_EMERGENCY = 0.40
WEIGHT_KEYWORD   = 0.25

ZONE_YELLOW = 0.45   # score >= ZONE_YELLOW → YELLOW
ZONE_RED    = 0.72   # score >= ZONE_RED    → RED

CONSECUTIVE_YELLOW_LIMIT = 3
RAPID_RISE_THRESHOLD     = 0.20
INCIDENT_GAP_LIMIT       = 2

AUTO_TRIGGER_SEC       = 60
AUTO_TRIGGER_MIN_SCORE = 0.90

# ── PHASE 5: ALERTS ───────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
ALERT_NUMBERS      = [n for n in [
    os.getenv("ALERT_NUMBER_1", ""), os.getenv("ALERT_NUMBER_2", ""),
] if n]

SMTP_EMAIL    = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAILS  = [e for e in [
    os.getenv("ALERT_EMAIL_1", ""), os.getenv("ALERT_EMAIL_2", ""),
] if e]

TWILIO_ENABLED = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
EMAIL_ENABLED  = bool(SMTP_EMAIL and SMTP_PASSWORD)

SOUND_YELLOW = str(BASE_DIR / "assets" / "beep.wav")
SOUND_RED    = str(BASE_DIR / "assets" / "alarm.wav")

# ── STAGE 2: QUEUE ────────────────────────────────────────────────────────────
QUEUE_MAXSIZE = 10   # bounded — drops oldest chunk if full (back-pressure)

# ── STARTUP VALIDATION ────────────────────────────────────────────────────────
def validate_config() -> None:
    """Called at startup. Fails loudly on any misconfiguration."""
    w = WEIGHT_EMOTION + WEIGHT_EMERGENCY + WEIGHT_KEYWORD
    assert abs(w - 1.0) < 0.001, \
        f"Scoring weights must sum to 1.0, got {w:.4f}"
    assert 0.0 < ZONE_YELLOW < ZONE_RED < 1.0, \
        f"Zone thresholds: 0 < ZONE_YELLOW({ZONE_YELLOW}) < ZONE_RED({ZONE_RED}) < 1.0"
    assert 0.0 < EMERGENCY_THRESHOLD < 1.0, \
        f"EMERGENCY_THRESHOLD must be in (0,1), got {EMERGENCY_THRESHOLD}"
    assert WHISPER_MODEL_SIZE in ("tiny","base","small","medium"), \
        f"Invalid WHISPER_MODEL_SIZE: {WHISPER_MODEL_SIZE}"
    for d in [OUTPUT_DIR, CHUNKS_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR,
              DECISIONS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "database").mkdir(parents=True, exist_ok=True)