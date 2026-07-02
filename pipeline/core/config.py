"""
pipeline/core/config.py
========================
SINGLE SOURCE OF TRUTH for all VoiceGuard constants.

Changes in this version:
  - WHISPER_MODEL_SIZE default changed to "small" (was "tiny")
    small gives ~80% Hinglish accuracy vs ~40% for tiny
  - DASHBOARD_DATA_SOURCE added for Stage 7
    "file" = read all_decisions.json (original behavior)
    "api"  = poll GET /chunks from FastAPI (live mode behavior)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent.parent
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
CHUNK_WINDOW_SEC      = 5                # File mode chunk size (batched processing)
LIVE_CHUNK_WINDOW_SEC = 2                # Live mode chunk size (real-time, lower latency)
CHUNK_OVERLAP_SEC     = 2
VAD_AGGRESSIVENESS    = 2
NOISE_REDUCE_STRENGTH = 0.75
AUDIO_LEVEL_WARNING_THRESHOLD = 500      # RMS level below this = too quiet (warn user)

# ── PHASE 2: STT ──────────────────────────────────────────────────────────────
# CHANGED: "small" default — ~80% Hinglish accuracy vs ~40% for "tiny"
# tiny=39MB/5s, base=74MB/8s, small=244MB/12s, medium=769MB/30s (all CPU times)
WHISPER_MODEL_SIZE  = "small"
WHISPER_DEVICE      = "cpu"
WHISPER_BEAM_SIZE   = 5       # CHANGED: was 3
WORD_CONFIDENCE_MIN = 0.35    # CHANGED: was 0.40 (slightly more lenient for Hindi)

# ── PHASE 3: MODELS ───────────────────────────────────────────────────────────
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"
EMERGENCY_MODEL = "facebook/bart-large-mnli"

EMERGENCY_LABELS = [
    "medical emergency",
    "fire emergency",
    "violence or assault",
    "accident or injury",
    "theft or robbery",
    "mental health crisis",
    "normal conversation",
]

EMERGENCY_LABEL_SHORT = {
    "medical emergency":           "medical",
    "fire emergency":               "fire",
    "violence or assault":          "violence",
    "accident or injury":           "accident",
    "theft or robbery":             "theft",
    "mental health crisis":         "mental_health",
    "normal conversation":          "normal",
}

EMERGENCY_THRESHOLD     = 0.55
MIN_WORDS_FOR_EMERGENCY = 4

# ── PHASE 4: SCORING ──────────────────────────────────────────────────────────
WEIGHT_EMOTION   = 0.25  # Reduced from 0.35
WEIGHT_EMERGENCY = 0.55  # Increased from 0.40
WEIGHT_KEYWORD   = 0.20  # Reduced from 0.25

ZONE_YELLOW = 0.45
ZONE_RED    = 0.55  # Further lowered from 0.60

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
QUEUE_MAXSIZE = 10

# ── STAGE 7: DASHBOARD DATA SOURCE ────────────────────────────────────────────
# "file" → read output/decisions/all_decisions.json (file mode, original behavior)
# "api"  → poll GET http://localhost:8000/chunks  (live mode, real-time)
DASHBOARD_DATA_SOURCE = os.getenv("DASHBOARD_DATA_SOURCE", "file")  # "file" | "api"
DASHBOARD_API_URL     = os.getenv("DASHBOARD_API_URL", "http://localhost:8000")

# ── STARTUP VALIDATION ────────────────────────────────────────────────────────
def validate_config() -> None:
    w = WEIGHT_EMOTION + WEIGHT_EMERGENCY + WEIGHT_KEYWORD
    assert abs(w - 1.0) < 0.001, f"Scoring weights must sum to 1.0, got {w:.4f}"
    assert 0.0 < ZONE_YELLOW < ZONE_RED < 1.0
    assert 0.0 < EMERGENCY_THRESHOLD < 1.0
    assert WHISPER_MODEL_SIZE in ("tiny","base","small","medium")
    for d in [OUTPUT_DIR, CHUNKS_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR,
              DECISIONS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "database").mkdir(parents=True, exist_ok=True)