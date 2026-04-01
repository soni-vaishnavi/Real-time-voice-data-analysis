# config.py

# ── PATHS ──────────────────────────────────────
INPUT_AUDIO_DIR     = "input/audio/"
DATASET_DIR         = "input/datasets/"
OUTPUT_CHUNKS_DIR   = "output/chunks/"
OUTPUT_TRANSCRIPTS  = "output/transcripts/"
OUTPUT_ANALYSIS     = "output/analysis/"
OUTPUT_DECISIONS    = "output/decisions/"
OUTPUT_REPORTS      = "output/reports/"
DATABASE_PATH       = "database/incidents.db"

# ── PHASE 1 SETTINGS ───────────────────────────
SAMPLE_RATE         = 16000      # Hz — Whisper expects this
CHUNK_WINDOW_SEC    = 5          # sliding window size
CHUNK_OVERLAP_SEC   = 2          # overlap between chunks
VAD_AGGRESSIVENESS  = 2          # 0-3, higher = stricter

# ── PHASE 2 SETTINGS ───────────────────────────
WHISPER_MODEL_SIZE  = "small"    # tiny/base/small/medium
WHISPER_LANGUAGE    = "hi"       # primary language
CONFIDENCE_THRESHOLD = 0.60      # ignore words below this

# ── PHASE 3 SETTINGS ───────────────────────────
EMOTION_MODEL       = "j-hartmann/emotion-english-distilroberta-base"
SARCASM_MODEL       = "helinivan/distilbert-base-uncased-finetuned-sarcasm"
EMERGENCY_MODEL     = "facebook/bart-large-mnli"
EMERGENCY_LABELS    = [
    "medical emergency",
    "fire emergency",
    "violence or fight",
    "accident",
    "theft or robbery",
    "mental health crisis",
    "normal conversation"
]

# ── PHASE 4 SETTINGS ───────────────────────────
WEIGHT_EMOTION      = 0.35
WEIGHT_EMERGENCY    = 0.40
WEIGHT_KEYWORD      = 0.25
ZONE_YELLOW         = 0.45       # threshold for yellow
ZONE_RED            = 0.72       # threshold for red
ESCALATION_COUNT    = 3          # consecutive yellow → red
AUTO_TRIGGER_SEC    = 60         # seconds before auto alert
AUTO_TRIGGER_MIN_SCORE = 0.90    # score needed for auto trigger

# ── PHASE 5 SETTINGS ───────────────────────────
TWILIO_ACCOUNT_SID  = "your_sid_here"
TWILIO_AUTH_TOKEN   = "your_token_here"
TWILIO_FROM_NUMBER  = "+1XXXXXXXXXX"
ALERT_NUMBERS       = ["+91XXXXXXXXXX"]   # your test number

SMTP_EMAIL          = "your_gmail@gmail.com"
SMTP_PASSWORD       = "your_app_password"
ALERT_EMAILS        = ["your_email@gmail.com"]

SOUND_YELLOW        = "assets/beep.wav"
SOUND_RED           = "assets/alarm.wav"
```

---

## requirements.txt — Every Dependency
```
# Phase 1 — Audio Pipeline
pydub==0.25.1
noisereduce==3.0.2
webrtcvad==2.0.10
numpy==1.24.3

# Phase 2 — Speech to Text
openai-whisper==20231117
whisper-timestamped==1.14.2
torch==2.1.0
indic-transliteration==2.3.57

# Phase 3 — Analysis
transformers==4.35.0
scipy==1.11.3

# Phase 4 — Decision Engine
# (pure Python — no extra libraries)

# Phase 5 — Dashboard + Alerts
streamlit==1.28.0
twilio==8.10.0
pygame==2.5.2

# Phase 6 — Reports
reportlab==4.0.7
matplotlib==3.8.0

# Database
# sqlite3 — built into Python

# Utilities
python-dotenv==1.0.0
tqdm==4.66.1


#J1AEB9C12U8J9QNTR92U7A8A