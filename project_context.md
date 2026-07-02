# PROJECT_CONTEXT.md
## VoiceGuard — Real-Time Voice Surveillance System
**BCA Final Year Project | Poornima University, Jaipur**
**Guide: Mr. Hemant Gautam | Batch 2024–25**
**Team: Nishant Rakhecha [13411] · Vaishnavi Soni [13554] · Manyata Gupta [14048]**

---

## 1. Project Overview

VoiceGuard is a **real-time Hindi/Hinglish audio surveillance system** that listens to recorded audio (CCTV mic, phone recording, etc.), transcribes it, detects whether an emergency is occurring, and alerts authorities via SMS, email, and sound alarm — all with a live web dashboard for human-in-the-loop confirmation.

### What problem does it solve?

Security personnel cannot monitor audio from dozens of cameras simultaneously. VoiceGuard automates the first-pass detection: it flags audio segments where someone is shouting for help, reporting a fire, being attacked, etc. — specifically in **Hindi, English, and Hinglish** (code-switched speech common in Rajasthan/urban India).

### What it is NOT

- Not a live streaming system — it processes pre-recorded `.wav`/`.mp3` files in batch
- Not a speaker identification system (diarization is optional and often falls back to `UNKNOWN`)
- Not yet a production-deployed API — it runs locally via `streamlit` and `python main.py`

### Core output

For every 5-second audio chunk, the system produces a **final emergency score (0.0–1.0)** and assigns a zone:
- **GREEN** (`< 0.45`): safe, no action
- **YELLOW** (`0.45–0.72`): suspicious, monitor
- **RED** (`≥ 0.72`): emergency, alert authorities

---

## 2. End-to-End Flow (Step-by-Step Pipeline)

### Entry point

```
python main.py [audio_file.wav] [--model tiny|small|medium]
```

Or test each phase individually:
```
python test_phase1.py  →  test_phase2.py  →  test_phase3.py
→  test_phase4.py  →  test_phase5.py  →  test_phase6.py
```

Dashboard (standalone, reads Phase 4 output):
```
streamlit run pipeline/phase5_alerts/dashboard.py
```

---

### Phase 1 — Audio Preprocessing (`pipeline/phase1_audio/`)

**Input:** Raw audio file (any format: `.wav`, `.mp3`, `.m4a`, `.ogg`)

**Step 1.1 — Preprocessor** (`preprocessor.py` → `preprocess()`)
- Loads audio via `pydub.AudioSegment.from_file()`
- Converts to **mono** (stereo → single channel)
- Resamples to **16,000 Hz** (Whisper's required sample rate)
- Sets **16-bit depth** (WebRTC VAD requirement)
- Normalizes volume to **-20.0 dBFS** (consistent loudness across different mic types)
- Saves to `output/chunks/preprocessed.wav`

**Step 1.2 — Noise Reducer** (`noise_reducer.py` → `process_noise_reduction()`)
- Converts `pydub.AudioSegment` → `numpy float32` array (normalized to -1.0/1.0)
- Runs `noisereduce.reduce_noise()` with `stationary=True` (assumes consistent background like fan/AC hum)
- Default `prop_decrease=0.75` (75% noise reduction — balanced, not aggressive)
- Converts back to `pydub.AudioSegment` and saves to `output/chunks/noise_reduced.wav`

**Step 1.3 — VAD Chunker** (`vad_chunker.py` → `process_vad_chunking()`)
- Reads raw WAV bytes using Python's built-in `wave` module
- Generates **30ms frames** (WebRTC VAD only accepts 10/20/30ms)
  - Frame size = `16000 * 0.030 * 2 = 960 bytes`
- Runs `webrtcvad.Vad(aggressiveness=2)` frame-by-frame
- Groups consecutive speech frames using a **ring buffer of 10 frames (300ms)**:
  - `> 90%` speech frames → start segment
  - `> 90%` silence frames → end segment
- Applies **sliding window** on top of VAD segments:
  - Window = 5 seconds, Overlap = 2 seconds, Step = 3 seconds
  - Prevents words at chunk boundaries from being silently dropped
  - Long segments (> 5s) are sub-divided; short ones kept as-is
- Each chunk saved as `output/chunks/chunk_NNN.wav`
- Metadata saved to `output/chunks/metadata.json` (array of `{chunk_id, file_path, start, end, duration}`)

**Step 1.4 — Diarizer** (`diarizer.py` → `process_diarization()`) [OPTIONAL]
- Requires HuggingFace token + accepting pyannote model terms
- Runs `pyannote/speaker-diarization-3.1` on the full preprocessed audio
- Assigns each chunk its **dominant speaker** based on time-overlap calculation
- Saved to `output/transcripts/diarization.json`
- If unavailable: all chunks get `speaker_id = "UNKNOWN"` (pipeline continues)

**Output:** `output/chunks/metadata.json` + `chunk_NNN.wav` files

---

### Phase 2 — Speech-to-Text (`pipeline/phase2_stt/`)

**Input:** `output/chunks/metadata.json` + WAV files

**Step 2.1 — Whisper Transcription** (`whisper_transcriber.py` → `transcribe_all_chunks()`)
- Loads `faster-whisper` model once (default: `tiny`, single instance cached in `_model` global)
- For each chunk, calls `model.transcribe()` with:
  - `language="en"` — **always English mode**, even for Hindi audio
    - *Why:* Hindi speech gets phonetically transcribed to Roman Hinglish ("bachao" not "बचाओ")
    - This is intentional — downstream keyword matching requires Roman script
  - `beam_size=3` (reduced from default 5 for CPU speed)
  - `word_timestamps=True` (per-word start/end times)
  - `vad_filter=True` (internal Whisper VAD as second pass)
  - `condition_on_previous_text=False` (prevents hallucination bleeding between chunks)
- Language detected **free from the same pass** via `info.language` and `info.language_probability`
  - `classify_language_mix()`: Hindi (hi > 0.60) / English (en > 0.60) / Hinglish (anything else)
- Hallucination filter (`is_hallucination()`): removes "thank you for watching", CJK characters, excessive word repetition, replacement characters (`\ufffd`)
- Per-word confidence filter: words with `probability < 0.40` are dropped
- Output per chunk: `{chunk_id, chunk_start, text, words[], avg_confidence, language_mix, speaker_id}`

**Step 2.2 — Transliterator** (`transliterator.py` → `transliterate_all_transcripts()`)
- Detects Devanagari via Unicode range check (`\u0900–\u097F`)
- If found, uses `indic_transliteration` library (ITRANS scheme → cleaned Roman)
- Fallback: character-by-character map covering 50+ Devanagari chars
- In practice: rarely triggered because `language="en"` in Whisper already produces Roman output

**Step 2.3 — Keyword Normalizer** (`keyword_normalizer.py` → `apply_keyword_normalization_all()`)
- Dictionary of **90+ emergency terms** in `EMERGENCY_KEYWORDS`:
  - Format: `"keyword": ("CATEGORY", boost_value)`
  - Categories: HELP, MEDICAL, FIRE, VIOLENCE, ACCIDENT, THEFT, MENTAL
  - Boosts range from `0.10` (low-signal words like "dil") to `0.35` (high-signal like "goli", "suicide")
- Longest-match greedy scanning — "ambulance bulao" (boost 0.32) matched before "ambulance" (boost 0.30)
- Position tracking via `used_positions: set()` prevents double-counting overlapping matches
- **Total boost capped at 0.40** to prevent keyword flooding overriding ML models
- Adds `keyword_analysis: {keywords_found, categories_found, total_boost, top_category}` to transcript

**Output:** `output/transcripts/all_transcripts_final.json`

---

### Phase 3 — Analysis Models (`pipeline/phase3_analysis/`)

**Input:** `output/transcripts/all_transcripts_final.json`

**Step 3.1 — Text Emotion Detection** (`emotion_detector.py` → `detect_emotion()`)
- Model: `j-hartmann/emotion-english-distilroberta-base` (~500 MB, cached after first load)
- 7-class output: anger, disgust, fear, joy, neutral, sadness, surprise
- Returns `{dominant_emotion, dominant_score, all_scores{}, emergency_weight, fear_score, anger_score}`
- Emergency weights mapped: fear=1.0, anger=0.7, sadness=0.5, surprise=0.3, disgust=0.2, joy/neutral=0.0
- Text truncated to 512 tokens before inference

**Step 3.2 — Sarcasm Detection** (`emotion_detector.py` → `detect_sarcasm()`)
- Model: `cardiffnlp/twitter-roberta-base-irony` (~400 MB)
- Binary: `LABEL_0` = non-irony, `LABEL_1` = irony
- Returns `{is_sarcastic, sarcasm_score, confidence}` (confidence tiers: high > 0.75, medium > 0.55)

**Step 3.3 — Audio Emotion Detection** (`audio_emotion.py` → `detect_audio_emotion()`)
- Model: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (~1.2 GB)
- Input: raw WAV file path (chunk from Phase 1)
- HuggingFace `audio-classification` pipeline handles audio loading internally
- 8 raw labels → normalized to 7 standard emotions via `AUDIO_EMOTION_LABEL_MAP`
- Scores accumulated (multiple raw labels may map to same standard emotion), then re-normalized to sum=1.0
- Optional: `extract_voice_features()` extracts pitch_mean, pitch_std, energy_mean, energy_std, zcr_mean, speech_rate via `librosa` — adds interpretation hints but not yet used in scoring formula

**Step 3.4 — Emergency Detection** (`emergency_detector.py` → `detect_emergency()`)
- Model: `facebook/bart-large-mnli` (~1.6 GB, zero-shot classification)
- **7 candidate labels** (plain English descriptions, not short names):
  ```
  "medical emergency, someone needs a doctor or ambulance"
  "fire emergency, there is a fire or smoke"
  "violence or physical assault, someone is being attacked or hurt"
  "accident or injury, someone has fallen or been injured"
  "theft or robbery, someone is stealing or being robbed"
  "mental health crisis, someone wants to harm themselves"
  "normal conversation, no emergency"
  ```
- `multi_label=False` → single best category per chunk
- **Keyword boost applied after BART inference** (in `analyzer.py`):
  - Phase 2's `total_boost` added to BART's matched category score
  - Top category re-evaluated after boost
  - This is the integration point between rule-based (keyword) and ML (BART)
- **Word count gate:** `MIN_WORDS_FOR_EMERGENCY = 4` — BART skipped entirely for fragments
- **Threshold in `analyzer.py`:** `EMERGENCY_THRESHOLD = 0.55` (raised from 0.35 in `emergency_detector.py` — see Known Issues §9)

**Step 3.5 — Emotion Fusion** (`emotion_fusion.py` → `fuse_emotions()`)
- Combines text and audio emotion scores with context-aware weights:
  | Condition | Audio weight | Text weight |
  |---|---|---|
  | Audio fear > 0.45 ("audio fear override") | 0.80 | 0.20 |
  | Text and audio agree on emotion | 0.50 | 0.50 |
  | Text and audio disagree | 0.70 | 0.30 |
- After weighted combination, scores re-normalized to sum=1.0
- Records `fusion_method` field for dashboard display and debugging

**Step 3.6 — Sarcasm Resolution** (`emotion_fusion.py` → `resolve_sarcasm_with_audio()`)
- Audio-enhanced resolution overrides text-only conflict matrix when audio available:
  - Sarcastic text + audio fear > 0.50 → override sarcasm, treat as real (`score_penalty = 0.0`)
  - Text fear > 0.60 + audio joy > 0.40 or neutral > 0.60 → confirmed sarcasm (`score_penalty = 0.60`)
  - Both text and audio show fear > 0.40 → definitively real (`score_penalty = 0.0`)
- Falls back to text-only matrix (`resolve_sarcasm_conflict()` in `emotion_detector.py`) when audio unavailable

**Step 3.7 — Inline scoring in analyzer.py** (`analyzer.py` → `compute_combined_score()`)
- Note: Phase 3 computes its own preliminary score. Phase 4 re-computes with `scorer.py`.
- The two implementations are nearly identical but not shared (see Known Issues §9)
- `apply_trend_analysis()` also called inside `analyzer.py` — runs a second time in Phase 4

**Output:** `output/analysis/all_analysis.json` (each chunk has `emotion_analysis{}`, `emergency_analysis{}`, `score{}`)

---

### Phase 4 — Decision Engine (`pipeline/phase4_decision/`)

**Input:** `output/analysis/all_analysis.json`

**Step 4.1 — Scorer** (`scorer.py` → `score_all_chunks()`)
- Reads `emotion_analysis.emotion{}` (fused emotion), `emergency_analysis{}`, `keyword_analysis{}`
- Formula:
  ```
  emotion_component   = emergency_weight × dominant_score
  emergency_component = top_score  (if is_emergency else top_score × 0.3)
  keyword_component   = min(total_boost, 1.0)

  base_score = (emotion_component × 0.35) +
               (emergency_component × 0.40) +
               (keyword_component × 0.25)

  sarcasm_deduction = base_score × score_penalty
  final_score = max(0.0, min(base_score − sarcasm_deduction, 1.0))
  ```

**Step 4.2 — Zone Classifier** (`zone_classifier.py` → `classify_all_zones()`)
- Assigns zone + severity + alert type per RED chunk:
  | Category | Severity | Auto-alert? |
  |---|---|---|
  | fire, violence | CRITICAL | Yes |
  | medical, accident, mental_health | HIGH | Yes |
  | theft | MEDIUM | No (needs human confirm) |
  | normal | LOW | No |

**Step 4.3 — Trend Analyzer** (`trend_analyzer.py` → `apply_trend_analysis()`)
- **Rule 1 — Consecutive YELLOW:** 3+ YELLOW in a row → 3rd chunk upgraded to RED, `trend_upgraded=True`, `auto_alert=True`
- **Rule 2 — Rapid rise:** Score increase ≥ 0.20 across any 3-chunk window → `rising_trend=True` (metadata only, zone unchanged)
- **Rule 3 — Incident grouping:** RED chunks separated by ≤ 2 GREEN/YELLOW chunks → same `incident_id` (e.g., `INC_001`)
  - YELLOW chunks within the gap extend the incident but reset the counter
  - `incident_counter` increments per new incident

**Output:** `output/decisions/all_decisions.json` ← **this is the single source of truth for all downstream consumers**

---

### Phase 5 — Alerts & Dashboard (`pipeline/phase5_alerts/`)

**Dashboard** (`dashboard.py`)
- Streamlit app, polls `all_decisions.json` every 3 seconds: `@st.cache_data(ttl=3)` + `time.sleep(3)` + `st.rerun()`
- **Deduplication:** `st.session_state.alerts_sent: set()` keyed on `incident_id` — prevents double-alerting for same incident
- **Auto-trigger countdown:** 60-second timer per RED incident. Elapsed tracked via `alert_start_time[iid] = time.time()`. `remaining = max(0, 60 − elapsed)`.
- **RED resolution flow:** CONFIRM → fires SMS + email, stops alarm, adds to `confirmed_incidents` list → chunk disappears from alert cards on next rerender
- **Audio playback:** looks for `output/chunks/{chunk_id}.wav` → serves via `st.audio()` if found

**SMS** (`sms_alert.py` → `send_emergency_sms()`)
- Twilio REST API
- Dry-run mode: `TWILIO_ENABLED = False` when credentials match placeholder strings
- Message built by `build_sms_message()` — includes category, severity, incident ID, score, first 60 chars of transcript

**Email** (`email_alert.py` → `send_emergency_email()`)
- Gmail SMTP (`smtp.gmail.com:587`) with STARTTLS
- Full HTML email with recent 5-chunk transcript history table
- Dry-run mode: same pattern as SMS — checks if `SMTP_EMAIL != "your_gmail@gmail.com"`

**Sound** (`sound_alert.py`)
- `pygame.mixer` with fallback sine-wave tone generation if `.wav` asset files missing
- YELLOW: single soft beep (880 Hz, 400ms, volume 0.4)
- RED: continuous alarm loop in daemon thread, stopped by `stop_alarm()` setting `_stop_alarm_flag` threading Event

---

### Phase 6 — Report Generation (`pipeline/phase6_reports/`)

**Incident Report** (`incident_report.py` → `generate_incident_report()`)
- ReportLab `SimpleDocTemplate` on A4
- Sections: red alert banner → incident summary table → confidence score with visual bar → flagged transcript (keywords highlighted with `<font color="#ef4444"><b>[WORD]</b></font>`) → 5-chunk context window table → alert actions table → footer note
- Emergency keywords highlighted via `_highlight_text()` — word-level scan against `HIGHLIGHT_KEYWORDS` list
- File: `output/reports/incident_INC_001_YYYYMMDD_HHMMSS.pdf`

**Session Report** (`session_report.py` → `generate_session_report()`)
- 4-page PDF: cover page with quick-stats → zone/emotion/category analysis → score timeline chart → incident table → alert log → full transcript log (last 30 chunks)
- Score timeline: matplotlib bar chart generated into `io.BytesIO()` PNG bytes, embedded directly in PDF via `reportlab.platypus.Image`
- File: `output/reports/session_report_YYYYMMDD_HHMMSS.pdf`

---

## 3. Folder Structure

```
voice_surveillance/
│
├── main.py                          # Full pipeline orchestrator (Phase 1→6 + dashboard launch)
├── config.py                        # Central config: paths, thresholds, model names, credentials
├── requirements.txt                 # All pip dependencies
├── test_environment.py              # Verify all libraries load correctly
│
├── test_phase1.py                   # Phase 1 standalone test
├── test_phase2.py                   # Phase 2 standalone test
├── test_phase3.py                   # Phase 3 standalone test
├── test_phase4.py                   # Phase 4 standalone test (injects fake RED chunk)
├── test_phase5.py                   # Phase 5 dry-run alert test
├── test_phase6.py                   # Phase 6 PDF generation test
│
├── pipeline/
│   ├── phase1_audio/
│   │   ├── __init__.py
│   │   ├── preprocessor.py          # Format conversion, mono, 16kHz, volume normalization
│   │   ├── noise_reducer.py         # noisereduce stationary noise removal
│   │   ├── vad_chunker.py           # WebRTC VAD + sliding window chunker
│   │   └── diarizer.py              # pyannote speaker diarization (optional)
│   │
│   ├── phase2_stt/
│   │   ├── __init__.py
│   │   ├── whisper_transcriber.py   # faster-whisper STT, hallucination filter
│   │   ├── transliterator.py        # Devanagari → Roman script conversion
│   │   └── keyword_normalizer.py    # Emergency keyword dictionary + scoring boosts
│   │
│   ├── phase3_analysis/
│   │   ├── __init__.py
│   │   ├── emotion_detector.py      # Text emotion (distilRoBERTa) + sarcasm (RoBERTa-irony)
│   │   ├── audio_emotion.py         # Audio emotion (wav2vec2) + librosa voice features
│   │   ├── emotion_fusion.py        # Dual-channel fusion rules + audio sarcasm override
│   │   ├── emergency_detector.py    # BART zero-shot classification (7 emergency labels)
│   │   ├── analyzer.py              # Orchestrates Phase 3, also scores/trends (see Known Issues)
│   │   └── sarcasm_detector.py      # EMPTY FILE — functionality is in emotion_detector.py
│   │
│   ├── phase4_decision/
│   │   ├── __init__.py              # Re-exports scorer, classifier, trend analyzer
│   │   ├── scorer.py                # Weighted formula: emotion×0.35 + emergency×0.40 + keyword×0.25
│   │   ├── zone_classifier.py       # GREEN/YELLOW/RED thresholds + severity + auto-alert flags
│   │   └── trend_analyzer.py        # 3-rule trend detection + incident ID grouping
│   │
│   ├── phase5_alerts/
│   │   ├── __init__.py
│   │   ├── dashboard.py             # Streamlit UI (full dashboard — 700+ lines)
│   │   ├── sms_alert.py             # Twilio SMS with dry-run fallback
│   │   ├── email_alert.py           # Gmail SMTP HTML email with dry-run fallback
│   │   └── sound_alert.py           # pygame alarm with generated-tone fallback
│   │
│   └── phase6_reports/
│       ├── __init__.py
│       ├── incident_report.py       # Single-incident PDF (ReportLab)
│       └── session_report.py        # Full session summary PDF (ReportLab + matplotlib)
│
├── input/
│   ├── audio/                       # Place raw audio files here (Test_Normal.wav etc.)
│   └── datasets/                    # [gitignored] training/test datasets
│
├── output/                          # [gitignored] all generated files
│   ├── chunks/                      # preprocessed.wav, noise_reduced.wav, chunk_NNN.wav, metadata.json
│   ├── transcripts/                 # all_transcripts_final.json, per-chunk JSONs
│   ├── analysis/                    # all_analysis.json, per-chunk analysis JSONs
│   ├── decisions/                   # all_decisions.json ← dashboard reads this
│   ├── reports/                     # Generated PDFs
│   └── pipeline.log                 # Appended each run
│
├── database/
│   └── incidents.db                 # SQLite (declared in config.py, not yet written to)
│
└── assets/
    ├── beep.wav                     # YELLOW zone sound (fallback: generated 880Hz tone)
    └── alarm.wav                    # RED zone alarm (fallback: generated 1200Hz tone)
```

---

## 4. ML Models Used and Where

| Model | HuggingFace ID | Size | Phase | File | Purpose |
|---|---|---|---|---|---|
| faster-whisper | (distil-whisper or standard) | 39MB–769MB | Phase 2 | `whisper_transcriber.py` | Hindi/Hinglish → Roman text |
| Text emotion | `j-hartmann/emotion-english-distilroberta-base` | ~500 MB | Phase 3 | `emotion_detector.py` | 7-class emotion from text |
| Sarcasm/irony | `cardiffnlp/twitter-roberta-base-irony` | ~400 MB | Phase 3 | `emotion_detector.py` | Binary: irony vs non-irony |
| Audio emotion | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | ~1.2 GB | Phase 3 | `audio_emotion.py` | 8-class emotion from voice |
| Emergency | `facebook/bart-large-mnli` | ~1.6 GB | Phase 3 | `emergency_detector.py` | Zero-shot: 7 emergency types |
| Speaker diarization | `pyannote/speaker-diarization-3.1` | ~1 GB | Phase 1 | `diarizer.py` | Speaker attribution (optional) |

**Total RAM requirement (all models loaded):** ~4.7 GB minimum (without diarization)

All models run on **CPU only** (`device=-1` in HuggingFace pipelines). No GPU required. Models are loaded once and cached in module-level globals (`_emotion_pipeline`, `_emergency_pipeline`, etc.) — reloading the module re-uses the cached instance.

**Model loading order in `analyzer.py`:**
```python
get_emotion_model()    # loads distilRoBERTa
get_sarcasm_model()    # loads RoBERTa-irony
get_emergency_model()  # loads BART
get_audio_emotion_model()  # loads wav2vec2 (if audio_mode=True)
```
All four loaded before the chunk loop to avoid mid-processing delays.

---

## 5. Feature Engineering (Detailed)

### 5.1 Audio features (Phase 1)

| Feature | How computed | Purpose |
|---|---|---|
| Mono conversion | `audio.set_channels(1)` | Whisper + VAD require mono |
| 16kHz resampling | `audio.set_frame_rate(16000)` | Whisper training sample rate |
| Volume normalization | `audio.apply_gain(target_dBFS − current_dBFS)`, target = -20.0 dBFS | Equalize across different mic types |
| VAD speech frames | `webrtcvad.Vad(2).is_speech(frame_bytes, 16000)` | Binary speech/silence per 30ms |
| Sliding window chunks | 5s window, 2s overlap → 3s step | Sentence-safe boundaries |

### 5.2 Text features (Phase 2)

| Feature | How computed | Where stored |
|---|---|---|
| Transcript text | `faster-whisper` → `segment.text` joined | `transcript["text"]` |
| Per-word timestamps | `word.start`, `word.end`, `word.probability` | `transcript["words"][]` |
| Word confidence filter | Drop words where `probability < 0.40` | Reduces noise in `avg_confidence` |
| Language mix | `classify_language_mix(info.language, info.language_probability)` | `transcript["language_mix"]` |
| Keyword matches | Greedy longest-match over `EMERGENCY_KEYWORDS` dict | `keyword_analysis["keywords_found"]` |
| Keyword boost | `sum(match.boost for match in keywords_found)`, capped at 0.40 | `keyword_analysis["total_boost"]` |
| Keyword category | Most frequent category across matched keywords | `keyword_analysis["top_category"]` |

### 5.3 Emotion features (Phase 3, text channel)

| Feature | How computed | Where stored |
|---|---|---|
| 7-class emotion scores | `j-hartmann` model → `{label: score}` dict | `emotion_analysis.text_emotion.all_scores` |
| Dominant emotion | `max(all_scores, key=all_scores.get)` | `text_emotion.dominant_emotion` |
| Emergency weight | Lookup table: fear=1.0, anger=0.7, sadness=0.5, etc. | `text_emotion.emergency_weight` |
| Sarcasm score | `cardiffnlp` model → `irony` label probability | `sarcasm.sarcasm_score` |
| Sarcasm confidence | High > 0.75, medium > 0.55, low otherwise | `sarcasm.confidence` |

### 5.4 Emotion features (Phase 3, audio channel)

| Feature | How computed | Where stored |
|---|---|---|
| 8-class audio emotion | `wav2vec2` pipeline on raw WAV → normalized to 7 labels | `audio_emotion.all_scores` |
| Audio fear score | `all_scores.get("fear", 0.0)` | `audio_emotion.fear_score` |
| Fusion method | Rule-based: agree/disagree/fear-override | `fused_emotion.fusion_method` |
| Fused emotion | Weighted average of text and audio scores, re-normalized | `fused_emotion.all_scores` |
| Voice features (unused) | `librosa`: pitch_mean, pitch_std, energy_mean, zcr_mean, speech_rate | `voice_features{}` (logged only) |

### 5.5 Emergency features (Phase 3)

| Feature | How computed | Where stored |
|---|---|---|
| 7-class emergency scores | BART zero-shot over 7 label descriptions | `emergency_analysis.all_scores` |
| Is emergency | `top_category != "normal" AND top_score >= 0.55` | `emergency_analysis.is_emergency` |
| Post-keyword BART score | `all_scores[kw_category] += total_boost`, then re-evaluate top | `emergency_analysis.top_category` |
| Risk level | Category lookup: fire=0.95, violence=0.85, medical=0.90, etc. | `emergency_analysis.risk_level` |

### 5.6 Decision features (Phase 4)

| Feature | How computed | Where stored |
|---|---|---|
| Emotion component | `emergency_weight × dominant_score × 0.35` | `score.components.emotion_component` |
| Emergency component | `top_score × 0.40` (or `× 0.12` if not is_emergency) | `score.components.emergency_component` |
| Keyword component | `min(total_boost, 1.0) × 0.25` | `score.components.keyword_component` |
| Sarcasm deduction | `base_score × score_penalty` | `score.components.sarcasm_deduction` |
| Final score | `max(0, base_score − deduction)`, capped at 1.0 | `score.final_score` |
| Zone | Threshold on `final_score` | `score.zone` |
| Severity | `CATEGORY_SEVERITY[emergency_category]` only if zone=RED | `score.severity` |
| Auto-alert | `zone == RED AND severity in {CRITICAL, HIGH}` | `score.auto_alert` |
| Requires confirm | `zone == RED AND severity == MEDIUM` | `score.requires_confirm` |
| Trend upgraded | 3 consecutive YELLOW → RED | `score.trend_upgraded` |
| Rising trend | Score rise ≥ 0.20 across 3 chunks | `score.rising_trend` |
| Incident ID | Grouping rule: gap ≤ 2 GREEN chunks | `score.incident_id` |

---

## 6. API Endpoints and Their Purpose

**VoiceGuard does not expose a REST API.** There are no HTTP endpoints. The system is a local batch pipeline + Streamlit dashboard.

However, the following **external APIs** are consumed:

### HuggingFace Inference (local, not cloud)
All models run locally after download. No API key required for inference. First run downloads models to HuggingFace cache (`~/.cache/huggingface/hub/`).

### Twilio SMS API
- **Endpoint:** `https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages.json`
- **Auth:** Account SID + Auth Token (Basic Auth)
- **Called from:** `sms_alert.py` → `send_sms()` → `client.messages.create()`
- **Dry-run mode:** `TWILIO_ENABLED = False` when `TWILIO_ACCOUNT_SID == "your_account_sid_here"`
- **Credentials source:** Environment variables `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `ALERT_NUMBER_1`

### Gmail SMTP
- **Host:** `smtp.gmail.com:587` with STARTTLS
- **Auth:** App Password (not regular Gmail password)
- **Called from:** `email_alert.py` → `send_email()` → `smtplib.SMTP`
- **Dry-run mode:** `EMAIL_ENABLED = False` when credentials match placeholders
- **Credentials source:** Environment variables `SMTP_EMAIL`, `SMTP_PASSWORD`, `ALERT_EMAIL_1`

### pyannote/speaker-diarization-3.1 (HuggingFace model API gated)
- **Auth:** HuggingFace token required (user must accept model terms at huggingface.co/pyannote)
- **Called from:** `diarizer.py` → `Pipeline.from_pretrained(..., use_auth_token=hf_token)`
- **Graceful degradation:** Returns `[]` if token missing, pipeline continues with `UNKNOWN` speakers

---

## 7. Database Interaction Flow

### Current state: NOT IMPLEMENTED

`config.py` declares:
```python
DATABASE_PATH = "database/incidents.db"
```

No other file in the reviewed codebase reads or writes to this path. The SQLite database is **planned but not yet integrated**. There are no `sqlite3` imports, no schema definitions, no INSERT/SELECT statements anywhere in the pipeline code.

### What the database should hold (based on system design)

Based on what the system currently stores in JSON and session state, the intended schema likely covers:

**`incidents` table** (currently in `all_decisions.json` + `st.session_state.alert_log`)
```sql
CREATE TABLE incidents (
    incident_id   TEXT PRIMARY KEY,   -- e.g. INC_001
    chunk_id      TEXT,
    timestamp     DATETIME,
    category      TEXT,               -- medical, fire, violence, etc.
    severity      TEXT,               -- CRITICAL, HIGH, MEDIUM, LOW
    zone          TEXT,               -- RED, YELLOW, GREEN
    final_score   REAL,
    emotion       TEXT,
    transcript    TEXT,
    action        TEXT,               -- CONFIRMED, DISMISSED, AUTO
    sms_sent      INTEGER,            -- boolean
    email_sent    INTEGER
);
```

**`chunks` table** (currently per-session JSON files)
```sql
CREATE TABLE chunks (
    chunk_id      TEXT PRIMARY KEY,
    session_id    TEXT,
    chunk_start   REAL,
    text          TEXT,
    final_score   REAL,
    zone          TEXT,
    incident_id   TEXT REFERENCES incidents(incident_id)
);
```

### Assumption
The database was planned for **cross-session incident history** — the current implementation loses all alert history when the Streamlit session restarts or `st.session_state` is cleared via the "Clear Alert Log" button.

---

## 8. Current Status

### Completed ✅

| Component | Status | Notes |
|---|---|---|
| Phase 1 — Audio pipeline | Complete | All 3 steps working; diarizer optional/functional |
| Phase 2 — Whisper STT | Complete | tiny/small/medium model sizes, hallucinaton filter working |
| Phase 2 — Transliterator | Complete | Rarely needed in practice (language=en mode) |
| Phase 2 — Keyword normalizer | Complete | 90+ keywords, boost system, longest-match |
| Phase 3 — Text emotion | Complete | distilRoBERTa, 7 emotions, emergency weights |
| Phase 3 — Sarcasm detection | Complete | cardiffnlp/twitter-roberta-base-irony |
| Phase 3 — Audio emotion | Complete | wav2vec2, dual-channel fusion implemented |
| Phase 3 — Emergency detection | Complete | BART zero-shot, 7 labels, keyword post-boost |
| Phase 3 — Emotion fusion | Complete | Rule-based, audio-wins-on-disagree, fear override |
| Phase 4 — Scorer | Complete | Weighted formula, sarcasm penalty |
| Phase 4 — Zone classifier | Complete | GREEN/YELLOW/RED, severity, auto-alert logic |
| Phase 4 — Trend analyzer | Complete | All 3 rules: consecutive YELLOW, rapid rise, incident grouping |
| Phase 5 — Dashboard | Complete | Full Streamlit UI, 7 sections, real-time polling |
| Phase 5 — SMS alert | Complete | Twilio + dry-run |
| Phase 5 — Email alert | Complete | Gmail SMTP + HTML body + dry-run |
| Phase 5 — Sound alert | Complete | pygame + generated-tone fallback |
| Phase 6 — Incident PDF | Complete | Full ReportLab PDF with all sections |
| Phase 6 — Session PDF | Complete | matplotlib chart + full transcript log |
| main.py orchestrator | Complete | All 6 phases + skip flags + dashboard auto-launch |
| config.py | Complete | All paths and thresholds centralized |

### Incomplete / Not Implemented ❌

| Component | Status | Notes |
|---|---|---|
| SQLite database | Not implemented | Path declared in config.py, never written to |
| `sarcasm_detector.py` | Empty file | Functionality lives in `emotion_detector.py` |
| Voice features in scoring | Not integrated | `librosa` features extracted but not used in any formula |
| Live audio streaming | Not implemented | System only processes pre-recorded files |
| Multi-file / session-aware pipeline | Not implemented | Each `main.py` run overwrites output JSONs |
| Cross-session incident history | Not implemented | `st.session_state` resets on refresh |
| User authentication on dashboard | Not implemented | Dashboard is open/unauthenticated |
| `pyannote` diarization setup | Partially implemented | Code complete but requires HuggingFace gated model access |
| GPU support | Not implemented | All models forced to CPU (`device=-1`) |
| Unit tests | Not implemented | Only integration-level test scripts exist |

---

## 9. Known Issues and Weak Points

### Issue 1 — EMERGENCY_THRESHOLD mismatch (HIGH SEVERITY)
**Location:** `emergency_detector.py` line ~103 vs `analyzer.py` line ~28

`emergency_detector.py` uses `EMERGENCY_THRESHOLD = 0.35` but `analyzer.py` overrides with `EMERGENCY_THRESHOLD = 0.55`.

When `test_phase3.py` runs, it calls `run_phase3()` from `analyzer.py` → uses 0.55 threshold.
When `main.py` runs Phase 3, it also calls `analyzer.py` → also 0.55.
But if someone calls `detect_emergency()` from `emergency_detector.py` directly → uses 0.35.

**Impact:** Inconsistent behavior depending on call path. A score of 0.40 would be classified as emergency in `emergency_detector.py` but not in `analyzer.py`.

**Fix:** Remove `EMERGENCY_THRESHOLD` from `emergency_detector.py`, define it only in `config.py`, import it everywhere.

---

### Issue 2 — Duplicate scoring implementations (MEDIUM SEVERITY)
**Location:** `analyzer.py → compute_combined_score()` vs `scorer.py → compute_score()`

Both compute nearly the same formula. `analyzer.py` version reads `emotion_analysis.fused_emotion` while `scorer.py` reads `emotion_analysis.emotion` (which is set to fused_emotion as a backward-compat alias in `analyzer.py`). `apply_trend_analysis()` also runs in both `analyzer.py` (line ~167) and `trend_analyzer.py` called from Phase 4.

**Impact:** Score computed twice per pipeline run. If the two implementations drift, test scores and production scores will diverge silently.

**Fix:** Have `analyzer.py` not score at all — just attach `emotion_analysis` and `emergency_analysis`. Leave all scoring to Phase 4.

---

### Issue 3 — Empty `sarcasm_detector.py` (LOW SEVERITY)
**Location:** `pipeline/phase3_analysis/sarcasm_detector.py`

File is empty. The import in other files would fail if anyone imports from it directly. Sarcasm logic actually lives in `emotion_detector.py`.

**Fix:** Either move the sarcasm functions into `sarcasm_detector.py` (and update imports) or delete the file and document the decision.

---

### Issue 4 — Voice features extracted but never used (LOW SEVERITY)
**Location:** `audio_emotion.py → extract_voice_features()` called in `analyze_audio_emotion()`

Pitch mean/std, energy, ZCR, speech rate are computed and stored in `voice_features{}` in the JSON. The scoring formula in neither `analyzer.py` nor `scorer.py` reads from `voice_features`. The interpretation hints are only logged.

**Impact:** ~librosa computation time per chunk wasted. Can add meaningful signal (shouting detected via energy_mean > 0.08 is a strong violence indicator).

**Assumption:** Planned for a future scoring component but not yet integrated.

---

### Issue 5 — `config.py` constants not imported by modules (MEDIUM SEVERITY)
**Location:** Multiple files

`config.py` defines `WEIGHT_EMOTION = 0.35`, `ZONE_RED = 0.72`, etc. But `scorer.py` and `zone_classifier.py` re-declare these constants locally rather than importing from config. If someone changes `config.py`, the scoring and zoning behavior won't change.

**Impact:** Config is not the single source of truth it appears to be.

**Fix:** Add `from config import WEIGHT_EMOTION, WEIGHT_EMERGENCY, WEIGHT_KEYWORD, ZONE_YELLOW, ZONE_RED` to `scorer.py` and `zone_classifier.py`. Same for `analyzer.py`.

---

### Issue 6 — Dashboard session state lost on refresh (MEDIUM SEVERITY)
**Location:** `dashboard.py`

`st.session_state.alert_log`, `confirmed_incidents`, `dismissed_incidents`, `alerts_sent` are all in-memory only. Refreshing the browser or restarting Streamlit wipes all history. A dismissed false alarm will reappear as a RED alert on next refresh.

**Impact:** Security personnel who dismiss a false alarm and then refresh the page will see it re-appear, potentially causing confusion or duplicate SMS alerts (if `alerts_sent` set is also wiped and the same `incident_id` gets alerted twice).

**Fix:** Persist `alert_log` and `confirmed_incidents` to SQLite after each action. This is exactly what the unimplemented `database/incidents.db` was intended for.

---

### Issue 7 — all_decisions.json overwritten on each run (MEDIUM SEVERITY)
**Location:** `main.py → run_phase4()`, `test_phase4.py`

Each pipeline run writes to the same `output/decisions/all_decisions.json`. There is no session ID or timestamp in the filename. Running `main.py` twice on different audio files overwrites the previous session's data.

**Impact:** No multi-session history. The dashboard always shows only the last-run audio file's results.

**Fix:** Write to `output/decisions/session_{timestamp}.json` and have dashboard let user pick session, or append to SQLite.

---

### Issue 8 — Fake RED chunk injection in test_phase4.py (LOW SEVERITY)
**Location:** `test_phase4.py → inject_fake_red()`

`inject_fake_red()` prepends a hardcoded fake emergency chunk to ensure the RED alert flow is testable without real emergency audio. This is fine for testing but the comment in the file says "Remove this block when testing with real emergency audio" — easy to forget.

**Impact:** If `test_phase4.py` is mistakenly used to generate `all_decisions.json` for the dashboard demo, there will always be one fake RED incident (`chunk_FAKE_RED`) present.

---

### Issue 9 — librosa is optional but not guarded uniformly (LOW SEVERITY)
**Location:** `audio_emotion.py → extract_voice_features()`

`librosa` import is inside a try/except in `extract_voice_features()` but not in `analyze_audio_emotion()` which calls it. If librosa is missing, `extract_voice_features()` returns `{}` silently — which is fine. But `librosa` is listed without a version pin in `requirements.txt` (`librosa` on its own line at the bottom), which may pull in incompatible versions.

---

### Issue 10 — No rate limiting or error recovery between chunks (LOW SEVERITY)
**Location:** `analyzer.py → run_phase3()` chunk loop

If BART inference fails on chunk N, the exception is caught and the chunk gets a default `normal` emergency result — but processing continues. However, if `get_emotion_model()` fails mid-batch (e.g., OOM), the entire Phase 3 crashes and partial results are not saved.

**Fix:** Save intermediate results to disk inside the chunk loop (or at least every 10 chunks) so a crash doesn't require reprocessing everything.

---

## 10. Next Actionable Steps (Priority Order)

### Priority 1 — Critical fixes (do before any demo)

**1.1 — Unify `EMERGENCY_THRESHOLD`**
- Delete `EMERGENCY_THRESHOLD` from `emergency_detector.py`
- Add it to `config.py` as `EMERGENCY_THRESHOLD = 0.55`
- Import in both `emergency_detector.py` and `analyzer.py`
- Run `test_phase3.py` to verify no regressions

**1.2 — Import scoring constants from `config.py`**
- In `scorer.py`: replace local `WEIGHT_EMOTION = 0.35` etc. with `from config import WEIGHT_EMOTION, ...`
- In `zone_classifier.py`: replace `ZONE_GREEN_MAX = 0.45`, `ZONE_YELLOW_MAX = 0.72` with imports
- This makes `config.py` actually work as advertised

**1.3 — Persist alert log to SQLite**
- Implement the `incidents` table schema in `database/incidents.db`
- Write to it on every CONFIRM/DISMISS/AUTO-ALERT action in `dashboard.py`
- Read from it on dashboard startup to restore alert history after refresh

---

### Priority 2 — Architecture cleanup (before adding new features)

**2.1 — Remove duplicate scoring from `analyzer.py`**
- Delete `compute_combined_score()` and `apply_trend_analysis()` from `analyzer.py`
- Phase 3 should only attach `emotion_analysis` and `emergency_analysis` to each chunk
- All scoring → Phase 4 exclusively

**2.2 — Integrate voice features into scoring**
- In `scorer.py`, add a 4th component: `voice_component`
- `energy_mean > 0.08` → `+0.10` shouting bonus
- `pitch_std > 40 and pitch_mean > 220` → `+0.05` stress bonus
- Re-balance weights: e.g., emotion×0.30, emergency×0.35, keyword×0.20, voice×0.15
- This uses already-computed data that is currently wasted

**2.3 — Fix empty `sarcasm_detector.py`**
- Move `detect_sarcasm()` and `resolve_sarcasm_conflict()` from `emotion_detector.py` into it
- Update all imports

**2.4 — Session-aware output files**
- Change `output/decisions/all_decisions.json` → `output/decisions/session_{timestamp}.json`
- Dashboard shows a session picker (dropdown of available session files)

---

### Priority 3 — Feature additions (after codebase is stable)

**3.1 — Real-time audio streaming**
- Replace file-based input with a microphone stream using `sounddevice` or `pyaudio`
- Process 5-second rolling windows in a background thread
- Write chunks to `output/chunks/` as they arrive
- Dashboard's 3-second polling already supports this — no dashboard changes needed

**3.2 — GPU acceleration**
- Add `device="cuda"` flag gated by `torch.cuda.is_available()`
- Expected speedup: 5–10× per chunk on a mid-range GPU
- Particularly impactful for BART (largest model at 1.6GB)

**3.3 — Multi-camera / multi-session support**
- Add `session_id` (camera ID + timestamp) to all output files
- Dashboard left sidebar: session/camera selector
- SQLite becomes mandatory at this point

**3.4 — Dashboard export buttons**
- "Export Incident PDF" button for any RED row in the incident log
- Calls `generate_incident_report(chunk, all_chunks, action="CONFIRMED")` and serves the PDF as a Streamlit download
- The infrastructure is all in place — just wire up the button

**3.5 — Confidence calibration / re-scoring**
- Current keyword boosts (e.g., 0.30 for "bachao") are hand-tuned, not data-driven
- Collect labeled clips, run the pipeline, measure precision/recall per threshold
- Calibrate `ZONE_YELLOW` and `ZONE_RED` thresholds against real data

**3.6 — Pin library versions more carefully**
- `requirements.txt` has `librosa`, `torch`, `streamlit`, `pygame`, `reportlab`, `matplotlib` listed bare (no version) at the bottom — these conflict with pinned versions above them
- Consolidate into a single pinned entry per library, preferably with a `pip compile` / `pip-tools` workflow

---

*This document was generated from full codebase analysis. Last updated: 2026-04-01.*
*Assumptions are marked inline. Section 7 (Database) and Section 6 (API) contain design inferences where implementation is absent.*

