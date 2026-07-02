"""
Phase 5 — VoiceGuard Authority Dashboard (v3)
Light theme, single scrollable page, operational layout.

Run from voice_surveillance/ folder:
    streamlit run pipeline/phase5_alerts/dashboard.py

Layout (top to bottom):
    1. Header bar — system status + controls
    2. Stats row — 5 metric cards
    3. Emergency alert card (only when RED active)
    4. Live transcript feed
    5. Analytics row — keyword freq | emotion bars | score timeline
    6. Incident log table
"""

import os
import sys
import json
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.phase5_alerts.dashboard_data import load_chunks, data_source_label
from pipeline.phase5_alerts.sms_alert   import send_emergency_sms,  TWILIO_ENABLED
from pipeline.phase5_alerts.email_alert import send_emergency_email, EMAIL_ENABLED
from pipeline.phase5_alerts.sound_alert import trigger_for_zone, stop_alarm, is_alarm_active

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DECISIONS_PATH   = "output/decisions/all_decisions.json"
CHUNKS_DIR       = "output/chunks"
AUTO_TRIGGER_SEC = 60
AUTO_TRIGGER_MIN = 0.90

# ── PAGE SETUP ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "VoiceGuard — Security Dashboard",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #f7f8fa !important;
    color: #1a1f2e !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Remove default padding ── */
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #f7f8fa !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f2f5; }
::-webkit-scrollbar-thumb { background: #c8d0dc; border-radius: 3px; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
    border: none !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background: white !important;
    border: 1px solid #dde2ea !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    color: #1a1f2e !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: white !important;
    border: 1px solid #dde2ea !important;
    border-radius: 6px !important;
    font-size: 13px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #e8ecf0 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stMetricValue"] {
    color: #1a1f2e !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Divider ── */
hr { border-color: #e8ecf0 !important; margin: 4px 0 !important; }

/* ── Success / info / warning ── */
.stSuccess { background: #f0fdf4 !important; border-color: #86efac !important; color: #166534 !important; border-radius: 8px !important; }
.stInfo    { background: #eff6ff !important; border-color: #93c5fd !important; color: #1e40af !important; border-radius: 8px !important; }
.stWarning { background: #fffbeb !important; border-color: #fcd34d !important; color: #92400e !important; border-radius: 8px !important; }
.stError   { background: #fef2f2 !important; border-color: #fca5a5 !important; color: #991b1b !important; border-radius: 8px !important; }

/* ── Blink animation ── */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulse  { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4)} 70%{box-shadow:0 0 0 8px rgba(239,68,68,0)} }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for k, v in {
    "alert_log":           [],
    "confirmed_incidents": [],
    "dismissed_incidents": [],
    "alert_start_time":    {},
    "alerts_sent":         set(),
    "paused":              False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
ZONE_COLOR  = {"RED": "#ef4444", "YELLOW": "#f59e0b", "GREEN": "#22c55e"}
ZONE_BG     = {"RED": "#fef2f2", "YELLOW": "#fffbeb", "GREEN": "#f0fdf4"}
ZONE_BORDER = {"RED": "#fca5a5", "YELLOW": "#fcd34d", "GREEN": "#86efac"}
SEV_COLOR   = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#f59e0b", "LOW": "#3b82f6"}
EMO_COLOR   = {
    "fear":    "#ef4444", "anger":   "#f97316", "disgust": "#a855f7",
    "sadness": "#6366f1", "surprise":"#f59e0b", "joy":     "#22c55e", "neutral": "#6b7280",
}
CAT_ICON    = {
    "medical":"🏥", "fire":"🔥", "violence":"⚔️",
    "accident":"🚗", "theft":"🔓", "mental_health":"🧠", "normal":"✅",
}

# Emergency keywords to highlight in transcripts
HIGHLIGHT_KEYWORDS = [
    "bachao","ambulance","help","fire","aag","doctor","police","khoon","blood",
    "shoot","gun","knife","chaku","maar","gir","accident","emergency","save",
    "please","attack","loot","robbery","suicide","dard","pain","hospital",
]

# ── HELPERS ────────────────────────────────────────────────────────────────────
def score_bar(score_pct: int, color: str, width: int = 120) -> str:
    """Render an inline HTML score bar."""
    filled = max(2, int(score_pct * width / 100))
    empty  = width - filled
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;vertical-align:middle">'
        f'<span style="display:inline-block;height:8px;width:{filled}px;'
        f'background:{color};border-radius:4px 0 0 4px"></span>'
        f'<span style="display:inline-block;height:8px;width:{empty}px;'
        f'background:#e5e7eb;border-radius:0 4px 4px 0"></span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
        f'color:{color};font-weight:600">{score_pct}%</span>'
        f'</span>'
    )

def highlight_transcript(text: str, keywords: List[str], extra_keywords: List[str] = None) -> str:
    """Highlight emergency keywords in transcript text."""
    if not text:
        return "—"
    all_kw = set(k.lower() for k in (keywords or []) + (extra_keywords or []) + HIGHLIGHT_KEYWORDS)
    words  = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(".,!?\"'")
        if clean in all_kw:
            result.append(
                f'<span style="background:#fef2f2;color:#dc2626;font-weight:700;'
                f'padding:0 3px;border-radius:3px;border:1px solid #fca5a5">{word}</span>'
            )
        else:
            result.append(f'<span style="color:#374151">{word}</span>')
    return " ".join(result)

def mono(text, color="#6b7280", size="12px"):
    return f'<span style="font-family:\'JetBrains Mono\',monospace;color:{color};font-size:{size}">{text}</span>'

def card(content: str, padding="16px 20px", bg="white", border="1px solid #e8ecf0",
         radius="10px", shadow="0 1px 4px rgba(0,0,0,0.06)", extra="") -> str:
    return (f'<div style="background:{bg};border:{border};border-radius:{radius};'
            f'padding:{padding};box-shadow:{shadow};{extra}">{content}</div>')

def label(text, color="#6b7280"):
    return f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:{color};margin-bottom:6px">{text}</div>'

# ── DATA ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3)
def get_chunks() -> List[Dict]:
    return load_chunks()

def get_chunk_audio_path(chunk_id: str) -> Optional[str]:
    """Find WAV file for a chunk in output/chunks/."""
    for ext in [".wav", ".WAV"]:
        p = os.path.join(CHUNKS_DIR, chunk_id + ext)
        if os.path.exists(p):
            return p
    return None

def calc_avg_score(chunks):
    if not chunks: return 0
    return round(sum(c.get("score",{}).get("final_score",0) for c in chunks) / len(chunks) * 100, 1)

def calc_processing_speed(chunks):
    """Estimate avg processing time per chunk from timestamps."""
    if len(chunks) < 2: return "N/A"
    starts = [c.get("chunk_start", 0) for c in chunks if c.get("chunk_start")]
    if len(starts) < 2: return "~0.3s"
    diffs = [starts[i+1]-starts[i] for i in range(len(starts)-1)]
    avg   = sum(diffs) / len(diffs)
    return f"~{avg:.1f}s/chunk"

# ── ALERT FIRING ───────────────────────────────────────────────────────────────
def fire_alerts(chunk, all_chunks, reason):
    iid = chunk.get("score", {}).get("incident_id", "N/A")
    if iid in st.session_state.alerts_sent:
        return
    idx    = next((i for i, c in enumerate(all_chunks) if c.get("chunk_id") == chunk.get("chunk_id")), 0)
    recent = all_chunks[max(0, idx - 4): idx + 1]
    sms_r  = send_emergency_sms(chunk)
    eml_r  = send_emergency_email(chunk, recent)
    st.session_state.alert_log.append({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "incident":   iid,
        "category":   chunk.get("score", {}).get("emergency_category", "?").upper(),
        "severity":   chunk.get("score", {}).get("severity", "HIGH"),
        "reason":     reason,
        "sms_sent":   sms_r["sent"],
        "email_sent": eml_r["sent"],
        "text":       (chunk.get("text") or "")[:70],
        "score":      round(chunk.get("score", {}).get("final_score", 0) * 100),
    })
    st.session_state.alerts_sent.add(iid)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — HEADER BAR
# ══════════════════════════════════════════════════════════════════════════════
def render_header(chunks, red_count):
    now    = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    speed  = calc_processing_speed(chunks)
    paused = st.session_state.paused

    # Status dot
    if red_count > 0:
        dot_color, dot_text, dot_bg = "#ef4444", f"🔴 EMERGENCY ({red_count} active)", "#fef2f2"
    elif any(c.get("score",{}).get("zone")=="YELLOW" for c in chunks):
        dot_color, dot_text, dot_bg = "#f59e0b", "🟡 SUSPICIOUS ACTIVITY", "#fffbeb"
    else:
        dot_color, dot_text, dot_bg = "#22c55e", "🟢 ALL CLEAR", "#f0fdf4"

    st.markdown(f"""
    <div style="background:white;border-bottom:2px solid #e8ecf0;
                padding:12px 28px;display:flex;align-items:center;
                justify-content:space-between;position:sticky;top:0;z-index:999;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">

      <!-- Left: logo + status -->
      <div style="display:flex;align-items:center;gap:20px">
        <div>
          <span style="font-size:20px;font-weight:800;color:#1a1f2e;letter-spacing:-0.5px">
            🛡️ VoiceGuard
          </span>
          <span style="font-size:11px;color:#9ca3af;margin-left:8px;font-weight:500">
            Security Dashboard v3
          </span>
        </div>
        <div style="background:{dot_bg};border:1px solid {dot_color}44;
                    padding:5px 14px;border-radius:20px;
                    font-size:12px;font-weight:700;color:{dot_color};
                    {'animation:blink 1.5s infinite' if red_count > 0 else ''}">
          {dot_text}
        </div>
      </div>

      <!-- Right: info -->
      <div style="display:flex;align-items:center;gap:24px">
        <div style="text-align:center">
          <div style="font-size:10px;color:#9ca3af;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.8px">Processing Speed</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                      color:#374151;font-weight:600">{speed}</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:10px;color:#9ca3af;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.8px">Last Updated</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                      color:#374151">{now}</div>
        </div>
        <div style="display:flex;align-items:center;gap:6px">
          <div style="width:8px;height:8px;border-radius:50%;
                      background:{'#f59e0b' if paused else '#22c55e'};
                      {'animation:blink 1s infinite' if not paused else ''}"></div>
          <span style="font-size:12px;color:#6b7280;font-weight:600">
            {'PAUSED' if paused else 'LIVE'}
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — STATS ROW
# ══════════════════════════════════════════════════════════════════════════════
def render_stats(chunks):
    total  = len(chunks)
    green  = sum(1 for c in chunks if c.get("score",{}).get("zone")=="GREEN")
    yellow = sum(1 for c in chunks if c.get("score",{}).get("zone")=="YELLOW")
    red    = sum(1 for c in chunks if c.get("score",{}).get("zone")=="RED")
    fired  = len(st.session_state.alert_log)
    avg    = calc_avg_score(chunks)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Chunks", total,  help="Total audio chunks analyzed")
    c2.metric("🟢 Safe",       green,  help="Chunks in GREEN zone")
    c3.metric("🟡 Warning",    yellow, help="Chunks in YELLOW zone")
    c4.metric("🔴 Emergency",  red,    help="Chunks in RED zone")
    c5.metric("Alerts Fired",  fired,  help="Total SMS/email alerts sent")

    # Connection status pills
    sms_ok   = TWILIO_ENABLED
    email_ok = EMAIL_ENABLED
    st.markdown(f"""
    <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
      <span style="background:{'#f0fdf4' if sms_ok else '#fffbeb'};
                   color:{'#166534' if sms_ok else '#92400e'};
                   border:1px solid {'#86efac' if sms_ok else '#fcd34d'};
                   padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600">
        {'✅' if sms_ok else '⚠️'} SMS {'LIVE' if sms_ok else 'DRY-RUN'}
      </span>
      <span style="background:{'#f0fdf4' if email_ok else '#fffbeb'};
                   color:{'#166534' if email_ok else '#92400e'};
                   border:1px solid {'#86efac' if email_ok else '#fcd34d'};
                   padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600">
        {'✅' if email_ok else '⚠️'} Email {'LIVE' if email_ok else 'DRY-RUN'}
      </span>
      <span style="background:#f0fdf4;color:#166534;border:1px solid #86efac;
                   padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600">
        ✅ Sound ACTIVE
      </span>
      <span style="background:#eff6ff;color:#1e40af;border:1px solid #93c5fd;
                   padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600">
        📊 Avg Score: {avg}%
      </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EMERGENCY ALERT CARD
# ══════════════════════════════════════════════════════════════════════════════
def render_alert_card(chunk, all_chunks):
    s           = chunk.get("score", {})
    iid         = s.get("incident_id", "N/A")
    category    = s.get("emergency_category", "unknown")
    severity    = s.get("severity", "HIGH")
    final_score = round(s.get("final_score", 0) * 100)
    emotion     = s.get("dominant_emotion", "?")
    text        = chunk.get("text", "")
    chunk_id    = chunk.get("chunk_id", "?")
    auto_alert  = s.get("auto_alert", False)
    trend_up    = s.get("trend_upgraded", False)
    rising      = s.get("rising_trend", False)
    comp        = s.get("components", {})
    keywords    = chunk.get("keywords_found", s.get("keywords_found", []))
    cat_icon    = CAT_ICON.get(category, "⚠️")
    sev_col     = SEV_COLOR.get(severity, "#ef4444")
    emo_col     = EMO_COLOR.get(emotion, "#6b7280")

    # Track start time
    if iid not in st.session_state.alert_start_time:
        st.session_state.alert_start_time[iid] = time.time()
        trigger_for_zone("RED")

    elapsed   = int(time.time() - st.session_state.alert_start_time.get(iid, time.time()))
    remaining = max(0, AUTO_TRIGGER_SEC - elapsed)
    timer_pct = int((remaining / AUTO_TRIGGER_SEC) * 100)
    timer_col = "#ef4444" if remaining < 20 else "#f59e0b"

    # Auto-trigger
    if remaining == 0 and auto_alert and iid not in st.session_state.alerts_sent:
        fire_alerts(chunk, all_chunks, "AUTO")
        st.session_state.confirmed_incidents.append(iid)

    # Highlighted transcript
    hl_text = highlight_transcript(text, keywords)

    # Badge HTML built outside f-string (avoids quote conflicts)
    trend_badge  = '<span style="background:#fffbeb;color:#92400e;border:1px solid #fcd34d;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700">&#8593; TREND ESCALATION</span>' if trend_up else ''
    rising_badge = '<span style="background:#fff7ed;color:#9a3412;border:1px solid #fdba74;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700">&#128200; RISING SCORE</span>'   if rising   else ''

    # Score breakdown grid — built outside f-string to avoid nested f-string issues
    breakdown_items = [
        ("Emotion",     emotion.upper(),                                    emo_col),
        ("Emo Weight",  f"{round(comp.get('emotion_component',0)*100)}%",   "#6366f1"),
        ("Emrg Weight", f"{round(comp.get('emergency_component',0)*100)}%", sev_col),
        ("Keyword Wt",  f"{round(comp.get('keyword_component',0)*100)}%",   "#3b82f6"),
        ("Sarcasm Ded", f"-{round(comp.get('sarcasm_deduction',0)*100)}%",  "#6b7280"),
    ]
    score_breakdown_html = "".join(
        f'<div style="background:white;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:10px 14px;text-align:center">'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#9ca3af;margin-bottom:4px">{lbl}</div>'
        f'<div style="font-size:15px;font-weight:700;color:{col};'
        f'font-family:JetBrains Mono,monospace">{val}</div></div>'
        for lbl, val, col in breakdown_items
    )

    st.markdown(f"""
    <div style="background:#fef2f2;border:2px solid #fca5a5;
                border-left:6px solid {sev_col};border-radius:12px;
                padding:24px;margin:8px 0;
                box-shadow:0 4px 20px rgba(239,68,68,0.15);
                animation:fadeIn 0.4s ease-out">

      <!-- Title row -->
      <div style="display:flex;align-items:flex-start;
                  justify-content:space-between;margin-bottom:16px">
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:32px">{cat_icon}</span>
          <div>
            <div style="font-size:20px;font-weight:800;color:{sev_col};
                        letter-spacing:-0.3px">
              {category.upper()} EMERGENCY
            </div>
            <div style="display:flex;gap:6px;margin-top:5px;flex-wrap:wrap">
              <span style="background:{sev_col};color:white;
                           padding:2px 10px;border-radius:4px;
                           font-size:11px;font-weight:700">{severity}</span>
              <span style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
                           padding:2px 10px;border-radius:4px;
                           font-size:11px;font-weight:700">{iid}</span>
              <span style="background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;
                           padding:2px 10px;border-radius:4px;
                           font-size:11px;font-weight:600">{chunk_id}</span>
              {trend_badge}
              {rising_badge}
            </div>
          </div>
        </div>
        <!-- Big score -->
        <div style="text-align:right">
          <div style="font-size:44px;font-weight:800;color:{sev_col};
                      font-family:'JetBrains Mono',monospace;line-height:1">
            {final_score}%
          </div>
          <div style="font-size:10px;color:#9ca3af;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.8px">
            Confidence Score
          </div>
        </div>
      </div>

      <!-- Confidence bar -->
      <div style="margin-bottom:16px">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.8px;color:#9ca3af;margin-bottom:6px">
          Confidence Level
        </div>
        <div style="background:#fee2e2;border-radius:6px;height:12px;overflow:hidden">
          <div style="background:linear-gradient(90deg,{sev_col},{sev_col}cc);
                      height:12px;width:{final_score}%;border-radius:6px;
                      box-shadow:0 0 8px {sev_col}66;
                      transition:width 0.5s ease"></div>
        </div>
      </div>

      <!-- Transcript with keyword highlighting -->
      <div style="background:white;border:1px solid #fca5a5;border-radius:8px;
                  padding:14px 18px;margin-bottom:16px">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.8px;color:#9ca3af;margin-bottom:8px">
          Flagged Transcript — Keywords highlighted in red
        </div>
        <div style="font-size:15px;line-height:1.7">{hl_text}</div>
      </div>

      <!-- Score breakdown row -->
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:16px">
        {score_breakdown_html}
      </div>

      <!-- Keywords row -->
      <div style="margin-bottom:16px">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.8px;color:#9ca3af;margin-right:10px">
          Keywords Found:
        </span>
        {' '.join(f'<span style="background:#fef2f2;color:#dc2626;border:1px solid #fca5a5;padding:2px 9px;border-radius:4px;font-size:12px;font-weight:700;margin-right:4px">{kw}</span>' for kw in (keywords if keywords else ["none"]))}
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── ACTION BUTTONS + TIMER ─────────────────────────────────────────────
    col_confirm, col_dismiss, col_audio, col_timer = st.columns([2, 2, 1.5, 1.5])

    with col_confirm:
        if st.button(f"✅  CONFIRM EMERGENCY — {iid}",
                     key=f"confirm_{iid}", type="primary", use_container_width=True):
            fire_alerts(chunk, all_chunks, "CONFIRMED")
            st.session_state.confirmed_incidents.append(iid)
            stop_alarm()
            st.success(f"Emergency confirmed. SMS + Email alerts sent for {iid}.")
            st.rerun()

    with col_dismiss:
        if st.button(f"❌  FALSE ALARM — DISMISS",
                     key=f"dismiss_{iid}", use_container_width=True):
            st.session_state.dismissed_incidents.append(iid)
            stop_alarm()
            st.info(f"Dismissed {iid} as false alarm. Logged.")
            st.rerun()

    with col_audio:
        audio_path = get_chunk_audio_path(chunk_id)
        if audio_path:
            with open(audio_path, "rb") as af:
                st.audio(af.read(), format="audio/wav")
        else:
            no_audio_msg = f"No audio file found<br>{chunk_id}.wav not in {CHUNKS_DIR}/"
            st.markdown(
                f'<div style="background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;'
                f'padding:10px;text-align:center;color:#9ca3af;font-size:12px;margin-top:4px">'
                f'▶ {no_audio_msg}</div>',
                unsafe_allow_html=True
            )

    with col_timer:
        bar_w = max(2, timer_pct)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:8px;
                    padding:12px;text-align:center">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.8px;color:#9ca3af;margin-bottom:4px">
            {'Auto-trigger in' if remaining > 0 else 'Timer expired'}
          </div>
          <div style="font-size:26px;font-weight:800;
                      font-family:'JetBrains Mono',monospace;
                      color:{timer_col};
                      {'animation:blink 0.8s infinite' if remaining < 15 else ''}">
            {remaining}s
          </div>
          <div style="background:#f3f4f6;border-radius:4px;height:5px;
                      margin-top:6px;overflow:hidden">
            <div style="background:{timer_col};height:5px;width:{bar_w}%;
                        border-radius:4px;transition:width 1s linear"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LIVE TRANSCRIPT FEED
# ══════════════════════════════════════════════════════════════════════════════
def render_live_feed(chunks):
    # Filter controls
    cf1, cf2, cf3, cf4 = st.columns([3, 1.5, 1.2, 1])
    with cf1:
        search = st.text_input("🔍", placeholder="Search transcript...", label_visibility="collapsed", key="search_feed")
    with cf2:
        zone_f = st.selectbox("Zone", ["ALL","RED","YELLOW","GREEN"], label_visibility="collapsed", key="zone_filter")
    with cf3:
        n_show = st.selectbox("Show last", [25, 50, "All"], label_visibility="collapsed", key="n_show")
    with cf4:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear(); st.rerun()

    # Filter data
    display = chunks
    if zone_f != "ALL":
        display = [c for c in display if c.get("score",{}).get("zone") == zone_f]
    if search:
        display = [c for c in display if search.lower() in (c.get("text") or "").lower()]
    if n_show != "All":
        display = display[-int(n_show):]

    st.markdown(f"<div style='font-size:11px;color:#9ca3af;margin-bottom:8px'>{len(display)} chunks shown</div>", unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="display:grid;
                grid-template-columns:60px 85px 160px 90px 100px 1fr;
                gap:8px;padding:9px 14px;background:#1a1f2e;
                border-radius:8px 8px 0 0;
                color:#9ca3af;font-size:10px;font-weight:700;
                text-transform:uppercase;letter-spacing:0.8px">
      <div>Time</div><div>Zone</div><div>Score</div>
      <div>Emotion</div><div>Category</div><div>Transcript</div>
    </div>
    """, unsafe_allow_html=True)

    rows = ""
    for i, c in enumerate(display):
        s      = c.get("score", {})
        zone   = s.get("zone", "GREEN")
        zc     = ZONE_COLOR.get(zone, "#22c55e")
        zbg    = "#ffffff" if i % 2 == 0 else "#f9fafb"
        zbg    = ZONE_BG.get(zone, zbg) if zone in ("RED","YELLOW") else zbg
        sc_pct = round(s.get("final_score", 0) * 100)
        emo    = s.get("dominant_emotion", "neutral")
        emo_c  = EMO_COLOR.get(emo, "#6b7280")
        cat    = s.get("emergency_category", "?")
        txt    = (c.get("text") or "—")[:72]
        ts     = round(c.get("chunk_start", 0), 1)
        kws    = c.get("keywords_found", s.get("keywords_found", []))
        hl_txt = highlight_transcript(txt, kws) if zone == "RED" else f'<span style="color:#4b5563">{txt}</span>'

        # Inline mini score bar
        bar_w  = max(2, sc_pct)
        bar    = (f'<span style="display:inline-flex;align-items:center;gap:5px">'
                  f'<span style="display:inline-block;height:6px;width:{int(bar_w*0.8)}px;'
                  f'background:{zc};border-radius:3px;opacity:0.8"></span>'
                  f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                  f'font-weight:700;color:{zc}">{sc_pct}%</span></span>')

        border_l = f"border-left:3px solid {zc}" if zone in ("RED","YELLOW") else "border-left:3px solid transparent"

        rows += f"""
        <div style="display:grid;
                    grid-template-columns:60px 85px 160px 90px 100px 1fr;
                    gap:8px;padding:8px 14px;background:{zbg};
                    border-bottom:1px solid #f3f4f6;{border_l};
                    font-size:12px;align-items:center;
                    {'border-radius:0 0 8px 8px' if i==len(display)-1 else ''}">
          <div style="font-family:'JetBrains Mono',monospace;color:#9ca3af;font-size:11px">{ts}s</div>
          <div style="color:{zc};font-weight:700;font-size:12px">● {zone}</div>
          <div>{bar}</div>
          <div style="color:{emo_c};font-size:11px;font-weight:500;text-transform:capitalize">{emo}</div>
          <div style="color:#6b7280;font-size:11px">{CAT_ICON.get(cat,'')} {cat}</div>
          <div style="font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{hl_txt}</div>
        </div>"""

    st.markdown(f'<div style="border:1px solid #e5e7eb;border-top:none;border-radius:0 0 8px 8px;overflow:hidden">{rows}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def render_analytics(chunks):
    col_kw, col_emo, col_tl = st.columns([1, 1, 2])

    # ── Keyword frequency ──
    with col_kw:
        st.markdown(card(
            label("🔑 Keyword Frequency") + "<div id='kw'></div>",
            padding="16px 20px"
        ), unsafe_allow_html=True)

        # Collect all keywords found across RED/YELLOW chunks
        kw_counter = Counter()
        for c in chunks:
            s    = c.get("score", {})
            zone = s.get("zone", "GREEN")
            if zone in ("RED", "YELLOW"):
                kws = c.get("keywords_found", s.get("keywords_found", []))
                for kw in kws:
                    kw_counter[kw.lower()] += 1

        if kw_counter:
            top = kw_counter.most_common(8)
            max_count = top[0][1] if top else 1
            bars_html = ""
            for kw, cnt in top:
                pct = int(cnt / max_count * 100)
                w   = max(4, pct)
                bars_html += f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                  <div style="width:70px;font-size:11px;color:#374151;
                              font-weight:600;text-align:right;
                              overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{kw}</div>
                  <div style="flex:1;background:#f3f4f6;border-radius:4px;height:8px">
                    <div style="background:#ef4444;height:8px;border-radius:4px;
                                width:{w}%;box-shadow:0 0 6px #ef444466"></div>
                  </div>
                  <div style="width:20px;font-size:11px;color:#ef4444;
                              font-family:'JetBrains Mono',monospace;font-weight:700">{cnt}</div>
                </div>"""
            st.markdown(f'<div style="background:white;border:1px solid #e8ecf0;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">{bars_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:white;border:1px solid #e8ecf0;border-radius:10px;padding:16px;color:#9ca3af;font-size:13px;text-align:center">No keywords detected yet<br>(appears in RED/YELLOW zones)</div>', unsafe_allow_html=True)

    # ── Emotion distribution ──
    with col_emo:
        st.markdown(card(
            label("😶 Emotion Distribution") + "<div></div>",
            padding="16px 20px"
        ), unsafe_allow_html=True)

        counts = Counter(c.get("score",{}).get("dominant_emotion","neutral") for c in chunks)
        total_e = sum(counts.values()) or 1
        emo_html = ""
        for emo, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct  = round(cnt / total_e * 100)
            col  = EMO_COLOR.get(emo, "#6b7280")
            w    = max(2, pct)
            emo_html += f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;
                          margin-bottom:4px;font-size:12px">
                <span style="color:#374151;font-weight:500;
                             text-transform:capitalize">{emo}</span>
                <span style="font-family:'JetBrains Mono',monospace;
                             color:{col};font-weight:700">{pct}%</span>
              </div>
              <div style="background:#f3f4f6;border-radius:5px;height:8px;overflow:hidden">
                <div style="background:{col};height:8px;width:{w}%;border-radius:5px;
                            box-shadow:0 0 6px {col}55;transition:width 0.4s"></div>
              </div>
            </div>"""
        st.markdown(f'<div style="background:white;border:1px solid #e8ecf0;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">{emo_html}</div>', unsafe_allow_html=True)

    # ── Score timeline ── (pure HTML, no pandas/altair = no recursion error)
    with col_tl:
        st.markdown(card(
            label("📈 Score Timeline") + "<div></div>",
            padding="16px 20px"
        ), unsafe_allow_html=True)

        if chunks:
            max_s  = max((c.get("score",{}).get("final_score",0) for c in chunks), default=1) or 0.01
            bars   = ""
            n      = min(len(chunks), 60)   # show max 60 bars
            subset = chunks[-n:]
            bar_w  = max(4, int(580 / n) - 2)

            for c in subset:
                s_val = c.get("score",{}).get("final_score", 0)
                zone  = c.get("score",{}).get("zone", "GREEN")
                col   = ZONE_COLOR.get(zone, "#22c55e")
                h     = max(4, int(s_val / max_s * 80))
                tip   = f"{round(s_val*100)}%"
                bars += f"""<div title="{tip}" style="display:inline-block;
                            width:{bar_w}px;height:{h}px;background:{col};
                            border-radius:2px 2px 0 0;margin:0 1px;
                            vertical-align:bottom;opacity:0.85;
                            transition:opacity 0.2s" onmouseover="this.style.opacity=1"
                            onmouseout="this.style.opacity=0.85"></div>"""

            # Zone threshold lines
            y72 = int(0.72 * 80)
            y45 = int(0.45 * 80)

            st.markdown(f"""
            <div style="background:white;border:1px solid #e8ecf0;border-radius:10px;
                        padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
              <div style="position:relative;height:92px;overflow:hidden">
                <!-- Threshold lines -->
                <div style="position:absolute;bottom:{y72}px;left:0;right:0;
                            border-top:1px dashed #ef444488;z-index:2">
                  <span style="font-size:9px;color:#ef4444;font-weight:700;
                               background:white;padding:0 3px">RED 72%</span>
                </div>
                <div style="position:absolute;bottom:{y45}px;left:0;right:0;
                            border-top:1px dashed #f59e0b88;z-index:2">
                  <span style="font-size:9px;color:#f59e0b;font-weight:700;
                               background:white;padding:0 3px">YELLOW 45%</span>
                </div>
                <!-- Bars -->
                <div style="position:absolute;bottom:0;left:0;right:0;
                            display:flex;align-items:flex-end;flex-wrap:nowrap;
                            overflow:hidden;height:80px">
                  {bars}
                </div>
              </div>
              <div style="display:flex;gap:12px;margin-top:6px;flex-wrap:wrap">
                <span style="font-size:10px;color:#22c55e;font-weight:700">■ GREEN (&lt;45%)</span>
                <span style="font-size:10px;color:#f59e0b;font-weight:700">■ YELLOW (45–72%)</span>
                <span style="font-size:10px;color:#ef4444;font-weight:700">■ RED (&gt;72%)</span>
                <span style="font-size:10px;color:#9ca3af">{n} most recent chunks shown</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — INCIDENT LOG
# ══════════════════════════════════════════════════════════════════════════════
def render_incident_log():
    if not st.session_state.alert_log:
        st.markdown("""
        <div style="background:white;border:1px solid #e8ecf0;border-radius:10px;
                    padding:20px;text-align:center;color:#9ca3af;font-size:13px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06)">
          No incidents logged this session — system is monitoring
        </div>
        """, unsafe_allow_html=True)
        return

    # Table header
    st.markdown("""
    <div style="display:grid;
                grid-template-columns:80px 90px 120px 80px 70px 70px 70px 1fr;
                gap:8px;padding:9px 16px;background:#1a1f2e;
                border-radius:8px 8px 0 0;
                color:#9ca3af;font-size:10px;font-weight:700;
                text-transform:uppercase;letter-spacing:0.8px">
      <div>Time</div><div>Incident</div><div>Category</div>
      <div>Score</div><div>Severity</div><div>Trigger</div>
      <div>SMS</div><div>Transcript</div>
    </div>
    """, unsafe_allow_html=True)

    rows = ""
    for i, e in enumerate(reversed(st.session_state.alert_log)):
        sc      = SEV_COLOR.get(e.get("severity","HIGH"), "#f97316")
        rc      = "#ef4444" if e["reason"]=="AUTO" else "#22c55e"
        sms_ic  = "✅" if e["sms_sent"]   else "❌"
        eml_ic  = "✅" if e["email_sent"] else "❌"
        bg      = "white" if i % 2 == 0 else "#f9fafb"
        r_str   = e.get("reason","?")
        score_v = e.get("score","?")

        rows += f"""
        <div style="display:grid;
                    grid-template-columns:80px 90px 120px 80px 70px 70px 70px 1fr;
                    gap:8px;padding:9px 16px;background:{bg};
                    border-bottom:1px solid #f3f4f6;
                    border-left:3px solid {sc};font-size:12px;align-items:center;
                    {'border-radius:0 0 8px 8px' if i==len(st.session_state.alert_log)-1 else ''}">
          <div style="font-family:'JetBrains Mono',monospace;color:#6b7280;
                      font-size:11px">{e['time']}</div>
          <div style="color:#1d4ed8;font-weight:700;font-size:11px">{e['incident']}</div>
          <div style="color:{sc};font-weight:600;font-size:11px">{e['category']}</div>
          <div style="font-family:'JetBrains Mono',monospace;color:{sc};
                      font-weight:700">{score_v}%</div>
          <div><span style="background:{sc}18;color:{sc};border:1px solid {sc}44;
                            padding:1px 7px;border-radius:4px;font-size:10px;
                            font-weight:700">{e['severity']}</span></div>
          <div><span style="background:{rc}18;color:{rc};border:1px solid {rc}44;
                            padding:1px 7px;border-radius:4px;font-size:10px;
                            font-weight:700">{r_str}</span></div>
          <div style="font-size:12px">{sms_ic} {eml_ic}</div>
          <div style="color:#6b7280;font-style:italic;font-size:11px;
                      overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
            "{e.get('text','')}"
          </div>
        </div>"""

    st.markdown(f'<div style="border:1px solid #e5e7eb;border-top:none;border-radius:0 0 8px 8px;overflow:hidden">{rows}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CONTROLS FOOTER
# ══════════════════════════════════════════════════════════════════════════════
def render_controls():
    st.markdown("""
    <div style="background:white;border:1px solid #e8ecf0;border-radius:10px;
                padding:14px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.8px;color:#9ca3af;margin-bottom:10px">
        System Controls
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        label_txt = "⏸  Pause Monitoring" if not st.session_state.paused else "▶  Resume Monitoring"
        if st.button(label_txt, use_container_width=True, key="pause_btn"):
            st.session_state.paused = not st.session_state.paused
            st.rerun()

    with c2:
        if st.button("🔕  Silence Alarm", use_container_width=True, key="silence_btn"):
            stop_alarm()
            st.success("Alarm silenced.")

    with c3:
        if st.button("🗑  Clear Alert Log", use_container_width=True, key="clear_btn"):
            st.session_state.alert_log = []
            st.session_state.confirmed_incidents = []
            st.session_state.dismissed_incidents = []
            st.session_state.alert_start_time    = {}
            st.session_state.alerts_sent         = set()
            st.rerun()

    with c4:
        # Export incident log as JSON
        if st.session_state.alert_log:
            log_json = json.dumps(st.session_state.alert_log, indent=2)
            st.download_button(
                "📥  Export Incident Log",
                data=log_json,
                file_name=f"incident_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="export_btn",
            )
        else:
            st.button("📥  Export Log (empty)", disabled=True, use_container_width=True, key="export_dis")

    with c5:
        if st.button("🔄  Force Refresh", use_container_width=True, key="force_refresh"):
            st.cache_data.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    chunks = get_chunks()

    # Unresolved RED incidents
    red_unresolved = [
        c for c in chunks
        if c.get("score",{}).get("zone") == "RED"
        and c.get("score",{}).get("incident_id") not in st.session_state.confirmed_incidents
        and c.get("score",{}).get("incident_id") not in st.session_state.dismissed_incidents
    ]

    # ── HEADER ──────────────────────────────────────────────────────────────
    render_header(chunks, len(red_unresolved))

    # ── MAIN CONTENT PADDING ─────────────────────────────────────────────
    st.markdown("<div style='padding:20px 28px'>", unsafe_allow_html=True)

    if not chunks:
        st.error(
            f"No data found for dashboard source: {data_source_label()}. "
            "If you are in file mode, run `python test_phase4.py` first. "
            "If you are in live mode, make sure `python main.py --mode live` is running."
        )
        st.stop()

    # ── STATS ROW ────────────────────────────────────────────────────────
    render_stats(chunks)

    # ── EMERGENCY ALERT CARDS ────────────────────────────────────────────
    if red_unresolved:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:14px;font-weight:700;color:#ef4444;
                    letter-spacing:0.3px;margin-bottom:8px">
          🚨 EMERGENCY ALERTS — Action Required
        </div>
        """, unsafe_allow_html=True)
        for chunk in red_unresolved:
            render_alert_card(chunk, chunks)
    else:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:14px;font-weight:700;color:#1a1f2e;
                letter-spacing:0.3px;margin-bottom:8px">
      📡 Live Transcript Feed
    </div>
    """, unsafe_allow_html=True)
    render_live_feed(chunks)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:14px;font-weight:700;color:#1a1f2e;
                letter-spacing:0.3px;margin-bottom:12px">
      📊 Analytics
    </div>
    """, unsafe_allow_html=True)
    render_analytics(chunks)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:14px;font-weight:700;color:#1a1f2e;
                letter-spacing:0.3px;margin-bottom:8px">
      📋 Incident Log
    </div>
    """, unsafe_allow_html=True)
    render_incident_log()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    render_controls()

    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-refresh (skip if paused)
    if not st.session_state.paused:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()