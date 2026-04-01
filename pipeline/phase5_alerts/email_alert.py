"""
Phase 5 - Email Alert
Sends emergency email via Gmail SMTP.

Setup:
    1. Go to myaccount.google.com → Security → 2-Step Verification → ON
    2. Then go to App Passwords → create one for "Mail"
    3. Use that 16-char password below (NOT your real Gmail password)

Testing without Gmail:
    Set EMAIL_ENABLED = False → email content printed to console.

No extra library needed — smtplib is built into Python.
"""

import os
import smtplib
import logging
from email.mime.text        import MIMEText
from email.mime.multipart   import MIMEMultipart
from datetime               import datetime
from typing                 import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SMTP_EMAIL    = os.getenv("SMTP_EMAIL",    "your_gmail@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_app_password_here")
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587

ALERT_EMAILS: List[str] = [
    os.getenv("ALERT_EMAIL_1", "your_email@gmail.com"),
    # os.getenv("ALERT_EMAIL_2", "admin@example.com"),  # uncomment when ready
]

EMAIL_ENABLED = (
    SMTP_EMAIL    != "your_gmail@gmail.com"
    and SMTP_PASSWORD != "your_app_password_here"
)


# ── EMAIL BUILDER ──────────────────────────────────────────────────────────────

def build_email_html(chunk: Dict, recent_chunks: List[Dict] = None) -> str:
    """
    Build a clean HTML email body for the emergency alert.
    Includes: incident details + recent transcript history.
    """
    score    = chunk.get("score", {})
    category = score.get("emergency_category", "unknown").upper()
    severity = score.get("severity", "HIGH")
    incident = score.get("incident_id", "N/A")
    zone     = score.get("zone", "RED")
    final    = round(score.get("final_score", 0) * 100)
    emotion  = score.get("dominant_emotion", "?").upper()
    text     = chunk.get("text", "N/A")
    time_str = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # Color coding
    sev_colors = {"CRITICAL": "#cc0000", "HIGH": "#e65c00", "MEDIUM": "#cc8800", "LOW": "#336699"}
    sev_color  = sev_colors.get(severity, "#cc0000")

    # Recent transcript rows
    transcript_rows = ""
    if recent_chunks:
        for c in recent_chunks[-5:]:   # last 5 chunks
            s     = c.get("score", {})
            z     = s.get("zone", "GREEN")
            sc    = round(s.get("final_score", 0) * 100)
            tx    = c.get("text", "")[:80]
            ts    = round(c.get("chunk_start", 0), 1)
            zcolor = {"RED": "#cc0000", "YELLOW": "#cc8800", "GREEN": "#2d862d"}.get(z, "#333")
            transcript_rows += f"""
            <tr>
                <td style="padding:6px;border-bottom:1px solid #eee;color:#666">{ts}s</td>
                <td style="padding:6px;border-bottom:1px solid #eee;
                           color:{zcolor};font-weight:bold">{z}</td>
                <td style="padding:6px;border-bottom:1px solid #eee">{sc}%</td>
                <td style="padding:6px;border-bottom:1px solid #eee;font-style:italic">"{tx}"</td>
            </tr>"""

    html = f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px">

<div style="max-width:680px;margin:0 auto;background:white;
            border-radius:8px;overflow:hidden;
            box-shadow:0 2px 8px rgba(0,0,0,0.15)">

  <!-- Header -->
  <div style="background:{sev_color};padding:24px;color:white">
    <h1 style="margin:0;font-size:22px">
      &#128680; EMERGENCY ALERT — {severity} SEVERITY
    </h1>
    <p style="margin:8px 0 0 0;opacity:0.9">{time_str}</p>
  </div>

  <!-- Incident Details -->
  <div style="padding:24px;border-bottom:1px solid #eee">
    <h2 style="color:#333;margin-top:0">Incident Details</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr>
        <td style="padding:8px 0;color:#666;width:160px">Emergency Type</td>
        <td style="padding:8px 0;font-weight:bold;color:{sev_color}">{category}</td>
      </tr>
      <tr>
        <td style="padding:8px 0;color:#666">Incident ID</td>
        <td style="padding:8px 0;font-weight:bold">{incident}</td>
      </tr>
      <tr>
        <td style="padding:8px 0;color:#666">Zone</td>
        <td style="padding:8px 0;font-weight:bold;color:#cc0000">{zone}</td>
      </tr>
      <tr>
        <td style="padding:8px 0;color:#666">Combined Score</td>
        <td style="padding:8px 0;font-weight:bold">{final}%</td>
      </tr>
      <tr>
        <td style="padding:8px 0;color:#666">Dominant Emotion</td>
        <td style="padding:8px 0">{emotion}</td>
      </tr>
    </table>
  </div>

  <!-- Flagged Transcript -->
  <div style="padding:24px;border-bottom:1px solid #eee;background:#fff8f8">
    <h2 style="color:#333;margin-top:0">Flagged Transcript</h2>
    <p style="background:#ffe0e0;border-left:4px solid #cc0000;
              padding:12px 16px;border-radius:4px;
              font-style:italic;font-size:15px;color:#333">
      "{text}"
    </p>
  </div>

  <!-- Recent Transcript History -->
  {"" if not transcript_rows else f'''
  <div style="padding:24px;border-bottom:1px solid #eee">
    <h2 style="color:#333;margin-top:0">Recent Transcript History</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:#f5f5f5">
        <th style="padding:8px;text-align:left">Time</th>
        <th style="padding:8px;text-align:left">Zone</th>
        <th style="padding:8px;text-align:left">Score</th>
        <th style="padding:8px;text-align:left">Transcript</th>
      </tr>
      {transcript_rows}
    </table>
  </div>
  '''}

  <!-- Footer -->
  <div style="padding:20px;background:#f9f9f9;
              color:#888;font-size:12px;text-align:center">
    Real-Time Voice Surveillance System &mdash; BCA Project<br>
    Poornima University, Jaipur &mdash; Auto-generated alert
  </div>

</div>
</body>
</html>
"""
    return html


def build_email_subject(chunk: Dict) -> str:
    score    = chunk.get("score", {})
    category = score.get("emergency_category", "UNKNOWN").upper()
    severity = score.get("severity", "HIGH")
    time_str = datetime.now().strftime("%H:%M:%S")
    return f"[{severity}] EMERGENCY ALERT — {category} detected at {time_str}"


# ── SEND FUNCTIONS ─────────────────────────────────────────────────────────────

def send_email(to_address: str, subject: str, html_body: str) -> bool:
    """
    Send a single HTML email. Returns True on success.
    Falls back to console print if Gmail not configured.
    """
    if not EMAIL_ENABLED:
        logger.warning("[EMAIL DRY-RUN] Would send to %s\nSubject: %s", to_address, subject)
        return True

    try:
        msg                    = MIMEMultipart("alternative")
        msg["Subject"]         = subject
        msg["From"]            = SMTP_EMAIL
        msg["To"]              = to_address
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_address, msg.as_string())

        logger.info(f"[EMAIL] Sent to {to_address}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("[EMAIL] Authentication failed — check SMTP_EMAIL and SMTP_PASSWORD (use App Password, not real password)")
        return False
    except Exception as e:
        logger.error(f"[EMAIL] Failed to send to {to_address}: {e}")
        return False


def send_emergency_email(chunk: Dict, recent_chunks: List[Dict] = None) -> Dict:
    """
    Send emergency email to all ALERT_EMAILS.
    Called by dashboard on RED confirm or auto-trigger.

    Returns:
        {
            "sent": True,
            "recipients": ["email@gmail.com"],
            "failed": [],
            "subject": "...",
            "timestamp": "14:35:42"
        }
    """
    subject  = build_email_subject(chunk)
    html     = build_email_html(chunk, recent_chunks)
    sent_to  = []
    failed   = []

    logger.info(f"[EMAIL] Sending emergency alert to {len(ALERT_EMAILS)} address(es)...")

    for address in ALERT_EMAILS:
        if send_email(address, subject, html):
            sent_to.append(address)
        else:
            failed.append(address)

    result = {
        "sent":       len(sent_to) > 0,
        "recipients": sent_to,
        "failed":     failed,
        "subject":    subject,
        "timestamp":  datetime.now().strftime("%H:%M:%S"),
    }

    if sent_to:
        logger.info(f"[EMAIL] Alert sent to: {sent_to}")
    if failed:
        logger.error(f"[EMAIL] Failed to reach: {failed}")

    return result