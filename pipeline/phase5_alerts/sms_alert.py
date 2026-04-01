"""
Phase 5 - SMS Alert
Sends emergency SMS via Twilio.

Setup:
    1. Create free Twilio account at twilio.com
    2. Get Account SID + Auth Token from dashboard
    3. Get a Twilio phone number (free trial includes one)
    4. Fill in config below OR use environment variables

Testing without Twilio:
    Set TWILIO_ENABLED = False → SMS is printed to console instead.
    Safe for demo if Twilio not set up yet.

Install:
    pip install twilio
"""

import os
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Fill these in, OR set as environment variables (recommended)
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "your_account_sid_here")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",   "your_auth_token_here")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER",  "+1XXXXXXXXXX")

# Numbers to alert — add your test number here
ALERT_NUMBERS: List[str] = [
    os.getenv("ALERT_NUMBER_1", "+91XXXXXXXXXX"),   # Security
    # os.getenv("ALERT_NUMBER_2", "+91XXXXXXXXXX"), # Admin (uncomment when ready)
]

# Set to False to run in dry-run mode (prints SMS to console, doesn't send)
TWILIO_ENABLED = (
    TWILIO_ACCOUNT_SID != "your_account_sid_here"
    and TWILIO_AUTH_TOKEN != "your_auth_token_here"
)


# ── MESSAGE BUILDER ────────────────────────────────────────────────────────────

def build_sms_message(chunk: Dict) -> str:
    """
    Build SMS text from a RED zone chunk.
    Kept under 160 chars for single SMS (no splitting).
    """
    score    = chunk.get("score", {})
    category = score.get("emergency_category", "unknown").upper()
    severity = score.get("severity", "HIGH")
    incident = score.get("incident_id", "N/A")
    text     = chunk.get("text", "")[:60]
    time_str = datetime.now().strftime("%H:%M:%S")
    final    = round(score.get("final_score", 0) * 100)

    lines = [
        f"EMERGENCY ALERT [{severity}]",
        f"Type: {category}",
        f"Time: {time_str}",
        f"Score: {final}%",
        f"Incident: {incident}",
        f'Transcript: "{text}"',
        "ACTION REQUIRED",
    ]
    return "\n".join(lines)


# ── SEND FUNCTIONS ─────────────────────────────────────────────────────────────

def send_sms(to_number: str, message: str) -> bool:
    """
    Send a single SMS. Returns True on success.
    Falls back to console print if Twilio not configured.
    """
    if not TWILIO_ENABLED:
        logger.warning("[SMS DRY-RUN] Would send to %s:\n%s", to_number, message)
        return True  # treated as success in dry-run

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=to_number,
        )
        logger.info(f"[SMS] Sent to {to_number} | SID: {msg.sid}")
        return True
    except ImportError:
        logger.error("[SMS] twilio not installed. Run: pip install twilio")
        return False
    except Exception as e:
        logger.error(f"[SMS] Failed to send to {to_number}: {e}")
        return False


def send_emergency_sms(chunk: Dict) -> Dict:
    """
    Send emergency SMS to all ALERT_NUMBERS.
    Called by dashboard on RED confirm or auto-trigger.

    Returns:
        {
            "sent": True,
            "recipients": ["+91XXX"],
            "failed": [],
            "message": "...",
            "timestamp": "14:35:42"
        }
    """
    message    = build_sms_message(chunk)
    sent_to    = []
    failed_to  = []

    logger.info(f"[SMS] Sending emergency alert to {len(ALERT_NUMBERS)} number(s)...")

    for number in ALERT_NUMBERS:
        if send_sms(number, message):
            sent_to.append(number)
        else:
            failed_to.append(number)

    result = {
        "sent":       len(sent_to) > 0,
        "recipients": sent_to,
        "failed":     failed_to,
        "message":    message,
        "timestamp":  datetime.now().strftime("%H:%M:%S"),
    }

    if sent_to:
        logger.info(f"[SMS] Alert sent to: {sent_to}")
    if failed_to:
        logger.error(f"[SMS] Failed to reach: {failed_to}")

    return result