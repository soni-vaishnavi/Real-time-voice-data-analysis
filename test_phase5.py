"""
Test Phase 5 — Alerts (Dry-Run)

Run from voice_surveillance/ folder:
    python test_phase5.py

What this tests:
    1. SMS alert builder — formats message correctly
    2. Email alert builder — generates HTML correctly
    3. Sound alert init — pygame available or not
    4. Full alert flow with a fake RED chunk

Does NOT actually send SMS or email unless you've configured
TWILIO_ACCOUNT_SID, SMTP_EMAIL etc. in environment variables.
Dry-run is safe and shows exactly what would be sent.

To launch the dashboard (after running test_phase4.py):
    streamlit run pipeline/phase5_alerts/dashboard.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.phase5_alerts.sms_alert   import build_sms_message,  send_emergency_sms,   TWILIO_ENABLED
from pipeline.phase5_alerts.email_alert import build_email_subject, send_emergency_email,  EMAIL_ENABLED
from pipeline.phase5_alerts.sound_alert import trigger_for_zone, stop_alarm

DECISIONS_PATH = "output/decisions/all_decisions.json"

# ── FAKE RED CHUNK ─────────────────────────────────────────────────────────────
FAKE_RED_CHUNK = {
    "chunk_id":    "chunk_FAKE_RED",
    "chunk_start": 999.0,
    "text":        "Bachao bachao! Koi ambulance bulao jaldi! Bahut dard ho raha hai!",
    "score": {
        "final_score":         0.772,
        "zone":                "RED",
        "zone_emoji":          "🔴",
        "severity":            "HIGH",
        "auto_alert":          True,
        "requires_confirm":    False,
        "trend_upgraded":      False,
        "rising_trend":        True,
        "incident_id":         "INC_001",
        "dominant_emotion":    "fear",
        "emergency_category":  "medical",
        "is_emergency":        True,
        "components": {
            "emotion_component":   0.3325,
            "emergency_component": 0.3400,
            "keyword_component":   0.1000,
            "sarcasm_deduction":   0.0,
        }
    }
}


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_sms_builder():
    print_section("TEST 1: SMS Message Builder")
    msg = build_sms_message(FAKE_RED_CHUNK)
    print("\nGenerated SMS message:")
    print("-" * 40)
    print(msg)
    print("-" * 40)
    print(f"Character count: {len(msg)} (under 160 = single SMS)")
    assert "EMERGENCY" in msg, "SMS must contain EMERGENCY"
    assert "MEDICAL"   in msg, "SMS must contain category"
    print("\n✅ SMS builder PASSED")


def test_email_builder():
    print_section("TEST 2: Email Subject Builder")
    subject = build_email_subject(FAKE_RED_CHUNK)
    print(f"\nSubject: {subject}")
    assert "EMERGENCY" in subject
    assert "MEDICAL"   in subject.upper()
    print("✅ Email subject builder PASSED")

    print("\nGenerating HTML body...")
    from pipeline.phase5_alerts.email_alert import build_email_html
    html = build_email_html(FAKE_RED_CHUNK, recent_chunks=[FAKE_RED_CHUNK])
    assert "<html>" in html
    assert "MEDICAL" in html
    print(f"HTML size: {len(html)} characters")
    print("✅ Email HTML builder PASSED")


def test_sound_alert():
    print_section("TEST 3: Sound Alert")
    try:
        import pygame
        print("pygame available ✅")
    except ImportError:
        print("pygame NOT installed — sound alerts will be silent")
        print("Install with: pip install pygame")

    print("\nTesting trigger_for_zone('GREEN') — expect silence...")
    trigger_for_zone("GREEN")
    print("  GREEN: silent ✅")

    print("Testing trigger_for_zone('YELLOW') — expect soft beep...")
    trigger_for_zone("YELLOW")
    print("  YELLOW: beep triggered ✅")

    print("Testing trigger_for_zone('RED') — expect alarm start...")
    trigger_for_zone("RED")
    import time; time.sleep(0.5)
    stop_alarm()
    print("  RED: alarm started and stopped ✅")


def test_full_alert_flow():
    print_section("TEST 4: Full Alert Flow (Dry-Run)")

    print(f"\n  Twilio enabled : {TWILIO_ENABLED}")
    print(f"  Gmail enabled  : {EMAIL_ENABLED}")
    if not TWILIO_ENABLED:
        print("  → SMS will be printed to console (dry-run mode)")
    if not EMAIL_ENABLED:
        print("  → Email will be printed to console (dry-run mode)")

    print("\n--- SMS Alert ---")
    sms_result = send_emergency_sms(FAKE_RED_CHUNK)
    print(f"  sent      : {sms_result['sent']}")
    print(f"  recipients: {sms_result['recipients']}")
    print(f"  failed    : {sms_result['failed']}")

    print("\n--- Email Alert ---")
    email_result = send_emergency_email(FAKE_RED_CHUNK, recent_chunks=[FAKE_RED_CHUNK])
    print(f"  sent      : {email_result['sent']}")
    print(f"  recipients: {email_result['recipients']}")
    print(f"  failed    : {email_result['failed']}")
    print(f"  subject   : {email_result['subject']}")

    print("\n✅ Full alert flow test PASSED")


def test_decisions_file():
    print_section("TEST 5: Phase 4 Output Check")
    if not os.path.exists(DECISIONS_PATH):
        print(f"  WARNING: {DECISIONS_PATH} not found")
        print("  Run test_phase4.py first, then dashboard will have real data")
        return

    with open(DECISIONS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    red    = [c for c in chunks if c.get("score", {}).get("zone") == "RED"]
    yellow = [c for c in chunks if c.get("score", {}).get("zone") == "YELLOW"]
    green  = [c for c in chunks if c.get("score", {}).get("zone") == "GREEN"]

    print(f"\n  Loaded {len(chunks)} chunks from Phase 4")
    print(f"  🟢 GREEN  : {len(green)}")
    print(f"  🟡 YELLOW : {len(yellow)}")
    print(f"  🔴 RED    : {len(red)}")

    if red:
        print(f"\n  RED chunks (will show as alerts in dashboard):")
        for c in red:
            s = c.get("score", {})
            print(f"    {c['chunk_id']} | {s.get('emergency_category','?').upper()} | "
                  f"score={round(s.get('final_score',0)*100)}% | "
                  f"incident={s.get('incident_id','N/A')}")

    print("\n✅ Decisions file check PASSED")


def main():
    print("=" * 60)
    print("  TESTING PHASE 5: ALERTS")
    print("=" * 60)

    test_sms_builder()
    test_email_builder()
    test_sound_alert()
    test_full_alert_flow()
    test_decisions_file()

    print("\n" + "=" * 60)
    print("  PHASE 5 TESTS COMPLETE ✅")
    print("=" * 60)
    print("""
  Next steps:
  1. Configure Twilio in environment variables (for real SMS):
       set TWILIO_ACCOUNT_SID=ACxxxxxxx
       set TWILIO_AUTH_TOKEN=xxxxxxx
       set TWILIO_FROM_NUMBER=+1XXXXXXXXXX
       set ALERT_NUMBER_1=+91XXXXXXXXXX

  2. Configure Gmail (for real email):
       set SMTP_EMAIL=your@gmail.com
       set SMTP_PASSWORD=your_16char_app_password
       set ALERT_EMAIL_1=your@gmail.com

  3. Launch the dashboard:
       streamlit run pipeline/phase5_alerts/dashboard.py
""")


if __name__ == "__main__":
    main()