"""
Test Phase 6 — Report Generation
Run from voice_surveillance/ folder:
    python test_phase6.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DECISIONS_PATH = "output/decisions/all_decisions.json"

print("=" * 60)
print("  TESTING PHASE 6: REPORT GENERATION")
print("=" * 60)

# ── Load Phase 4 data ─────────────────────────────────────────
if not os.path.exists(DECISIONS_PATH):
    print(f"\n[ERROR] {DECISIONS_PATH} not found.")
    print("  Run python test_phase4.py first.")
    sys.exit(1)

with open(DECISIONS_PATH) as f:
    all_chunks = json.load(f)

red_chunks = [c for c in all_chunks if c.get("score", {}).get("zone") == "RED"]
print(f"\n  Loaded {len(all_chunks)} chunks ({len(red_chunks)} RED) from Phase 4")

# ── TEST 1: Incident Report ────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 1: Incident Report (single RED incident)")
print("=" * 60)

from pipeline.phase6_reports.incident_report import generate_incident_report

if red_chunks:
    path = generate_incident_report(red_chunks[0], all_chunks, action="CONFIRMED")
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"\n  Path   : {path}")
    print(f"  Exists : {exists}")
    print(f"  Size   : {size:,} bytes ({round(size/1024, 1)} KB)")
    if exists and size > 5000:
        print("  ✅ Incident report PASSED")
    else:
        print("  ❌ Incident report FAILED — file too small or missing")
else:
    print("  [SKIP] No RED chunks found")

# ── TEST 2: Session Report ─────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 2: Session Report (full session summary)")
print("=" * 60)

from pipeline.phase6_reports.session_report import generate_session_report

# Fake alert log for test
fake_alert_log = [
    {
        "time":       "09:26:21",
        "incident":   "INC_001",
        "category":   "MEDICAL",
        "severity":   "HIGH",
        "reason":     "CONFIRMED",
        "sms_sent":   True,
        "email_sent": True,
        "text":       "Bachao bachao! Koi ambulance bulao jaldi!",
        "score":      77,
    }
]

path2  = generate_session_report(all_chunks, alert_log=fake_alert_log)
exists2 = os.path.exists(path2)
size2   = os.path.getsize(path2) if exists2 else 0
print(f"\n  Path   : {path2}")
print(f"  Exists : {exists2}")
print(f"  Size   : {size2:,} bytes ({round(size2/1024, 1)} KB)")
if exists2 and size2 > 10000:
    print("  ✅ Session report PASSED")
else:
    print("  ❌ Session report FAILED — file too small or missing")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 6 TESTS COMPLETE")
print("=" * 60)
print(f"\n  Reports saved to: output/reports/")
print(f"  Open them to verify PDF quality.")
print(f"\n  Next steps:")
print(f"  1. Integrate into dashboard — Export Session Report button")
print(f"  2. Auto-generate incident report on CONFIRM click")
print(f"  3. Run main.py to test full pipeline end-to-end")