"""
verify_stage0.py
=================
Verify all Stage 0 fixes. Run from project root:
    python verify_stage0.py

All checks use RUNTIME inspection — not source-text scanning.
Source text scanning fails when docstrings mention old values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  STAGE 0 VERIFICATION")
print("=" * 60)

errors = []

# ── CHECK 1: Config loads and validates ────────────────────────────────────────
print("\n[1] Config unification")
try:
    from pipeline.core.config import (
        WEIGHT_EMOTION, WEIGHT_EMERGENCY, WEIGHT_KEYWORD,
        ZONE_YELLOW, ZONE_RED, EMERGENCY_THRESHOLD, validate_config,
    )
    assert abs(WEIGHT_EMOTION + WEIGHT_EMERGENCY + WEIGHT_KEYWORD - 1.0) < 0.001
    assert EMERGENCY_THRESHOLD == 0.55, f"Expected 0.55, got {EMERGENCY_THRESHOLD}"
    assert 0 < ZONE_YELLOW < ZONE_RED < 1.0
    validate_config()
    print(f"  ✅ Config loaded | weights sum=1.0 | THRESHOLD={EMERGENCY_THRESHOLD} | "
          f"ZONE_YELLOW={ZONE_YELLOW} | ZONE_RED={ZONE_RED}")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"Config: {e}")

# ── CHECK 2: Root config shim ──────────────────────────────────────────────────
print("\n[2] Root config.py shim")
try:
    import config as root_cfg
    assert root_cfg.WEIGHT_EMOTION == 0.35
    assert root_cfg.ZONE_GREEN_MAX == 0.45
    print(f"  ✅ Root config shim works | WEIGHT_EMOTION={root_cfg.WEIGHT_EMOTION} | ZONE_GREEN_MAX={root_cfg.ZONE_GREEN_MAX}")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"Root config: {e}")

# ── CHECK 3: Sarcasm rules ─────────────────────────────────────────────────────
print("\n[3] Sarcasm rules (rule-based, 0 MB)")
try:
    from pipeline.phase3_analysis.sarcasm_rules import detect_sarcasm
    r1 = detect_sarcasm("haan haan bilkul ambulance bulao bore ho gaya main")
    r2 = detect_sarcasm("bachao! ambulance bulao jaldi!")
    r3 = detect_sarcasm("")
    assert r1["is_sarcastic"] == True,  f"Should detect sarcasm: {r1}"
    assert r2["is_sarcastic"] == False, f"Genuine distress should NOT be sarcasm: {r2}"
    assert r3["is_sarcastic"] == False, f"Empty text: {r3}"
    assert r1["method"] == "rules"
    print(f"  ✅ Sarcasm rules work | sarcastic={r1['sarcasm_score']:.2f} | "
          f"genuine_override={r2['sarcasm_score']:.2f} | method={r1['method']}")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"Sarcasm rules: {e}")

# ── CHECK 4: emotion_detector has no sarcasm model (RUNTIME check) ─────────────
print("\n[4] emotion_detector.py — sarcasm model removed")
try:
    import pipeline.phase3_analysis.emotion_detector as emdet
    # Runtime checks only — not source text scanning
    assert not hasattr(emdet, '_sarcasm_pipeline'), "_sarcasm_pipeline global should not exist"
    assert not hasattr(emdet, 'get_sarcasm_model'), "get_sarcasm_model() should not exist"
    from pipeline.phase3_analysis.emotion_detector import detect_emotion
    assert callable(detect_emotion)
    print("  ✅ emotion_detector has no sarcasm model | detect_emotion importable")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"emotion_detector: {e}")

# ── CHECK 5: scorer.py uses config weights (RUNTIME check) ────────────────────
print("\n[5] scorer.py imports weights from config")
try:
    import pipeline.phase4_decision.scorer as scorer_mod
    # Runtime: verify the values match config (not just that they're defined)
    from pipeline.core.config import WEIGHT_EMOTION, WEIGHT_EMERGENCY, WEIGHT_KEYWORD
    assert scorer_mod.WEIGHT_EMOTION   == WEIGHT_EMOTION,   "WEIGHT_EMOTION mismatch"
    assert scorer_mod.WEIGHT_EMERGENCY == WEIGHT_EMERGENCY, "WEIGHT_EMERGENCY mismatch"
    assert scorer_mod.WEIGHT_KEYWORD   == WEIGHT_KEYWORD,   "WEIGHT_KEYWORD mismatch"
    from pipeline.phase4_decision.scorer import compute_score
    print(f"  ✅ scorer.py imports from config | WEIGHT_EMOTION={scorer_mod.WEIGHT_EMOTION}")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"scorer: {e}")

# ── CHECK 6: zone_classifier uses config thresholds (RUNTIME) ─────────────────
print("\n[6] zone_classifier.py imports thresholds from config")
try:
    import pipeline.phase4_decision.zone_classifier as zc
    from pipeline.core.config import ZONE_YELLOW, ZONE_RED
    assert zc.ZONE_YELLOW == ZONE_YELLOW, f"ZONE_YELLOW mismatch: {zc.ZONE_YELLOW} vs {ZONE_YELLOW}"
    assert zc.ZONE_RED    == ZONE_RED,    f"ZONE_RED mismatch: {zc.ZONE_RED} vs {ZONE_RED}"
    print(f"  ✅ zone_classifier.py imports from config | ZONE_YELLOW={zc.ZONE_YELLOW} | ZONE_RED={zc.ZONE_RED}")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"zone_classifier: {e}")

# ── CHECK 7: emergency_detector uses correct threshold (RUNTIME) ───────────────
print("\n[7] emergency_detector.py imports threshold from config")
try:
    import pipeline.phase3_analysis.emergency_detector as ed
    from pipeline.core.config import EMERGENCY_THRESHOLD
    # Runtime check only — we verify the actual value, not source text
    assert ed.EMERGENCY_THRESHOLD == 0.55, \
        f"EMERGENCY_THRESHOLD should be 0.55, got {ed.EMERGENCY_THRESHOLD}"
    assert ed.EMERGENCY_THRESHOLD == EMERGENCY_THRESHOLD, \
        "emergency_detector threshold must match config value"
    from pipeline.phase3_analysis.emergency_detector import (
        start_background_loading, get_emergency_model, is_bart_ready
    )
    print(f"  ✅ emergency_detector.py | EMERGENCY_THRESHOLD={ed.EMERGENCY_THRESHOLD} | "
          f"background loading functions available")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"emergency_detector: {e}")

# ── CHECK 8: analyzer.py has no duplicate scoring (RUNTIME) ───────────────────
print("\n[8] analyzer.py — no duplicate scoring")
try:
    import pipeline.phase3_analysis.analyzer as az
    # Runtime: the function should not exist as a callable on the module
    assert not hasattr(az, 'compute_combined_score'), \
        "compute_combined_score() must not exist in analyzer.py — Phase 4 only"
    from pipeline.phase3_analysis.analyzer import run_phase3
    assert callable(run_phase3)
    print("  ✅ analyzer.py has no duplicate scoring | run_phase3 importable")
except Exception as e:
    print(f"  ❌ {e}")
    errors.append(f"analyzer: {e}")

# ── CHECK 9: Scoring formula end-to-end (no models) ──────────────────────────
print("\n[9] Scoring formula end-to-end (mock data, no models)")
try:
    from pipeline.phase4_decision.scorer import compute_score
    from pipeline.phase4_decision.zone_classifier import classify_zone

    genuine = {
        "emotion_analysis": {
            "emotion": {"dominant_emotion":"fear","dominant_score":0.92,
                        "emergency_weight":1.0,"fear_score":0.92,"anger_score":0.05},
            "sarcasm_resolution": {"score_penalty": 0.0},
        },
        "emergency_analysis": {"top_category":"medical","top_score":0.84,"is_emergency":True},
        "keyword_analysis":   {"total_boost": 0.62},
    }
    genuine["score"] = compute_score(genuine)
    classify_zone(genuine["score"])
    assert genuine["score"]["zone"]       == "RED",  f"Expected RED, got {genuine['score']}"
    assert genuine["score"]["severity"]   == "HIGH"
    assert genuine["score"]["auto_alert"] == True
    assert genuine["score"]["final_score"] > 0.72

    sarcastic = {
        "emotion_analysis": {
            "emotion": {"dominant_emotion":"neutral","dominant_score":0.65,
                        "emergency_weight":0.0,"fear_score":0.05,"anger_score":0.10},
            "sarcasm_resolution": {"score_penalty": 0.70},
        },
        "emergency_analysis": {"top_category":"normal","top_score":0.25,"is_emergency":False},
        "keyword_analysis":   {"total_boost": 0.47},
    }
    sarcastic["score"] = compute_score(sarcastic)
    classify_zone(sarcastic["score"])
    assert sarcastic["score"]["zone"] == "GREEN", f"Expected GREEN, got {sarcastic['score']}"

    print(f"  ✅ Emergency scores correctly | "
          f"genuine={genuine['score']['final_score']:.3f} (RED) | "
          f"sarcasm={sarcastic['score']['final_score']:.3f} (GREEN)")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  ❌ {e}")
    errors.append(f"Scoring: {e}")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors:
    print("  ✅ ALL STAGE 0 CHECKS PASSED")
    print("  Codebase is clean. Proceed to Stage 1.")
else:
    print(f"  ❌ {len(errors)} CHECK(S) FAILED:")
    for err in errors:
        print(f"     - {err}")
print("=" * 60)