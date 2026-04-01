print("===== VOICE SURVEILLANCE ENVIRONMENT TEST =====")

# Phase 1 — Audio Pipeline
print("\n[Phase 1] Audio Libraries")

try:
    from pydub import AudioSegment
    import noisereduce as nr
    import webrtcvad
    import numpy as np
    print("✔ Audio libraries loaded successfully")
except Exception as e:
    print("✘ Audio libraries error:", e)


# Phase 2 — Speech to Text
print("\n[Phase 2] Speech-to-Text")

try:
    import whisper
    import whisper_timestamped
    import torch
    from indic_transliteration import sanscript
    print("✔ Whisper + Torch loaded successfully")
    print("Torch version:", torch.__version__)
except Exception as e:
    print("✘ STT libraries error:", e)


# Phase 3 — Analysis
print("\n[Phase 3] NLP Analysis")

try:
    import transformers
    import scipy
    print("✔ Transformers + Scipy loaded successfully")
except Exception as e:
    print("✘ Analysis libraries error:", e)


# Phase 5 — Dashboard + Alerts
print("\n[Phase 5] Dashboard + Alerts")

try:
    import streamlit
    from twilio.rest import Client
    import pygame
    print("✔ Streamlit + Twilio + Pygame loaded successfully")
except Exception as e:
    print("✘ Alert libraries error:", e)


# Phase 6 — Reports
print("\n[Phase 6] Reporting")

try:
    from reportlab.pdfgen import canvas
    import matplotlib.pyplot as plt
    print("✔ Report libraries loaded successfully")
except Exception as e:
    print("✘ Report libraries error:", e)


# Utilities
print("\n[Utilities]")

try:
    from dotenv import load_dotenv
    from tqdm import tqdm
    print("✔ Utility libraries loaded successfully")
except Exception as e:
    print("✘ Utility libraries error:", e)


# Quick Torch test
print("\n[Torch Test]")

try:
    x = torch.tensor([1,2,3])
    print("✔ Torch tensor created:", x)
except Exception as e:
    print("✘ Torch test failed:", e)


print("\n===== TEST COMPLETE =====")