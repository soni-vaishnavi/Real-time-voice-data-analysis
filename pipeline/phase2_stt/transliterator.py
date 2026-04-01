"""
Phase 2 - Step 2: Transliterator
Converts Devanagari script (Hindi) → Roman script
Ensures consistent text format for downstream NLP models

Why needed?
- Whisper sometimes outputs Hindi in Devanagari: "बचाओ"
- Our keyword dict and NLP models expect Roman: "bachao"
- This module normalizes everything to Roman script
"""

import re
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── DEVANAGARI DETECTION ──────────────────────────────────────────────────────

def contains_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script characters"""
    # Devanagari Unicode range: U+0900 to U+097F
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(devanagari_pattern.search(text))


# ── TRANSLITERATION ───────────────────────────────────────────────────────────

def transliterate_to_roman(text: str) -> str:
    """
    Convert Devanagari text to Roman (Latin) script.
    
    Uses indic-transliteration library which handles:
    - All Hindi characters
    - Matras (vowel signs)
    - Conjunct consonants
    - Numerals
    
    Falls back to basic character mapping if library unavailable.
    """
    if not contains_devanagari(text):
        # Already Roman script — return as is
        return text

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        roman_text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        # Clean up ITRANS artifacts (capitalization quirks)
        roman_text = clean_itrans_output(roman_text)
        logger.debug(f"Transliterated: '{text}' → '{roman_text}'")
        return roman_text

    except ImportError:
        logger.warning("indic-transliteration not installed. Using basic mapping.")
        logger.warning("Install with: pip install indic-transliteration")
        return basic_devanagari_to_roman(text)

    except Exception as e:
        logger.warning(f"Transliteration failed for '{text}': {e}. Using original.")
        return text


def clean_itrans_output(text: str) -> str:
    """
    Clean up ITRANS transliteration output.
    ITRANS uses uppercase for some sounds (T, D, N etc.)
    We normalize to readable lowercase Roman.
    """
    # Common ITRANS → readable Roman mappings
    replacements = {
        "aa": "a",   # long 'a' sound → simplified
        "ii": "i",   # long 'i'
        "uu": "u",   # long 'u'
        "sh": "sh",  # keep
        "Sh": "sh",  # retroflex sh → sh
        "ch": "ch",  # keep
        "Ch": "chh", # aspirated ch
        "T":  "t",   # retroflex T → t
        "D":  "d",   # retroflex D → d
        "N":  "n",   # retroflex N → n
        "L":  "l",   # retroflex L → l
        "R":  "r",   # retroflex R → r
        "JN": "gn",  # jña sound
        "kh": "kh",  # keep
        "gh": "gh",  # keep
        "jh": "jh",  # keep
        "~":  "",    # remove anusvara marker
        "H":  "h",   # visarga → h
        ".":  "",    # remove punctuation artifacts
    }

    result = text
    for itrans, roman in replacements.items():
        result = result.replace(itrans, roman)

    return result.lower().strip()


def basic_devanagari_to_roman(text: str) -> str:
    """
    Basic fallback Devanagari → Roman mapping.
    Covers most common Hindi characters used in speech.
    Not perfect but functional for keyword detection.
    """
    char_map = {
        # Vowels
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee',
        'उ': 'u', 'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai',
        'ओ': 'o', 'औ': 'au', 'ऋ': 'ri',

        # Consonants
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'व': 'w',
        'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gn',

        # Matras (vowel signs)
        'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
        'ं': 'n', 'ः': 'h', '्': '',

        # Numerals
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',

        # Common punctuation
        '।': '.', '॥': '.',
    }

    result = ""
    for char in text:
        result += char_map.get(char, char)

    return result.strip()


# ── PROCESS FULL TRANSCRIPT ───────────────────────────────────────────────────

def transliterate_transcript(transcript: Dict) -> Dict:
    """
    Apply transliteration to all text fields in a transcript dict.
    Converts any Devanagari → Roman in text and individual words.
    
    Args:
        transcript: dict from whisper_transcriber
    
    Returns:
        transcript with all text fields in Roman script
    """
    # Transliterate main text fields
    if contains_devanagari(transcript.get("text", "")):
        original = transcript["text"]
        transcript["text"] = transliterate_to_roman(original)
        logger.info(f"Transliterated text: '{original[:50]}' → '{transcript['text'][:50]}'")

    if contains_devanagari(transcript.get("raw_hindi_text", "")):
        transcript["raw_hindi_text"] = transliterate_to_roman(transcript["raw_hindi_text"])

    # Transliterate individual words
    for word_obj in transcript.get("words", []):
        if contains_devanagari(word_obj.get("word", "")):
            word_obj["word"] = transliterate_to_roman(word_obj["word"])
            word_obj["transliterated"] = True
        else:
            word_obj["transliterated"] = False

    transcript["transliteration_applied"] = True
    return transcript


def transliterate_all_transcripts(transcripts: List[Dict]) -> List[Dict]:
    """Apply transliteration to all transcripts in list"""
    logger.info(f"Applying transliteration to {len(transcripts)} transcripts")
    processed = []
    devanagari_count = 0

    for t in transcripts:
        had_devanagari = contains_devanagari(t.get("text", ""))
        result = transliterate_transcript(t)
        if had_devanagari:
            devanagari_count += 1
        processed.append(result)

    logger.info(f"Transliteration complete | {devanagari_count} transcripts had Devanagari")
    return processed