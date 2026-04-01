"""
Phase 2 - Step 3: Keyword Normalizer
Maps all variations of emergency words (Hindi, English, Hinglish)
to standard normalized categories.

Why needed?
- "bachao", "bacha lo", "bachaoo" all mean HELP
- "aag lagi", "aag hai", "fire" all mean FIRE
- Models need consistent input — not raw variations
"""

import re
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── KEYWORD DICTIONARY ────────────────────────────────────────────────────────
# Format: "variation": ("CATEGORY", confidence_boost)
# confidence_boost: how much to add to emergency score when this word found
# Critical words (bachao, fire) = high boost
# Supporting words (doctor, help) = medium boost

EMERGENCY_KEYWORDS = {

    # ── HELP / DISTRESS ───────────────────────────────────────
    "bachao":         ("HELP", 0.30),
    "bacha lo":       ("HELP", 0.30),
    "bacha":          ("HELP", 0.20),
    "bachaao":        ("HELP", 0.30),
    "madad":          ("HELP", 0.25),
    "madad karo":     ("HELP", 0.28),
    "madad kijiye":   ("HELP", 0.28),
    "help":           ("HELP", 0.25),
    "help me":        ("HELP", 0.28),
    "help karo":      ("HELP", 0.28),
    "help chahiye":   ("HELP", 0.25),
    "please help":    ("HELP", 0.28),
    "koi hai":        ("HELP", 0.15),
    "koi to aao":     ("HELP", 0.20),
    "save me":        ("HELP", 0.28),
    "rescue":         ("HELP", 0.22),

    # ── MEDICAL EMERGENCY ─────────────────────────────────────
    "ambulance":      ("MEDICAL", 0.30),
    "ambulance bulao":("MEDICAL", 0.32),
    "doctor":         ("MEDICAL", 0.22),
    "doctor bulao":   ("MEDICAL", 0.28),
    "doctor chahiye": ("MEDICAL", 0.25),
    "hospital":       ("MEDICAL", 0.20),
    "hospital le jao":("MEDICAL", 0.28),
    "nurse":          ("MEDICAL", 0.18),
    "dawai":          ("MEDICAL", 0.18),
    "dawa":           ("MEDICAL", 0.15),
    "injection":      ("MEDICAL", 0.15),
    "behosh":         ("MEDICAL", 0.25),
    "behosh ho gaya": ("MEDICAL", 0.30),
    "gir gaya":       ("MEDICAL", 0.20),
    "gir gayi":       ("MEDICAL", 0.20),
    "faint":          ("MEDICAL", 0.25),
    "unconscious":    ("MEDICAL", 0.28),
    "breathing":      ("MEDICAL", 0.18),
    "saans nahi":     ("MEDICAL", 0.30),
    "dil":            ("MEDICAL", 0.10),
    "heart attack":   ("MEDICAL", 0.32),
    "khoon":          ("MEDICAL", 0.25),
    "blood":          ("MEDICAL", 0.22),
    "bleeding":       ("MEDICAL", 0.28),
    "dard":           ("MEDICAL", 0.15),
    "pain":           ("MEDICAL", 0.12),
    "chest pain":     ("MEDICAL", 0.28),
    "seene mein dard":("MEDICAL", 0.30),
    "stroke":         ("MEDICAL", 0.30),

    # ── FIRE EMERGENCY ────────────────────────────────────────
    "aag":            ("FIRE", 0.28),
    "aag lagi":       ("FIRE", 0.35),
    "aag lag gayi":   ("FIRE", 0.35),
    "aag lagi hai":   ("FIRE", 0.35),
    "fire":           ("FIRE", 0.28),
    "fire laga":      ("FIRE", 0.32),
    "fire hai":       ("FIRE", 0.32),
    "jal raha":       ("FIRE", 0.25),
    "jal rahi":       ("FIRE", 0.25),
    "dhuan":          ("FIRE", 0.22),
    "smoke":          ("FIRE", 0.22),
    "burn":           ("FIRE", 0.20),
    "jalao mat":      ("FIRE", 0.28),
    "bomb":           ("FIRE", 0.30),
    "blast":          ("FIRE", 0.32),
    "explosion":      ("FIRE", 0.32),
    "gas leak":       ("FIRE", 0.28),

    # ── VIOLENCE / ASSAULT ────────────────────────────────────
    "maro mat":       ("VIOLENCE", 0.30),
    "mat maaro":      ("VIOLENCE", 0.30),
    "maar diya":      ("VIOLENCE", 0.32),
    "maar rahe":      ("VIOLENCE", 0.28),
    "pitai":          ("VIOLENCE", 0.25),
    "maar":           ("VIOLENCE", 0.15),
    "attack":         ("VIOLENCE", 0.28),
    "attack kar":     ("VIOLENCE", 0.30),
    "chaku":          ("VIOLENCE", 0.32),
    "knife":          ("VIOLENCE", 0.30),
    "gun":            ("VIOLENCE", 0.32),
    "pistol":         ("VIOLENCE", 0.32),
    "goli":           ("VIOLENCE", 0.35),
    "shoot":          ("VIOLENCE", 0.35),
    "shooting":       ("VIOLENCE", 0.35),
    "fight":          ("VIOLENCE", 0.15),
    "ladai":          ("VIOLENCE", 0.15),
    "jhagda":         ("VIOLENCE", 0.12),
    "dhamki":         ("VIOLENCE", 0.20),
    "threat":         ("VIOLENCE", 0.20),
    "rape":           ("VIOLENCE", 0.35),
    "assault":        ("VIOLENCE", 0.32),
    "chhoddo":        ("VIOLENCE", 0.22),
    "chhoddo mujhe":  ("VIOLENCE", 0.28),
    "hatao":          ("VIOLENCE", 0.15),
    "chhodo":         ("VIOLENCE", 0.15),

    # ── ACCIDENT ──────────────────────────────────────────────
    "accident":       ("ACCIDENT", 0.30),
    "accident hua":   ("ACCIDENT", 0.32),
    "takkar":         ("ACCIDENT", 0.25),
    "takkar lagi":    ("ACCIDENT", 0.28),
    "gaadi":          ("ACCIDENT", 0.10),
    "crash":          ("ACCIDENT", 0.28),
    "toot gaya":      ("ACCIDENT", 0.15),
    "girne wala":     ("ACCIDENT", 0.20),

    # ── THEFT / ROBBERY ───────────────────────────────────────
    "chor":           ("THEFT", 0.25),
    "chori":          ("THEFT", 0.22),
    "loot":           ("THEFT", 0.28),
    "loot liya":      ("THEFT", 0.30),
    "purse chori":    ("THEFT", 0.28),
    "wallet chori":   ("THEFT", 0.28),
    "robbery":        ("THEFT", 0.30),
    "snatch":         ("THEFT", 0.25),

    # ── MENTAL HEALTH CRISIS ──────────────────────────────────
    "suicide":        ("MENTAL", 0.35),
    "khatam karna":   ("MENTAL", 0.30),
    "jeena nahi":     ("MENTAL", 0.32),
    "mar jaaunga":    ("MENTAL", 0.25),
    "mar jaungi":     ("MENTAL", 0.25),
    "chhod do mujhe": ("MENTAL", 0.22),
    "kuch nahi bachha":("MENTAL", 0.25),
    "end it":         ("MENTAL", 0.28),
    "jump":           ("MENTAL", 0.15),
}


# ── KEYWORD DETECTION ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Clean text for keyword matching — lowercase, remove punctuation"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text)        # normalize spaces
    return text


def detect_keywords(text: str) -> List[Dict]:
    """
    Scan transcript text for emergency keywords.
    Returns all matched keywords with their category and boost value.
    
    Uses longest-match strategy:
    "ambulance bulao" matched before "ambulance" alone
    (more specific = more confident detection)
    
    Args:
        text: normalized transcript text (Roman script)
    
    Returns:
        List of {keyword, category, boost, position}
    """
    normalized = normalize_text(text)
    found_keywords = []
    used_positions = set()  # track positions to avoid double-counting

    # Sort keywords by length (longest first — greedy/longest match)
    sorted_keywords = sorted(EMERGENCY_KEYWORDS.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        category, boost = EMERGENCY_KEYWORDS[keyword]
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

        for match in pattern.finditer(normalized):
            start, end = match.start(), match.end()

            # Check if this position already used by longer match
            position_range = set(range(start, end))
            if position_range & used_positions:
                continue  # skip — already matched by longer keyword

            found_keywords.append({
                "keyword": keyword,
                "category": category,
                "boost": boost,
                "position": start,
                "matched_text": match.group()
            })
            used_positions.update(position_range)

    # Sort by position (order of appearance in text)
    found_keywords.sort(key=lambda x: x["position"])

    return found_keywords


def get_keyword_summary(keywords_found: List[Dict]) -> Dict:
    """
    Summarize keyword detection results.
    
    Returns:
        {
            categories_found: list of unique emergency categories
            total_boost: sum of all keyword boosts (capped at 0.40)
            top_category: most frequently detected category
            keywords_list: simple list of keyword strings
        }
    """
    if not keywords_found:
        return {
            "categories_found": [],
            "total_boost": 0.0,
            "top_category": None,
            "keywords_list": []
        }

    categories = [k["category"] for k in keywords_found]
    unique_categories = list(set(categories))

    # Count category frequency
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    top_category = max(category_counts, key=category_counts.get)

    # Sum boosts but cap at 0.40 to prevent keyword flooding
    total_boost = min(sum(k["boost"] for k in keywords_found), 0.40)

    return {
        "categories_found": unique_categories,
        "total_boost": round(total_boost, 3),
        "top_category": top_category,
        "keywords_list": [k["keyword"] for k in keywords_found],
        "keyword_details": keywords_found
    }


def apply_keyword_normalization(transcript: Dict) -> Dict:
    """
    Apply keyword detection to a transcript dict.
    Adds keyword analysis results directly to transcript.
    
    Args:
        transcript: dict from whisper_transcriber (after transliteration)
    
    Returns:
        transcript with keyword_analysis added
    """
    text = transcript.get("text", "")

    keywords_found = detect_keywords(text)
    summary = get_keyword_summary(keywords_found)

    transcript["keyword_analysis"] = {
        "keywords_found": keywords_found,
        "categories_found": summary["categories_found"],
        "total_boost": summary["total_boost"],
        "top_category": summary["top_category"],
        "keywords_list": summary["keywords_list"]
    }

    if keywords_found:
        logger.info(
            f"{transcript.get('chunk_id', '?')} | "
            f"Keywords: {summary['keywords_list']} | "
            f"Category: {summary['top_category']} | "
            f"Boost: +{summary['total_boost']}"
        )

    return transcript


def apply_keyword_normalization_all(transcripts: List[Dict]) -> List[Dict]:
    """Apply keyword normalization to all transcripts"""
    logger.info(f"Applying keyword normalization to {len(transcripts)} transcripts")

    processed = []
    flagged_count = 0

    for t in transcripts:
        result = apply_keyword_normalization(t)
        if result["keyword_analysis"]["keywords_found"]:
            flagged_count += 1
        processed.append(result)

    logger.info(f"Keyword normalization complete | {flagged_count} chunks had emergency keywords")
    return processed