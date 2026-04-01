"""
Phase 6 — Incident Report Generator
Generates a professional PDF for a single RED emergency incident.

Usage:
    from pipeline.phase6_reports.incident_report import generate_incident_report
    path = generate_incident_report(chunk, all_chunks, action="CONFIRMED")
    # returns path to saved PDF e.g. output/reports/incident_INC_001_20260312.pdf
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

# ── OUTPUT DIR ─────────────────────────────────────────────────────────────────
REPORTS_DIR = "output/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── COLORS ─────────────────────────────────────────────────────────────────────
C_RED       = colors.HexColor("#ef4444")
C_RED_LIGHT = colors.HexColor("#fef2f2")
C_RED_BORDER= colors.HexColor("#fca5a5")
C_ORANGE    = colors.HexColor("#f97316")
C_YELLOW    = colors.HexColor("#f59e0b")
C_GREEN     = colors.HexColor("#22c55e")
C_BLUE      = colors.HexColor("#3b82f6")
C_DARK      = colors.HexColor("#1a1f2e")
C_GREY      = colors.HexColor("#6b7280")
C_LIGHT     = colors.HexColor("#f3f4f6")
C_WHITE     = colors.white
C_NAVY      = colors.HexColor("#1e3a5f")

SEV_COLOR = {
    "CRITICAL": C_RED,
    "HIGH":     C_ORANGE,
    "MEDIUM":   C_YELLOW,
    "LOW":      C_BLUE,
}
CAT_LABEL = {
    "medical":      "Medical Emergency",
    "fire":         "Fire / Explosion",
    "violence":     "Violence / Assault",
    "accident":     "Accident",
    "theft":        "Theft / Robbery",
    "mental_health":"Mental Health Crisis",
    "normal":       "Normal Activity",
}

HIGHLIGHT_KEYWORDS = [
    "bachao","ambulance","help","fire","aag","doctor","police","khoon","blood",
    "shoot","gun","knife","chaku","maar","gir","accident","emergency","save",
    "please","attack","loot","robbery","suicide","dard","pain","hospital",
]

# ── STYLES ─────────────────────────────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=22,
            textColor=C_DARK, spaceAfter=4, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName="Helvetica", fontSize=11,
            textColor=C_GREY, spaceAfter=2, alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=12,
            textColor=C_NAVY, spaceBefore=14, spaceAfter=6,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=C_DARK, spaceAfter=4, leading=15,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=8,
            textColor=C_GREY, spaceAfter=2,
        ),
        "mono": ParagraphStyle(
            "mono", fontName="Courier", fontSize=10,
            textColor=C_DARK, spaceAfter=4, leading=15,
        ),
        "mono_red": ParagraphStyle(
            "mono_red", fontName="Courier-Bold", fontSize=10,
            textColor=C_RED, spaceAfter=4, leading=15,
        ),
        "alert_title": ParagraphStyle(
            "alert_title", fontName="Helvetica-Bold", fontSize=16,
            textColor=C_WHITE, spaceAfter=2,
        ),
        "alert_sub": ParagraphStyle(
            "alert_sub", fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#fecaca"), spaceAfter=0,
        ),
        "label": ParagraphStyle(
            "label", fontName="Helvetica-Bold", fontSize=8,
            textColor=C_GREY, spaceAfter=2,
            wordWrap="CJK",
        ),
        "value": ParagraphStyle(
            "value", fontName="Helvetica-Bold", fontSize=13,
            textColor=C_DARK, spaceAfter=0,
        ),
        "transcript": ParagraphStyle(
            "transcript", fontName="Helvetica-Oblique", fontSize=11,
            textColor=C_DARK, leading=18, spaceAfter=4,
            leftIndent=8,
        ),
    }

# ── HELPERS ────────────────────────────────────────────────────────────────────
def _hr(color=None, thickness=1):
    return HRFlowable(
        width="100%", thickness=thickness,
        color=color or C_LIGHT, spaceAfter=6, spaceBefore=6,
    )

def _score_bar_table(score_pct: int, color) -> Table:
    """Render a score bar as a ReportLab table row."""
    filled = max(1, int(score_pct * 1.4))   # max ~140 units wide
    empty  = 140 - filled
    bar_table = Table(
        [["", ""]],
        colWidths=[filled * mm, empty * mm],
        rowHeights=[5 * mm],
    )
    bar_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), color),
        ("BACKGROUND", (1, 0), (1, 0), C_LIGHT),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    return bar_table

def _highlight_text(text: str, keywords: List[str]) -> str:
    """Return text with keywords wrapped in ReportLab red bold tags."""
    if not keywords or not text:
        return text
    kw_set = set(k.lower() for k in keywords)
    words  = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(".,!?\"'")
        if clean in kw_set:
            result.append(f'<font color="#ef4444"><b>[{word}]</b></font>')
        else:
            result.append(word)
    return " ".join(result)

# ── PAGE TEMPLATE ──────────────────────────────────────────────────────────────
def _header_footer(canvas, doc):
    """Draw header and footer on every page."""
    canvas.saveState()
    w, h = A4

    # Header line
    canvas.setStrokeColor(C_NAVY)
    canvas.setLineWidth(2)
    canvas.line(15 * mm, h - 18 * mm, w - 15 * mm, h - 18 * mm)

    # Header text
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(C_NAVY)
    canvas.drawString(15 * mm, h - 14 * mm, "VoiceGuard — Real-Time Voice Surveillance System")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_GREY)
    canvas.drawRightString(w - 15 * mm, h - 14 * mm, "CONFIDENTIAL — FOR AUTHORITY USE ONLY")

    # Footer
    canvas.setStrokeColor(C_LIGHT)
    canvas.setLineWidth(1)
    canvas.line(15 * mm, 18 * mm, w - 15 * mm, 18 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_GREY)
    canvas.drawString(15 * mm, 12 * mm,
        "Poornima University, Jaipur | BCA Final Year Project | Guide: Mr. Hemant Gautam")
    canvas.drawRightString(w - 15 * mm, 12 * mm, f"Page {doc.page}")

    canvas.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_incident_report(
    chunk: Dict,
    all_chunks: List[Dict],
    action: str = "CONFIRMED",
    output_dir: str = REPORTS_DIR,
) -> str:
    """
    Generate a PDF incident report for a single RED emergency chunk.

    Args:
        chunk:      The RED chunk dict from all_decisions.json
        all_chunks: Full list of all chunks (for context window)
        action:     "CONFIRMED", "AUTO", or "MANUAL"
        output_dir: Where to save the PDF

    Returns:
        Absolute path to the generated PDF file.
    """
    s           = chunk.get("score", {})
    iid         = s.get("incident_id", "INC_000")
    category    = s.get("emergency_category", "unknown")
    severity    = s.get("severity", "HIGH")
    final_score = round(s.get("final_score", 0) * 100)
    emotion     = s.get("dominant_emotion", "?")
    text        = chunk.get("text", "")
    chunk_id    = chunk.get("chunk_id", "?")
    chunk_start = round(chunk.get("chunk_start", 0), 1)
    comp        = s.get("components", {})
    keywords    = chunk.get("keywords_found", s.get("keywords_found", []))
    trend_up    = s.get("trend_upgraded", False)
    rising      = s.get("rising_trend", False)
    sev_col     = SEV_COLOR.get(severity, C_RED)
    cat_label   = CAT_LABEL.get(category, category.title())
    now         = datetime.now()

    # Output path
    filename = f"incident_{iid}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    out_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    # Context: 5 chunks before this one
    idx     = next((i for i, c in enumerate(all_chunks)
                    if c.get("chunk_id") == chunk_id), 0)
    context = all_chunks[max(0, idx - 5): idx + 1]

    styles = _styles()
    story  = []

    # ── COVER BLOCK ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))

    # Red alert banner
    banner_data = [[
        Paragraph(f"&#x26A0;  {category.upper()} EMERGENCY", ParagraphStyle("at", fontName="Helvetica-Bold", fontSize=15, textColor=colors.white, spaceAfter=3)),
        Paragraph(f"INCIDENT REPORT  —  Severity: {severity}  |  ID: {iid}", ParagraphStyle("at2", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#fecaca"), spaceAfter=0)),
        Paragraph(f"Incident ID: {iid}  |  Generated: {now.strftime('%d %b %Y, %H:%M:%S')}", styles["alert_sub"]),
    ]]
    banner = Table(banner_data, colWidths=[175 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), sev_col),
        ("ROUNDEDCORNERS",[4, 4, 4, 4]),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    story.append(banner)
    story.append(Spacer(1, 6 * mm))

    # ── INCIDENT SUMMARY TABLE ────────────────────────────────────────────────
    story.append(Paragraph("1. Incident Summary", styles["section"]))
    story.append(_hr(C_NAVY, 1))

    summary_data = [
        [
            Paragraph("INCIDENT ID", styles["label"]),
            Paragraph("CATEGORY", styles["label"]),
            Paragraph("SEVERITY", styles["label"]),
            Paragraph("DETECTION TIME", styles["label"]),
        ],
        [
            Paragraph(iid, styles["value"]),
            Paragraph(cat_label, styles["value"]),
            Paragraph(severity, styles["value"]),
            Paragraph(now.strftime("%H:%M:%S"), styles["value"]),
        ],
        [
            Paragraph("AUDIO CHUNK", styles["label"]),
            Paragraph("CHUNK TIMESTAMP", styles["label"]),
            Paragraph("EMOTION DETECTED", styles["label"]),
            Paragraph("ACTION TAKEN", styles["label"]),
        ],
        [
            Paragraph(chunk_id, styles["value"]),
            Paragraph(f"{chunk_start}s", styles["value"]),
            Paragraph(emotion.upper(), styles["value"]),
            Paragraph(action, styles["value"]),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[43 * mm] * 4)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_WHITE),
        ("BACKGROUND",    (0, 0), (-1, 0), C_LIGHT),
        ("BACKGROUND",    (0, 2), (-1, 2), C_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        # Severity color
        ("TEXTCOLOR",     (2, 1), (2, 1), sev_col),
        ("TEXTCOLOR",     (2, 3), (2, 3),
         C_GREEN if action == "CONFIRMED" else C_RED),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 4 * mm))

    # Flags row
    flags = []
    if trend_up: flags.append("TREND ESCALATION (3 consecutive YELLOW → RED)")
    if rising:   flags.append("RISING SCORE PATTERN detected")
    if flags:
        story.append(Paragraph(
            "<b>&#9888; Flags:</b> " + "  |  ".join(flags),
            ParagraphStyle("flag", fontName="Helvetica", fontSize=9,
                           textColor=C_ORANGE, spaceAfter=4,
                           backColor=colors.HexColor("#fff7ed"),
                           borderPad=6, borderColor=colors.HexColor("#fdba74"),
                           borderWidth=1, borderRadius=4)
        ))
        story.append(Spacer(1, 2 * mm))

    # ── CONFIDENCE SCORE ──────────────────────────────────────────────────────
    story.append(Paragraph("2. Confidence Score Analysis", styles["section"]))
    story.append(_hr(C_NAVY, 1))

    # Big score display
    score_data = [[
        Paragraph(f"<font size='28'><b>{final_score}%</b></font>", ParagraphStyle(
            "big_score", fontName="Helvetica-Bold", fontSize=28,
            textColor=sev_col, alignment=TA_CENTER
        )),
        Table(
            [
                [Paragraph("Component", styles["label"]),
                 Paragraph("Weight", styles["label"]),
                 Paragraph("Score Contribution", styles["label"])],
                [Paragraph("Emotion Detection",    styles["body"]),
                 Paragraph("35%", styles["body"]),
                 Paragraph(f"{round(comp.get('emotion_component',0)*100)}%", styles["body"])],
                [Paragraph("Emergency Detection",  styles["body"]),
                 Paragraph("40%", styles["body"]),
                 Paragraph(f"{round(comp.get('emergency_component',0)*100)}%", styles["body"])],
                [Paragraph("Keyword Boost",        styles["body"]),
                 Paragraph("25%", styles["body"]),
                 Paragraph(f"{round(comp.get('keyword_component',0)*100)}%", styles["body"])],
                [Paragraph("Sarcasm Deduction",    styles["body"]),
                 Paragraph("-",   styles["body"]),
                 Paragraph(f"-{round(comp.get('sarcasm_deduction',0)*100)}%", styles["body"])],
            ],
            colWidths=[50*mm, 22*mm, 45*mm],
            rowHeights=[8*mm, 7*mm, 7*mm, 7*mm, 7*mm],
        ),
    ]]
    score_data[0][1].setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#f3f4f6")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TEXTCOLOR",     (2, 1), (2, 4), C_BLUE),
    ]))

    outer_table = Table(score_data, colWidths=[35*mm, 140*mm])
    outer_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (0, 0), 0),
        ("RIGHTPADDING",  (0, 0), (0, 0), 8),
    ]))
    story.append(outer_table)

    # Visual score bar
    story.append(Spacer(1, 3 * mm))
    # Confidence bar as a two-cell table row
    filled_w = max(2, int(final_score * 1.55))
    empty_w  = 155 - filled_w
    story.append(Paragraph("<b>Confidence Bar:</b>", styles["body"]))
    bar_tbl = Table(
        [["", ""]],
        colWidths=[filled_w * mm, empty_w * mm],
        rowHeights=[8 * mm],
    )
    bar_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), sev_col),
        ("BACKGROUND",    (1, 0), (1, 0), colors.HexColor("#e5e7eb")),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ]))
    story.append(bar_tbl)
    story.append(Paragraph(
        f"  Zone: <b>{'RED (Emergency)' if final_score > 72 else 'YELLOW (Warning)' if final_score > 45 else 'GREEN (Safe)'}</b>"
        f"  |  Threshold crossed: <b>{'>72%' if final_score > 72 else '>45%'}</b>",
        styles["small"]
    ))
    story.append(Spacer(1, 4 * mm))

    # ── FLAGGED TRANSCRIPT ────────────────────────────────────────────────────
    story.append(Paragraph("3. Flagged Transcript", styles["section"]))
    story.append(_hr(C_NAVY, 1))

    hl_text = _highlight_text(text, keywords if keywords else HIGHLIGHT_KEYWORDS)
    transcript_box_data = [[
        Paragraph(f'"{hl_text}"', styles["transcript"])
    ]]
    transcript_box = Table(transcript_box_data, colWidths=[175 * mm])
    transcript_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_RED_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 1, C_RED_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(transcript_box)

    # Keywords
    if keywords:
        story.append(Spacer(1, 3 * mm))
        kw_text = "  ".join(f"[{kw.upper()}]" for kw in keywords)
        story.append(Paragraph(
            f"<b>Emergency Keywords Detected:</b>  "
            f'<font color="#ef4444"><b>{kw_text}</b></font>',
            styles["body"]
        ))
    story.append(Spacer(1, 4 * mm))

    # ── CONTEXT WINDOW ────────────────────────────────────────────────────────
    story.append(Paragraph("4. Context — Recent Audio Chunks", styles["section"]))
    story.append(_hr(C_NAVY, 1))
    story.append(Paragraph(
        "The 5 chunks immediately before the flagged incident:",
        styles["small"]
    ))
    story.append(Spacer(1, 2 * mm))

    ctx_header = [
        Paragraph("Chunk ID", styles["label"]),
        Paragraph("Time", styles["label"]),
        Paragraph("Zone", styles["label"]),
        Paragraph("Score", styles["label"]),
        Paragraph("Emotion", styles["label"]),
        Paragraph("Transcript", styles["label"]),
    ]
    ctx_rows = [ctx_header]
    for c in context:
        cs     = c.get("score", {})
        czone  = cs.get("zone", "GREEN")
        cscore = round(cs.get("final_score", 0) * 100)
        cemo   = cs.get("dominant_emotion", "?")
        ctxt   = (c.get("text") or "—")[:55]
        cid    = c.get("chunk_id", "?")
        cts    = round(c.get("chunk_start", 0), 1)
        is_flagged = c.get("chunk_id") == chunk_id

        zone_col = (C_RED if czone=="RED" else
                    C_YELLOW if czone=="YELLOW" else C_GREEN)
        row = [
            Paragraph(f"<b>{cid}</b>" if is_flagged else cid, styles["small"]),
            Paragraph(f"{cts}s", styles["small"]),
            Paragraph(f"<font color='{zone_col.hexval()}'><b>{czone}</b></font>",
                      styles["small"]),
            Paragraph(f"{cscore}%", styles["small"]),
            Paragraph(cemo, styles["small"]),
            Paragraph(f"{'→ ' if is_flagged else ''}{ctxt}", styles["small"]),
        ]
        ctx_rows.append(row)

    ctx_table = Table(ctx_rows, colWidths=[28*mm, 14*mm, 16*mm, 14*mm, 18*mm, 85*mm])
    ctx_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#f3f4f6")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
    ])
    # Highlight the flagged row
    flagged_row = next((i+1 for i, c in enumerate(context)
                        if c.get("chunk_id") == chunk_id), None)
    if flagged_row:
        ctx_style.add("BACKGROUND", (0, flagged_row), (-1, flagged_row),
                      colors.HexColor("#fef2f2"))
    ctx_table.setStyle(ctx_style)
    story.append(ctx_table)
    story.append(Spacer(1, 4 * mm))

    # ── ALERT ACTIONS ─────────────────────────────────────────────────────────
    story.append(Paragraph("5. Alert Actions Taken", styles["section"]))
    story.append(_hr(C_NAVY, 1))

    actions_data = [
        [Paragraph("Action", styles["label"]),
         Paragraph("Channel", styles["label"]),
         Paragraph("Status", styles["label"]),
         Paragraph("Timestamp", styles["label"])],
        [Paragraph("Emergency Alert", styles["body"]),
         Paragraph("SMS (Twilio)", styles["body"]),
         Paragraph("Sent (Dry-run)", styles["body"]),
         Paragraph(now.strftime("%H:%M:%S"), styles["body"])],
        [Paragraph("Emergency Alert", styles["body"]),
         Paragraph("Email (Gmail SMTP)", styles["body"]),
         Paragraph("Sent (Dry-run)", styles["body"]),
         Paragraph(now.strftime("%H:%M:%S"), styles["body"])],
        [Paragraph("Alarm", styles["body"]),
         Paragraph("Sound (pygame)", styles["body"]),
         Paragraph("Triggered", styles["body"]),
         Paragraph(now.strftime("%H:%M:%S"), styles["body"])],
        [Paragraph(f"Decision", styles["body"]),
         Paragraph("Authority Dashboard", styles["body"]),
         Paragraph(f"<b>{action}</b>", styles["body"]),
         Paragraph(now.strftime("%H:%M:%S"), styles["body"])],
    ]
    actions_table = Table(actions_data, colWidths=[45*mm, 50*mm, 45*mm, 35*mm])
    actions_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#f3f4f6")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",     (2, 4), (2, 4), C_GREEN),
    ]))
    story.append(actions_table)
    story.append(Spacer(1, 6 * mm))

    # ── FOOTER NOTE ───────────────────────────────────────────────────────────
    story.append(_hr(C_GREY, 0.5))
    story.append(Paragraph(
        f"This report was automatically generated by VoiceGuard on "
        f"{now.strftime('%d %B %Y at %H:%M:%S')}. "
        f"Report ID: {iid}-{now.strftime('%Y%m%d%H%M%S')}. "
        f"This document is confidential and intended for authorized personnel only.",
        styles["small"]
    ))

    # ── BUILD PDF ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        title=f"Incident Report — {iid}",
        author="VoiceGuard System",
        subject=f"{cat_label} Emergency",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    return os.path.abspath(out_path)


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load real Phase 4 data for test
    decisions_path = "output/decisions/all_decisions.json"
    if os.path.exists(decisions_path):
        with open(decisions_path) as f:
            all_chunks = json.load(f)
        red_chunks = [c for c in all_chunks if c.get("score",{}).get("zone")=="RED"]
        if red_chunks:
            path = generate_incident_report(red_chunks[0], all_chunks, action="CONFIRMED")
            print(f"[OK] Incident report saved: {path}")
        else:
            print("[WARN] No RED chunks found in data")
    else:
        print(f"[ERROR] {decisions_path} not found — run test_phase4.py first")