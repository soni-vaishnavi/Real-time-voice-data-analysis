"""
Phase 6 — Session Report Generator
Generates a full session summary PDF covering all analyzed chunks.

Usage:
    from pipeline.phase6_reports.session_report import generate_session_report
    path = generate_session_report(all_chunks, alert_log)
    # returns path e.g. output/reports/session_report_20260312_102345.pdf
"""

import os
import io
import json
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether,
)

# ── OUTPUT DIR ─────────────────────────────────────────────────────────────────
REPORTS_DIR = "output/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── COLORS ─────────────────────────────────────────────────────────────────────
C_RED    = colors.HexColor("#ef4444")
C_ORANGE = colors.HexColor("#f97316")
C_YELLOW = colors.HexColor("#f59e0b")
C_GREEN  = colors.HexColor("#22c55e")
C_BLUE   = colors.HexColor("#3b82f6")
C_NAVY   = colors.HexColor("#1e3a5f")
C_DARK   = colors.HexColor("#1a1f2e")
C_GREY   = colors.HexColor("#6b7280")
C_LIGHT  = colors.HexColor("#f3f4f6")
C_WHITE  = colors.white

ZONE_COLOR = {"RED": C_RED, "YELLOW": C_YELLOW, "GREEN": C_GREEN}
SEV_COLOR  = {"CRITICAL": C_RED, "HIGH": C_ORANGE, "MEDIUM": C_YELLOW, "LOW": C_BLUE}
EMO_COLOR  = {
    "fear":    C_RED,   "anger":   C_ORANGE, "disgust": colors.HexColor("#a855f7"),
    "sadness": C_BLUE,  "surprise":C_YELLOW, "joy":     C_GREEN, "neutral": C_GREY,
}
CAT_LABEL = {
    "medical":"Medical Emergency","fire":"Fire / Explosion",
    "violence":"Violence / Assault","accident":"Accident",
    "theft":"Theft / Robbery","mental_health":"Mental Health Crisis","normal":"Normal",
}

# ── STYLES ─────────────────────────────────────────────────────────────────────
def _styles():
    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontName="Helvetica-Bold", fontSize=28,
            textColor=C_WHITE, spaceAfter=6, alignment=TA_CENTER, leading=34,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontName="Helvetica", fontSize=13,
            textColor=colors.HexColor("#bfdbfe"), spaceAfter=4, alignment=TA_CENTER,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#93c5fd"), spaceAfter=3, alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=13,
            textColor=C_NAVY, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=C_DARK, spaceAfter=4, leading=15,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=8,
            textColor=C_GREY, spaceAfter=2,
        ),
        "label": ParagraphStyle(
            "label", fontName="Helvetica-Bold", fontSize=8,
            textColor=C_GREY, spaceAfter=1,
        ),
        "value": ParagraphStyle(
            "value", fontName="Helvetica-Bold", fontSize=14,
            textColor=C_DARK, spaceAfter=0,
        ),
        "transcript_small": ParagraphStyle(
            "transcript_small", fontName="Helvetica-Oblique", fontSize=8,
            textColor=C_GREY, leading=12,
        ),
    }

def _hr(color=None, thickness=1):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color or C_LIGHT, spaceAfter=6, spaceBefore=4)

# ── HEADER / FOOTER ────────────────────────────────────────────────────────────
def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setStrokeColor(C_NAVY)
        canvas.setLineWidth(1.5)
        canvas.line(15*mm, h-18*mm, w-15*mm, h-18*mm)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(C_NAVY)
        canvas.drawString(15*mm, h-14*mm, "VoiceGuard — Session Analysis Report")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_GREY)
        canvas.drawRightString(w-15*mm, h-14*mm, "CONFIDENTIAL")

    canvas.setStrokeColor(C_LIGHT)
    canvas.setLineWidth(1)
    canvas.line(15*mm, 18*mm, w-15*mm, 18*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_GREY)
    canvas.drawString(15*mm, 12*mm,
        "Poornima University, Jaipur  |  BCA Final Year Project  |  Guide: Mr. Hemant Gautam")
    canvas.drawRightString(w-15*mm, 12*mm, f"Page {doc.page}")
    canvas.restoreState()


# ── COVER PAGE ─────────────────────────────────────────────────────────────────
def _cover_page(story, now, stats, styles):
    # Navy cover block
    cover_data = [[
        Paragraph("VoiceGuard", ParagraphStyle(
            "vg", fontName="Helvetica-Bold", fontSize=36,
            textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=2)),
        Paragraph("Real-Time Voice Surveillance System", styles["cover_sub"]),
        Paragraph("SESSION ANALYSIS REPORT", ParagraphStyle(
            "sr", fontName="Helvetica-Bold", fontSize=16,
            textColor=colors.HexColor("#bfdbfe"), alignment=TA_CENTER, spaceAfter=8)),
        Spacer(1, 6*mm),
        Paragraph(f"Generated: {now.strftime('%d %B %Y  |  %H:%M:%S')}", styles["cover_meta"]),
        Spacer(1, 3*mm),
        Paragraph("Poornima University, Jaipur", styles["cover_meta"]),
        Paragraph("BCA Final Year Project — Batch 2024–2025", styles["cover_meta"]),
        Paragraph("Guide: Mr. Hemant Gautam", styles["cover_meta"]),
        Spacer(1, 6*mm),
        Paragraph("Team Members:", ParagraphStyle(
            "tm", fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#93c5fd"), alignment=TA_CENTER)),
        Paragraph("Nishant Rakhecha [13411]  |  Vaishnavi Soni [13554]  |  Manyata Gupta [14048]",
                  styles["cover_meta"]),
    ]]
    cover_table = Table([[item] for item in cover_data[0]], colWidths=[175*mm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 8*mm))

    # Quick stats row on cover
    stat_items = [
        ("TOTAL CHUNKS", str(stats["total"]),           C_DARK),
        ("SAFE (GREEN)", str(stats["green"]),            C_GREEN),
        ("WARNING",      str(stats["yellow"]),           C_YELLOW),
        ("EMERGENCIES",  str(stats["red"]),              C_RED),
        ("ALERTS FIRED", str(stats["alerts_fired"]),     C_ORANGE),
        ("AVG SCORE",    f"{stats['avg_score']}%",       C_BLUE),
    ]
    stat_data  = [[Paragraph(lbl, styles["label"]) for lbl,_,_ in stat_items]]
    stat_data += [[Paragraph(f'<font color="{c.hexval()}">{val}</font>',
                              ParagraphStyle("sv", fontName="Helvetica-Bold",
                                             fontSize=20, alignment=TA_CENTER))
                   for _, val, c in stat_items]]
    stat_table = Table(stat_data, colWidths=[29*mm]*6)
    stat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_WHITE),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("BACKGROUND",    (0,0),(-1,0), C_LIGHT),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(stat_table)
    story.append(PageBreak())


# ── SCORE TIMELINE CHART ───────────────────────────────────────────────────────
def _score_timeline_image(chunks) -> Optional[bytes]:
    """Generate a matplotlib score timeline PNG, return as bytes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        scores = [c.get("score",{}).get("final_score",0)*100 for c in chunks]
        zones  = [c.get("score",{}).get("zone","GREEN") for c in chunks]
        x      = list(range(len(scores)))

        color_map = {"RED":"#ef4444","YELLOW":"#f59e0b","GREEN":"#22c55e"}
        bar_colors = [color_map.get(z,"#22c55e") for z in zones]

        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
        fig.patch.set_facecolor("#f9fafb")
        ax.set_facecolor("#f9fafb")

        ax.bar(x, scores, color=bar_colors, width=0.8, alpha=0.85)
        ax.axhline(72, color="#ef4444", linestyle="--", linewidth=1, alpha=0.6, label="RED threshold (72%)")
        ax.axhline(45, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.6, label="YELLOW threshold (45%)")
        ax.set_xlabel("Chunk Index", fontsize=9, color="#6b7280")
        ax.set_ylabel("Score (%)", fontsize=9, color="#6b7280")
        ax.set_ylim(0, 110)
        ax.tick_params(labelsize=8, colors="#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e5e7eb")

        patches = [
            mpatches.Patch(color="#22c55e", label="GREEN (Safe)"),
            mpatches.Patch(color="#f59e0b", label="YELLOW (Warning)"),
            mpatches.Patch(color="#ef4444", label="RED (Emergency)"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  facecolor="#f9fafb", edgecolor="#e5e7eb")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[WARN] Could not generate chart: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_session_report(
    all_chunks: List[Dict],
    alert_log:  List[Dict] = None,
    output_dir: str = REPORTS_DIR,
) -> str:
    """
    Generate a full session PDF report.

    Args:
        all_chunks: All chunk dicts from all_decisions.json
        alert_log:  List of alert entries from dashboard session state
        output_dir: Where to save the PDF

    Returns:
        Absolute path to the generated PDF.
    """
    alert_log  = alert_log or []
    now        = datetime.now()
    os.makedirs(output_dir, exist_ok=True)

    # ── Compute stats ──────────────────────────────────────────────────────────
    total  = len(all_chunks)
    green  = sum(1 for c in all_chunks if c.get("score",{}).get("zone")=="GREEN")
    yellow = sum(1 for c in all_chunks if c.get("score",{}).get("zone")=="YELLOW")
    red    = sum(1 for c in all_chunks if c.get("score",{}).get("zone")=="RED")
    avg_sc = round(sum(c.get("score",{}).get("final_score",0) for c in all_chunks)/max(1,total)*100,1)

    emo_counter = Counter(c.get("score",{}).get("dominant_emotion","neutral") for c in all_chunks)
    cat_counter = Counter(c.get("score",{}).get("emergency_category","normal") for c in all_chunks)

    red_chunks    = [c for c in all_chunks if c.get("score",{}).get("zone")=="RED"]
    yellow_chunks = [c for c in all_chunks if c.get("score",{}).get("zone")=="YELLOW"]

    stats = {
        "total":total,"green":green,"yellow":yellow,"red":red,
        "avg_score":avg_sc,"alerts_fired":len(alert_log),
    }

    filename = f"session_report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    out_path = os.path.join(output_dir, filename)

    styles = _styles()
    story  = []

    # ── COVER ──────────────────────────────────────────────────────────────────
    _cover_page(story, now, stats, styles)

    # ══ PAGE 2: ZONE + EMOTION ANALYSIS ═══════════════════════════════════════

    story.append(Paragraph("1. Zone Distribution Analysis", styles["section"]))
    story.append(_hr(C_NAVY))

    # Zone summary table
    zone_data = [
        [Paragraph("Zone", styles["label"]),
         Paragraph("Count", styles["label"]),
         Paragraph("Percentage", styles["label"]),
         Paragraph("Description", styles["label"])],
        [Paragraph('<font color="#22c55e"><b>GREEN</b></font>', styles["body"]),
         Paragraph(str(green), styles["body"]),
         Paragraph(f"{round(green/max(1,total)*100)}%", styles["body"]),
         Paragraph("Safe — no emergency activity detected", styles["body"])],
        [Paragraph('<font color="#f59e0b"><b>YELLOW</b></font>', styles["body"]),
         Paragraph(str(yellow), styles["body"]),
         Paragraph(f"{round(yellow/max(1,total)*100)}%", styles["body"]),
         Paragraph("Suspicious — elevated emotion/keywords, monitor closely", styles["body"])],
        [Paragraph('<font color="#ef4444"><b>RED</b></font>', styles["body"]),
         Paragraph(str(red), styles["body"]),
         Paragraph(f"{round(red/max(1,total)*100)}%", styles["body"]),
         Paragraph("Emergency — immediate authority action required", styles["body"])],
    ]
    zone_table = Table(zone_data, colWidths=[30*mm, 25*mm, 30*mm, 90*mm])
    zone_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ]))
    story.append(zone_table)
    story.append(Spacer(1, 6*mm))

    # Emotion analysis
    story.append(Paragraph("2. Emotion Distribution", styles["section"]))
    story.append(_hr(C_NAVY))

    emo_rows = [[
        Paragraph("Emotion", styles["label"]),
        Paragraph("Count", styles["label"]),
        Paragraph("% of Chunks", styles["label"]),
        Paragraph("Visual", styles["label"]),
    ]]
    for emo, cnt in emo_counter.most_common():
        pct     = round(cnt / max(1, total) * 100)
        ecol    = EMO_COLOR.get(emo, C_GREY)
        bar_w   = max(1, int(pct * 0.8))
        bar_cell= Table([[""]], colWidths=[bar_w*mm], rowHeights=[5*mm])
        bar_cell.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), ecol),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ]))
        emo_rows.append([
            Paragraph(emo.capitalize(), styles["body"]),
            Paragraph(str(cnt), styles["body"]),
            Paragraph(f"{pct}%", styles["body"]),
            bar_cell,
        ])

    emo_table = Table(emo_rows, colWidths=[35*mm, 25*mm, 30*mm, 85*mm])
    emo_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ]))
    story.append(emo_table)
    story.append(Spacer(1, 6*mm))

    # Category breakdown
    story.append(Paragraph("3. Emergency Category Breakdown", styles["section"]))
    story.append(_hr(C_NAVY))

    cat_rows = [[
        Paragraph("Category", styles["label"]),
        Paragraph("Count", styles["label"]),
        Paragraph("% of Chunks", styles["label"]),
    ]]
    for cat, cnt in cat_counter.most_common():
        pct = round(cnt / max(1,total) * 100)
        cat_rows.append([
            Paragraph(CAT_LABEL.get(cat, cat.title()), styles["body"]),
            Paragraph(str(cnt), styles["body"]),
            Paragraph(f"{pct}%", styles["body"]),
        ])
    cat_table = Table(cat_rows, colWidths=[70*mm, 30*mm, 35*mm])
    cat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ]))
    story.append(cat_table)

    # ══ PAGE 3: SCORE TIMELINE CHART ══════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Score Timeline", styles["section"]))
    story.append(_hr(C_NAVY))
    story.append(Paragraph(
        f"Combined detection score for all {total} audio chunks. "
        f"Dashed lines show zone thresholds (RED: >72%, YELLOW: >45%).",
        styles["body"]
    ))
    story.append(Spacer(1, 3*mm))

    chart_bytes = _score_timeline_image(all_chunks)
    if chart_bytes:
        img_buf = io.BytesIO(chart_bytes)
        img = Image(img_buf, width=175*mm, height=50*mm)
        story.append(img)
    else:
        story.append(Paragraph(
            "[Chart not available — install matplotlib: pip install matplotlib]",
            styles["small"]
        ))
    story.append(Spacer(1, 6*mm))

    # ══ PAGE 3 cont: INCIDENT TABLE ═══════════════════════════════════════════
    story.append(Paragraph("5. Flagged Incidents (RED + YELLOW)", styles["section"]))
    story.append(_hr(C_NAVY))

    flagged = red_chunks + yellow_chunks
    if flagged:
        inc_rows = [[
            Paragraph("Chunk ID", styles["label"]),
            Paragraph("Time", styles["label"]),
            Paragraph("Zone", styles["label"]),
            Paragraph("Score", styles["label"]),
            Paragraph("Category", styles["label"]),
            Paragraph("Emotion", styles["label"]),
            Paragraph("Incident", styles["label"]),
            Paragraph("Transcript", styles["label"]),
        ]]
        for c in flagged:
            s     = c.get("score",{})
            zone  = s.get("zone","GREEN")
            zcol  = ZONE_COLOR.get(zone, C_GREEN)
            score = round(s.get("final_score",0)*100)
            cat   = CAT_LABEL.get(s.get("emergency_category","?"),"?")[:12]
            emo   = s.get("dominant_emotion","?")
            iid   = s.get("incident_id","—")
            txt   = (c.get("text") or "—")[:50]
            ts    = round(c.get("chunk_start",0),1)
            inc_rows.append([
                Paragraph(c.get("chunk_id","?"), styles["small"]),
                Paragraph(f"{ts}s", styles["small"]),
                Paragraph(f'<font color="{zcol.hexval()}">{zone}</font>', styles["small"]),
                Paragraph(f"{score}%", styles["small"]),
                Paragraph(cat, styles["small"]),
                Paragraph(emo, styles["small"]),
                Paragraph(iid, styles["small"]),
                Paragraph(txt, styles["transcript_small"]),
            ])
        inc_table = Table(inc_rows, colWidths=[26*mm,14*mm,16*mm,14*mm,22*mm,16*mm,16*mm,51*mm])
        inc_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
            ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
            ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, colors.HexColor("#fef2f2")]),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ]))
        story.append(inc_table)
    else:
        story.append(Paragraph("No flagged incidents in this session.", styles["body"]))

    # ══ PAGE 4: ALERT LOG + FULL TRANSCRIPT ═══════════════════════════════════
    story.append(PageBreak())

    # Alert log
    story.append(Paragraph("6. Alert Action Log", styles["section"]))
    story.append(_hr(C_NAVY))

    if alert_log:
        log_rows = [[
            Paragraph("Time", styles["label"]),
            Paragraph("Incident", styles["label"]),
            Paragraph("Category", styles["label"]),
            Paragraph("Score", styles["label"]),
            Paragraph("Severity", styles["label"]),
            Paragraph("Trigger", styles["label"]),
            Paragraph("SMS", styles["label"]),
            Paragraph("Email", styles["label"]),
        ]]
        for e in alert_log:
            sc = SEV_COLOR.get(e.get("severity","HIGH"), C_ORANGE)
            log_rows.append([
                Paragraph(e.get("time","?"), styles["small"]),
                Paragraph(e.get("incident","?"), styles["small"]),
                Paragraph(e.get("category","?"), styles["small"]),
                Paragraph(f"{e.get('score','?')}%", styles["small"]),
                Paragraph(e.get("severity","?"), styles["small"]),
                Paragraph(e.get("reason","?"), styles["small"]),
                Paragraph("Yes" if e.get("sms_sent") else "No", styles["small"]),
                Paragraph("Yes" if e.get("email_sent") else "No", styles["small"]),
            ])
        log_table = Table(log_rows, colWidths=[20*mm,22*mm,30*mm,18*mm,20*mm,20*mm,18*mm,27*mm])
        log_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
            ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
            ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ]))
        story.append(log_table)
    else:
        story.append(Paragraph("No alerts fired during this session.", styles["body"]))

    story.append(Spacer(1, 6*mm))

    # Full transcript log (last 30 chunks)
    story.append(Paragraph("7. Complete Transcript Log (most recent 30 chunks)", styles["section"]))
    story.append(_hr(C_NAVY))

    tx_rows = [[
        Paragraph("Chunk", styles["label"]),
        Paragraph("Time", styles["label"]),
        Paragraph("Zone", styles["label"]),
        Paragraph("Score", styles["label"]),
        Paragraph("Transcript", styles["label"]),
    ]]
    for c in all_chunks[-30:]:
        s     = c.get("score",{})
        zone  = s.get("zone","GREEN")
        zcol  = ZONE_COLOR.get(zone, C_GREEN)
        score = round(s.get("final_score",0)*100)
        txt   = (c.get("text") or "—")[:80]
        ts    = round(c.get("chunk_start",0),1)
        tx_rows.append([
            Paragraph(c.get("chunk_id","?"), styles["small"]),
            Paragraph(f"{ts}s", styles["small"]),
            Paragraph(f'<font color="{zcol.hexval()}">{zone}</font>', styles["small"]),
            Paragraph(f"{score}%", styles["small"]),
            Paragraph(txt, styles["transcript_small"]),
        ])
    tx_table = Table(tx_rows, colWidths=[28*mm, 16*mm, 20*mm, 16*mm, 95*mm])
    tx_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("BOX",           (0,0),(-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, colors.HexColor("#f3f4f6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    story.append(tx_table)
    story.append(Spacer(1, 6*mm))

    # Final note
    story.append(_hr(C_GREY, 0.5))
    story.append(Paragraph(
        f"End of Session Report  |  Generated: {now.strftime('%d %B %Y at %H:%M:%S')}  |  "
        f"Total duration analysed: {round(all_chunks[-1].get('chunk_end', all_chunks[-1].get('chunk_start',0)),1)}s  |  "
        f"System: VoiceGuard v1.0",
        styles["small"]
    ))

    # ── BUILD ──────────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=25*mm, bottomMargin=25*mm,
        leftMargin=15*mm, rightMargin=15*mm,
        title="VoiceGuard Session Report",
        author="VoiceGuard System",
        subject="Voice Surveillance Session Analysis",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return os.path.abspath(out_path)


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decisions_path = "output/decisions/all_decisions.json"
    if os.path.exists(decisions_path):
        with open(decisions_path) as f:
            all_chunks = json.load(f)
        path = generate_session_report(all_chunks, alert_log=[])
        print(f"[OK] Session report saved: {path}")
    else:
        print(f"[ERROR] {decisions_path} not found — run test_phase4.py first")