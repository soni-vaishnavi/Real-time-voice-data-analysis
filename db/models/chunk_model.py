"""
db/models/chunk_model.py
==========================
One row per analyzed audio chunk.
Stores complete analysis results — scores, emotion, category, keywords.
"""

import json
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text

from db.database import Base


class Chunk(Base):
    __tablename__ = "chunks"

    # Identity
    id               = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id         = Column(String, unique=True, index=True, nullable=False)
    session_id       = Column(String, index=True, nullable=False)
    chunk_index      = Column(Integer, default=0)
    chunk_start_sec  = Column(Float, default=0.0)
    audio_path       = Column(String, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)

    # Transcript
    transcript       = Column(Text, nullable=True)
    language_mix     = Column(String, nullable=True)

    # Score
    final_score      = Column(Float, default=0.0, index=True)
    zone             = Column(String, default="GREEN", index=True)
    severity         = Column(String, default="LOW")
    auto_alert       = Column(Boolean, default=False)

    # Analysis details
    dominant_emotion    = Column(String, nullable=True)
    emergency_category  = Column(String, nullable=True)
    is_emergency        = Column(Boolean, default=False)
    incident_id         = Column(String, nullable=True, index=True)

    # Score components
    emotion_component   = Column(Float, default=0.0)
    emergency_component = Column(Float, default=0.0)
    keyword_component   = Column(Float, default=0.0)
    sarcasm_deduction   = Column(Float, default=0.0)

    # Keyword info
    keywords_found      = Column(Text, nullable=True)   # JSON list stored as text
    keyword_boost       = Column(Float, default=0.0)

    # Metadata
    bart_used           = Column(Boolean, default=False)
    processing_ms       = Column(Integer, default=0)
    trend_upgraded      = Column(Boolean, default=False)
    rising_trend        = Column(Boolean, default=False)

    def get_keywords(self):
        """Return keywords_found as Python list."""
        try:
            return json.loads(self.keywords_found or "[]")
        except Exception:
            return []

    def to_dict(self):
        """Return a dict matching the worker result format for API serialization."""
        return {
            "chunk_id":      self.chunk_id,
            "session_id":    self.session_id,
            "chunk_index":   self.chunk_index,
            "chunk_start":   self.chunk_start_sec,
            "text":          self.transcript or "",
            "language_mix":  self.language_mix or "",
            "processed_at":  self.created_at.isoformat() if self.created_at else None,
            "processing_ms": self.processing_ms,
            "score": {
                "final_score":        self.final_score,
                "zone":               self.zone,
                "zone_emoji":         {"GREEN":"🟢","YELLOW":"🟡","RED":"🔴"}.get(self.zone,"🟢"),
                "severity":           self.severity,
                "auto_alert":         self.auto_alert,
                "dominant_emotion":   self.dominant_emotion,
                "emergency_category": self.emergency_category,
                "is_emergency":       self.is_emergency,
                "incident_id":        self.incident_id,
                "trend_upgraded":     self.trend_upgraded,
                "rising_trend":       self.rising_trend,
                "components": {
                    "emotion_component":   self.emotion_component,
                    "emergency_component": self.emergency_component,
                    "keyword_component":   self.keyword_component,
                    "sarcasm_deduction":   self.sarcasm_deduction,
                },
            },
            "keywords_found": self.get_keywords(),
            "keyword_boost":  self.keyword_boost,
            "bart_used":      self.bart_used,
        }

    def __repr__(self):
        return f"<Chunk {self.chunk_id} zone={self.zone} score={self.final_score:.3f}>"