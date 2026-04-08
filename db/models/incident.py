"""
db/models/incident.py
=======================
Groups RED chunks that belong to the same emergency event.
An incident is created when the first RED chunk is detected.
Subsequent RED chunks within the same session and within
INCIDENT_GAP_SEC seconds are added to the same incident.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text

from db.database import Base


class Incident(Base):
    __tablename__ = "incidents"

    id               = Column(String, primary_key=True)     # "INC_001", "INC_002", ...
    session_id       = Column(String, index=True, nullable=False)
    category         = Column(String, nullable=False)        # medical, fire, violence, ...
    severity         = Column(String, nullable=False)        # CRITICAL, HIGH, MEDIUM, LOW
    peak_score       = Column(Float, default=0.0)
    first_chunk_id   = Column(String, nullable=True)
    latest_chunk_id  = Column(String, nullable=True)
    chunk_count      = Column(Integer, default=1)
    red_chunk_count  = Column(Integer, default=1)           # how many RED chunks so far
    status           = Column(String, default="open")        # "open"|"confirmed"|"dismissed"|"auto_fired"
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at      = Column(DateTime, nullable=True)
    latest_text      = Column(Text, nullable=True)           # most recent flagged transcript

    def __repr__(self):
        return f"<Incident {self.id} status={self.status} category={self.category}>"