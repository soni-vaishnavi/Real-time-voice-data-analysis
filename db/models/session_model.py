"""
db/models/session_model.py
============================
One row per recording session.
A session = one mic capture run or one file pipeline run.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text

from db.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id           = Column(String, primary_key=True)          # e.g. "session_20260408_193500"
    source_type  = Column(String, nullable=False)             # "file" | "mic"
    source_path  = Column(String, nullable=True)              # file path if source_type="file"
    started_at   = Column(DateTime, default=datetime.utcnow)
    ended_at     = Column(DateTime, nullable=True)
    total_chunks = Column(Integer, default=0)
    red_count    = Column(Integer, default=0)
    yellow_count = Column(Integer, default=0)
    green_count  = Column(Integer, default=0)
    avg_score    = Column(Float, default=0.0)
    status       = Column(String, default="active")           # "active" | "completed" | "error"

    def __repr__(self):
        return f"<Session id={self.id} status={self.status} chunks={self.total_chunks}>"