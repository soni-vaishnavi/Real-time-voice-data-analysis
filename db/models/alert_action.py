"""
db/models/alert_action.py
===========================
Audit log — every alert fired and every operator action is recorded here.
Survives dashboard restarts (unlike the old session_state approach).
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text

from db.database import Base


class AlertAction(Base):
    __tablename__ = "alert_actions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    incident_id  = Column(String, index=True, nullable=False)
    chunk_id     = Column(String, index=True, nullable=False)
    session_id   = Column(String, nullable=True)
    action       = Column(String, nullable=False)      # "AUTO_FIRED"|"CONFIRMED"|"REJECTED"|"SMS_SENT"|"EMAIL_SENT"
    operator_id  = Column(String, default="system")    # "system" for auto actions
    reason       = Column(String, nullable=True)        # "AUTO"|"CONFIRMED"|"REJECTED"|timeout note
    category     = Column(String, nullable=True)
    severity     = Column(String, nullable=True)
    final_score  = Column(String, nullable=True)        # stored as string for display
    zone         = Column(String, nullable=True)
    text_preview = Column(Text, nullable=True)          # first 120 chars of transcript
    sms_sent     = Column(Boolean, default=False)
    email_sent   = Column(Boolean, default=False)
    sound_played = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<AlertAction id={self.id} incident={self.incident_id} action={self.action}>"