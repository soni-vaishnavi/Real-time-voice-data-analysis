"""
Phase 6 — Report Generation
"""
from .incident_report import generate_incident_report
from .session_report  import generate_session_report

__all__ = ["generate_incident_report", "generate_session_report"]