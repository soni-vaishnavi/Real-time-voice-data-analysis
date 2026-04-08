"""
db/database.py
===============
SQLAlchemy engine + session factory for VoiceGuard SQLite database.

SQLite in WAL mode:
  - Multiple readers + one writer at the same time (no lock conflicts)
  - Worker thread writes chunks; API routes read concurrently
  - Zero server process — single .db file

Usage:
    from db.database import SessionLocal, init_db

    # Initialize at startup (creates tables if not exist)
    init_db()

    # Use in worker / API routes
    with SessionLocal() as session:
        session.add(some_object)
        session.commit()
"""

import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base

from pipeline.core.config import DATABASE_URL

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args = {"check_same_thread": False},  # needed for SQLite + threads
    echo         = False,
)

# Enable WAL mode on every new connection — critical for concurrent access
@event.listens_for(engine, "connect")
def _enable_wal(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind           = engine,
    autocommit     = False,
    autoflush      = False,
    expire_on_commit = False,   # keep objects readable after commit
)

# ── Base class for all ORM models ─────────────────────────────────────────────
Base = declarative_base()


# ── FastAPI dependency injection ──────────────────────────────────────────────
def get_db():
    """
    FastAPI dependency. Yields a DB session, closes on exit.
    Usage in route:
        def my_route(db = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Startup initialization ────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create all tables if they don't exist.
    Call once at application startup before any DB operations.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS.
    """
    # Import all models so Base knows about them
    from db.models import chunk_model, session_model, incident, alert_action  # noqa
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized ✅  (WAL mode, tables created)")