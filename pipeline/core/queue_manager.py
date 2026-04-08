"""
pipeline/core/queue_manager.py
================================
STAGE 2 — Bounded Audio Queue with Deduplication

Wraps Python's queue.Queue with:
  - maxsize=QUEUE_MAXSIZE (drops OLDEST chunk when full, not newest)
  - MD5-based deduplication (rejects identical chunks within 30s window)
  - Queue metrics (size, total in, total dropped)

Why drop oldest, not newest?
  Surveillance priority is recency — the most recent audio is most likely
  to contain an active emergency. An old queued chunk from 40 seconds ago
  is less actionable than the current one.
"""

import queue
import hashlib
import logging
from typing import Dict, Optional

from pipeline.core.config import QUEUE_MAXSIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AudioQueue:
    """
    Thread-safe bounded queue for audio chunks.

    Usage:
        q = AudioQueue()
        q.put(item)          # non-blocking, drops oldest if full
        item = q.get()       # blocks until item available
        q.task_done()        # signal processing complete
    """

    def __init__(self, maxsize: int = QUEUE_MAXSIZE):
        self._q       = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._seen_hashes: list  = []   # rolling list of recent hashes
        self._seen_set:    set   = set()
        self._total_in       = 0
        self._total_dropped  = 0
        self._dedup_window   = 60  # seconds worth of hashes to remember (approx 12 chunks)
        logger.info(f"AudioQueue initialized | maxsize={maxsize}")

    def put(self, item: Dict) -> bool:
        """
        Add item to queue. Non-blocking.

        If queue is full, removes the oldest item first, then adds the new one.
        Returns True if item was queued, False if it was a duplicate.
        """
        # Deduplication: hash first 2 KB of audio file
        chunk_hash = self._hash_item(item)
        if chunk_hash and chunk_hash in self._seen_set:
            logger.debug(f"Duplicate chunk rejected: {item.get('chunk_id')}")
            return False

        # Track hash for deduplication window
        if chunk_hash:
            self._seen_hashes.append(chunk_hash)
            self._seen_set.add(chunk_hash)
            # Keep rolling window to ~60 seconds of chunks
            if len(self._seen_hashes) > 24:
                old = self._seen_hashes.pop(0)
                self._seen_set.discard(old)

        # Drop oldest if full (not newest)
        if self._q.full():
            try:
                dropped = self._q.get_nowait()
                self._q.task_done()
                self._total_dropped += 1
                logger.warning(
                    f"Queue full — dropped oldest chunk: {dropped.get('chunk_id','?')} | "
                    f"total dropped={self._total_dropped}"
                )
            except queue.Empty:
                pass

        try:
            self._q.put_nowait(item)
            self._total_in += 1
            return True
        except queue.Full:
            self._total_dropped += 1
            return False

    def get(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next item. Raises queue.Empty if timeout expires.
        Caller must call task_done() after processing.
        """
        return self._q.get(timeout=timeout)

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()

    def is_full(self) -> bool:
        return self._q.full()

    def stats(self) -> Dict:
        return {
            "current_size":  self._q.qsize(),
            "maxsize":       self._maxsize,
            "total_in":      self._total_in,
            "total_dropped": self._total_dropped,
        }

    @staticmethod
    def _hash_item(item: Dict) -> Optional[str]:
        """Hash first 2 KB of the audio file for deduplication."""
        try:
            audio_path = item.get("audio_path", "")
            if audio_path:
                import os
                if os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        return hashlib.md5(f.read(2048)).hexdigest()
        except Exception:
            pass
        return None