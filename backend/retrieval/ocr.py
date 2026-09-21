"""
One shared, lazily-built EasyOCR reader for the whole backend.

Used by image analysis (text in uploaded pictures) and by ingestion (scanned
PDFs). Building the reader is expensive and its first use downloads weights,
which can fail transiently; both callers want the same behaviour: try once,
remember a failure for a cooldown, never take the caller down with it.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

RETRY_COOLDOWN_S = 600.0
_lock = threading.Lock()
_reader = None          # None = not tried, False = permanently unavailable, else the reader
_failed_at = 0.0


def get_reader():
    """Return the shared reader, or None if OCR is unavailable right now."""
    global _reader, _failed_at
    with _lock:
        if _reader is not None:
            return _reader or None
        if _failed_at and time.monotonic() - _failed_at < RETRY_COOLDOWN_S:
            return None
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=True)
            logger.info("EasyOCR reader initialised.")
        except ImportError:
            logger.warning("easyocr not installed; OCR disabled.")
            _reader = False
        except Exception as e:
            logger.warning(f"OCR unavailable ({type(e).__name__}: {str(e)[:120]}); will retry after cooldown.")
            _failed_at = time.monotonic()
            return None
        return _reader or None


def read_text(image, detail: int = 0, paragraph: bool = True) -> list[str]:
    """OCR an image (bytes, path or numpy array). Empty list if OCR is unavailable."""
    reader = get_reader()
    if reader is None:
        return []
    try:
        return reader.readtext(image, detail=detail, paragraph=paragraph)
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return []


def _reset_for_tests():
    global _reader, _failed_at
    with _lock:
        _reader, _failed_at = None, 0.0
