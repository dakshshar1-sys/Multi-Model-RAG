"""
The shared OCR reader must never take a caller down with it.

Background: EasyOCR downloads its weights from GitHub release assets on first use;
one of them returned a 504 mid-request and the uncaught error aborted the whole
image analysis. OCR is an enhancement for image analysis and one of two extraction
paths for ingestion, so a failure is logged, remembered for a cooldown, and callers
continue.

Run:  cd backend && python -m pytest tests/test_image_analyzer_ocr.py -v
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import ocr


@pytest.fixture(autouse=True)
def fresh_reader_state():
    ocr._reset_for_tests()
    yield
    ocr._reset_for_tests()


def _fake_easyocr(monkeypatch, behaviour):
    mod = types.ModuleType("easyocr")
    class Reader:
        def __init__(self, *a, **k):
            behaviour(self)
        def readtext(self, image, detail=0, paragraph=True):
            return ["hello", "world"]
    mod.Reader = Reader
    monkeypatch.setitem(sys.modules, "easyocr", mod)


def test_download_failure_is_non_fatal_and_remembered(monkeypatch):
    calls = []
    def boom(self):
        calls.append(1); raise Exception("HTTP Error 504: Gateway Time-out")
    _fake_easyocr(monkeypatch, boom)
    assert ocr.get_reader() is None
    assert ocr.get_reader() is None
    assert len(calls) == 1, "no re-attempt inside the cooldown"
    ocr._failed_at -= ocr.RETRY_COOLDOWN_S + 1
    assert ocr.get_reader() is None and len(calls) == 2, "retried once the cooldown passed"
    assert ocr.read_text(b"...") == [], "read_text degrades to nothing, not an exception"


def test_missing_easyocr_is_skipped_permanently(monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", None)  # import raises ImportError
    assert ocr.get_reader() is None
    assert ocr._reader is False


def test_working_reader_is_shared_and_cached(monkeypatch):
    made = []
    _fake_easyocr(monkeypatch, lambda self: made.append(1))
    r1 = ocr.get_reader(); r2 = ocr.get_reader()
    assert r1 is r2 and r1 is not None and len(made) == 1
    assert ocr.read_text(b"...") == ["hello", "world"]
    from retrieval.image_analyzer import ImageAnalyzer
    assert ImageAnalyzer()._get_ocr_reader() is r1, "image analysis uses the same reader"
