"""
OCR must never take image analysis down with it.

Background: EasyOCR downloads its weights from GitHub release assets on first use;
one of them returned a 504 mid-request and the uncaught error aborted the whole
analysis before the local vision model was even called. OCR is an enhancement, so
its failure is logged, remembered for a cooldown, and analysis continues.

Run:  cd backend && python -m pytest tests/test_image_analyzer_ocr.py -v
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.image_analyzer import ImageAnalyzer


def _fake_easyocr(monkeypatch, behaviour):
    mod = types.ModuleType("easyocr")
    class Reader:
        def __init__(self, *a, **k):
            behaviour(self)
    mod.Reader = Reader
    monkeypatch.setitem(sys.modules, "easyocr", mod)


def test_download_failure_is_non_fatal_and_remembered(monkeypatch):
    calls = []
    def boom(self):
        calls.append(1); raise Exception("HTTP Error 504: Gateway Time-out")
    _fake_easyocr(monkeypatch, boom)
    a = ImageAnalyzer()
    assert a._get_ocr_reader() is None
    assert a._get_ocr_reader() is None
    assert len(calls) == 1, "no re-attempt inside the cooldown"
    a._ocr_failed_at -= ImageAnalyzer.OCR_RETRY_COOLDOWN_S + 1
    assert a._get_ocr_reader() is None and len(calls) == 2, "retried once the cooldown passed"


def test_missing_easyocr_is_skipped_permanently(monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", None)  # import raises ImportError
    a = ImageAnalyzer()
    assert a._get_ocr_reader() is None
    assert a.ocr_reader is False


def test_working_reader_is_cached(monkeypatch):
    made = []
    _fake_easyocr(monkeypatch, lambda self: made.append(1))
    a = ImageAnalyzer()
    r1 = a._get_ocr_reader(); r2 = a._get_ocr_reader()
    assert r1 is r2 and r1 is not None and len(made) == 1
