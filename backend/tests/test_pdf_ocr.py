"""
OCR fallback for scanned PDFs.

Background: the live index held a machine-learning notes PDF as unreadable garbage
("Custuuing -Aqotithms! Cuukeing Custos Mpdules") because a scanner app had written
its own bad OCR into the PDF text layer and ingestion trusted it. Pages whose layer
is absent or garbage are now rendered and OCR'd instead.

Run:  cd backend && python -m pytest tests/test_pdf_ocr.py -v
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import pdf_ocr
from ingestion.pdf_ocr import (QUALITY_THRESHOLD, extract_pages, known_word_ratio, page_needs_ocr,
                               render_page_images)

# Verbatim from the live index (scanner-app OCR of handwritten notes).
GARBAGE = ("Custuuing -Aqotithms! Cuukeing Custos Mpdules hau mmy atimibutes rito meoningl "
           "ousecntclwters cluteaig . Unlabec data Aplianoasi. fCAratib re don by tsial omd "
           "slecten untalboleol atsot hat Cnd ctey Custo SOutts ose dyramie 6rauping balecl")
GOOD = ("Clustering algorithms group unlabelled data into meaningful clusters based on "
        "similarity. K-means selects centroids and assigns each point to the nearest one, "
        "iterating until the assignments stop changing. It is simple and fast.")
TECHNICAL = ("FastAPI was chosen for its asynchronous capabilities, essential for streaming SSE "
             "updates without blocking the event loop. LangChain provides the glue between the "
             "LLM and the FAISS vector database; Ollama hosts Llama 3.2 locally on 4GB VRAM.")


# ── quality signal ──────────────────────────────────────────────────────────

def test_known_word_ratio_separates_garbage_from_text():
    g, _ = known_word_ratio(GARBAGE)
    p, _ = known_word_ratio(GOOD)
    t, _ = known_word_ratio(TECHNICAL)
    assert g is not None, "vocabulary must load offline from the cached embedding model"
    assert g < 0.35, g
    assert p > 0.8, p
    assert t > 0.65, f"product-name-dense technical prose must stay well above the {QUALITY_THRESHOLD} threshold (got {t:.2f})"
    assert g < QUALITY_THRESHOLD < p


def test_page_decisions():
    assert page_needs_ocr(GARBAGE)[0] is True
    assert page_needs_ocr("")[0] is True
    assert page_needs_ocr("Page 3")[0] is True, "a near-empty layer counts as absent"
    assert page_needs_ocr(GOOD)[0] is False
    assert page_needs_ocr(TECHNICAL)[0] is False


def test_without_vocabulary_only_empty_pages_are_ocrd(monkeypatch):
    monkeypatch.setattr(pdf_ocr, "_vocab", None)
    monkeypatch.setattr(pdf_ocr, "_vocab_failed", True)
    assert page_needs_ocr(GARBAGE)[0] is False, "cannot judge quality: keep the layer"
    assert page_needs_ocr("")[0] is True


# ── synthetic scanned PDF: no text layer at all ─────────────────────────────

def _image_only_pdf(lines: list[str], pages: int = 1) -> bytes:
    """A PDF made purely of rendered page images (what a scanner produces)."""
    from PIL import Image, ImageDraw, ImageFont
    imgs = []
    for _ in range(pages):
        img = Image.new("RGB", (1240, 1754), "white")           # A4 at 150 DPI
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except Exception:
            font = ImageFont.load_default()
        y = 120
        for line in lines:
            d.text((100, y), line, fill="black", font=font); y += 70
        imgs.append(img)
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def test_image_only_pdf_has_no_text_layer_and_renders():
    pdf = _image_only_pdf(["Clustering algorithms group unlabelled data."], pages=2)
    from pypdf import PdfReader
    layer = [p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf)).pages]
    assert all(page_needs_ocr(t)[0] for t in layer), "scanner output has no usable text layer"
    rendered = list(render_page_images(pdf, [0, 1], dpi=100))
    assert [i for i, _ in rendered] == [0, 1]
    assert rendered[0][1].size[0] > 500


def test_extract_pages_ocrs_bad_pages_and_keeps_good_ones(monkeypatch):
    pdf = _image_only_pdf(["x"], pages=3)
    ocr_calls = []
    monkeypatch.setattr(pdf_ocr, "ocr_image", lambda img: (ocr_calls.append(1), "Clustering algorithms group unlabelled data into meaningful clusters based on similarity measures and centroids")[1])
    results, report = extract_pages(pdf, [GOOD, "", GARBAGE])
    assert [r.extraction for r in results] == ["text", "ocr", "ocr"]
    assert len(ocr_calls) == 2, "only the bad pages are rendered and OCR'd"
    assert results[0].text == GOOD
    assert results[2].text.startswith("Clustering")
    assert report.pages == 3 and report.ocr_pages == 2 and report.ocr_failed_pages == 0
    assert report.mean_ratio_after > report.mean_ratio_before


def test_ocr_unavailable_keeps_layer_and_reports_failure(monkeypatch):
    pdf = _image_only_pdf(["x"], pages=1)
    monkeypatch.setattr(pdf_ocr, "ocr_image", lambda img: "")
    results, report = extract_pages(pdf, [GARBAGE])
    assert results[0].extraction == "ocr-failed" and results[0].text == GARBAGE
    assert report.ocr_failed_pages == 1 and report.ocr_pages == 0


@pytest.mark.slow
def test_real_ocr_reads_a_printed_scan():
    """End to end with the real EasyOCR reader (CPU; ~10-30 s)."""
    from retrieval.ocr import get_reader
    if get_reader() is None:
        pytest.skip("OCR reader unavailable in this environment")
    lines = ["Clustering algorithms group unlabelled data.",
             "K-means selects centroids and assigns points.",
             "The verifier flags numerical hallucinations."]
    pdf = _image_only_pdf(lines)
    layer = [""]
    results, report = extract_pages(pdf, layer)
    assert results[0].extraction == "ocr", report.details
    text = results[0].text.lower()
    assert "clustering" in text and "centroids" in text and "hallucinations" in text
    # The ratio is a garbage detector, not an absolute grade: correct OCR of short
    # technical text scores ~0.6 because words like "centroids" are not whole-word
    # vocabulary entries. It must clear the same bar the text layer is held to.
    assert report.mean_ratio_after > QUALITY_THRESHOLD, report.as_dict()


# ── delete_by_source ────────────────────────────────────────────────────────

def test_delete_by_source_removes_only_that_file(tmp_path):
    from langchain_core.documents import Document
    from retrieval.vector_db import VectorDatabase
    db = VectorDatabase(index_path=str(tmp_path / "idx"))
    db.add_documents([Document(page_content="old garbage one", metadata={"source": "notes.pdf"}),
                      Document(page_content="old garbage two", metadata={"source": "notes.pdf"}),
                      Document(page_content="keep me", metadata={"source": "other.pdf"})])
    assert db.delete_by_source("notes.pdf") == 2
    assert db.delete_by_source("notes.pdf") == 0
    remaining = {d.metadata.get("source") for d in db.retrieve("anything", top_k=10)}
    assert "notes.pdf" not in remaining and "other.pdf" in remaining
