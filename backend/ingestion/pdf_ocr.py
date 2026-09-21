"""
OCR fallback for scanned PDFs.

PyPDF trusts whatever text layer a PDF carries. A scanner app's built-in OCR
often writes garbage into that layer ("Custuuing -Aqotithms" for "Clustering
Algorithms"), and an image-only scan carries none at all; both used to be
indexed as-is, producing chunks no embedding can retrieve. Here each page's
text layer is scored, and pages that fail are rendered and run through the
shared EasyOCR reader instead.

Quality signal: the fraction of words that are whole-word entries in the
embedding model's own WordPiece vocabulary (already on disk, no new dependency).
Measured on real data: OCR garbage 0.00-0.30, prose 0.84-0.91, technical text
full of product names 0.84-0.90. Threshold 0.55.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.55     # known-word ratio below this => the text layer is garbage
MIN_WORDS = 12               # fewer words than this on a page => treat as absent
RENDER_DPI = 170             # OCR quality vs speed; 150-200 is the usual band
_WORD_RE = re.compile(r"[A-Za-z]{3,}")

_vocab: set[str] | None = None
_vocab_failed = False


def _load_vocab() -> set[str] | None:
    """Whole-word entries of the sentence-transformers tokenizer vocabulary, cached."""
    global _vocab, _vocab_failed
    if _vocab is not None or _vocab_failed:
        return _vocab
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        _vocab = {w for w in tok.get_vocab() if w.isalpha() and len(w) > 2}
        logger.info(f"Text-quality vocabulary loaded: {len(_vocab)} words")
    except Exception as e:
        logger.warning(f"Text-quality vocabulary unavailable ({e}); only empty pages will be OCR'd.")
        _vocab_failed = True
    return _vocab


def known_word_ratio(text: str) -> tuple[float | None, int]:
    """(ratio of words found in the vocabulary or None if unavailable, word count)."""
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    vocab = _load_vocab()
    if not words:
        return 0.0, 0
    if vocab is None:
        return None, len(words)
    return sum(w in vocab for w in words) / len(words), len(words)


def page_needs_ocr(text: str) -> tuple[bool, str]:
    """Decide for one page. Returns (needs_ocr, reason)."""
    ratio, n = known_word_ratio(text)
    if n < MIN_WORDS:
        return True, f"text layer absent or near-empty ({n} words)"
    if ratio is not None and ratio < QUALITY_THRESHOLD:
        return True, f"text layer is garbage (known-word ratio {ratio:.2f})"
    return False, f"text layer ok (known-word ratio {ratio if ratio is None else round(ratio, 2)})"


def render_page_images(pdf_bytes: bytes, page_indices: list[int], dpi: int = RENDER_DPI):
    """Yield (page_index, PIL.Image) for the requested pages using pypdfium2."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in page_indices:
            page = pdf[i]
            try:
                yield i, page.render(scale=dpi / 72.0).to_pil().convert("RGB")
            finally:
                page.close()
    finally:
        pdf.close()


def ocr_image(img) -> str:
    """OCR one PIL image with the shared reader; '' if OCR is unavailable."""
    import numpy as np
    from retrieval.ocr import read_text
    lines = read_text(np.asarray(img), detail=0, paragraph=True)
    return "\n".join(str(x) for x in lines if str(x).strip())


@dataclass
class PageResult:
    index: int
    text: str
    extraction: str            # "text" | "ocr" | "ocr-failed"
    reason: str
    ratio_before: float | None
    ratio_after: float | None = None


@dataclass
class ExtractionReport:
    pages: int
    ocr_pages: int
    ocr_failed_pages: int
    mean_ratio_before: float | None
    mean_ratio_after: float | None
    details: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"pages": self.pages, "ocr_pages": self.ocr_pages, "ocr_failed_pages": self.ocr_failed_pages,
                "text_quality_before": self.mean_ratio_before, "text_quality_after": self.mean_ratio_after,
                "pages_detail": self.details}


def extract_pages(pdf_bytes: bytes, layer_texts: list[str]) -> tuple[list[PageResult], ExtractionReport]:
    """
    Given the text-layer text of each page (from PyPDF), return the text to index
    per page, OCR-ing the pages whose layer is absent or garbage.
    """
    results: list[PageResult] = []
    to_ocr: list[int] = []
    for i, t in enumerate(layer_texts):
        needs, reason = page_needs_ocr(t)
        ratio, _ = known_word_ratio(t)
        results.append(PageResult(i, t, "text", reason, ratio))
        if needs:
            to_ocr.append(i)

    if to_ocr:
        logger.info(f"OCR needed on {len(to_ocr)}/{len(layer_texts)} pages")
        try:
            for i, img in render_page_images(pdf_bytes, to_ocr):
                text = ocr_image(img)
                r = results[i]
                if text.strip():
                    r.text, r.extraction = text, "ocr"
                    r.ratio_after, _ = known_word_ratio(text)
                else:
                    r.extraction = "ocr-failed"   # keep whatever the layer had
                    r.ratio_after = r.ratio_before
        except Exception as e:
            logger.error(f"Page rendering/OCR failed: {e}")
            for i in to_ocr:
                results[i].extraction = "ocr-failed"

    ocr_pages = [r for r in results if r.extraction == "ocr"]
    failed = [r for r in results if r.extraction == "ocr-failed"]
    before = [r.ratio_before for r in results if r.ratio_before is not None]
    after = [(r.ratio_after if r.ratio_after is not None else r.ratio_before) for r in results
             if (r.ratio_after if r.ratio_after is not None else r.ratio_before) is not None]
    report = ExtractionReport(
        pages=len(results), ocr_pages=len(ocr_pages), ocr_failed_pages=len(failed),
        mean_ratio_before=round(sum(before) / len(before), 3) if before else None,
        mean_ratio_after=round(sum(after) / len(after), 3) if after else None,
        details=[{"page": r.index + 1, "extraction": r.extraction, "reason": r.reason,
                  "quality_before": None if r.ratio_before is None else round(r.ratio_before, 2),
                  "quality_after": None if r.ratio_after is None else round(r.ratio_after, 2)} for r in results],
    )
    return results, report
