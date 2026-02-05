"""Hybrid text extraction — merges embedded text with selective OCR."""

from pathlib import Path
from typing import Callable

import io
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from .ocr import extract_text_from_image


def text_quality_score(text: str) -> float:
    """
    Score text quality.

    Returns alnum_count * (alnum_count / total_chars).
    A completely empty string scores 0.
    High-quality text (mostly alphanumeric) scores high.
    Garbage text (mostly special chars/whitespace) scores low.
    """
    if not text:
        return 0.0
    total = len(text)
    if total == 0:
        return 0.0
    alnum = sum(1 for c in text if c.isalnum())
    return alnum * (alnum / total)


def extract_hybrid_text(
    pdf_path: str | Path,
    quality_threshold: float = 50.0,
    dpi: int = 200,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[str, str, str, int]:
    """
    Extract text using a hybrid approach: embedded text first, OCR only where needed.

    Per-page logic:
      1. Extract embedded text for all pages (fast, PyMuPDF).
      2. Score each page's quality.
      3. OCR only pages below quality_threshold (renders + calls OCR only for those).
      4. Merge: per page, pick source with higher quality score (prefer embedded on tie).

    Args:
        pdf_path: Path to the PDF file.
        quality_threshold: Minimum quality score for embedded text to be accepted.
        dpi: Resolution for rendering pages to images (for OCR).
        progress_callback: Optional callback(current_page, total_pages, source).

    Returns:
        (merged_text, embedded_text, ocr_text, page_count)
    """
    if fitz is None:
        raise ImportError("pymupdf is required. Install with: pip install pymupdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- Step 1: extract embedded text per page ---
    embedded_pages: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            embedded_pages.append(page.get_text())

    # --- Step 2: score each page ---
    embedded_scores = [text_quality_score(p) for p in embedded_pages]

    # --- Step 3 & 4: OCR low-quality pages, merge ---
    merged_pages: list[str] = []
    ocr_pages: list[str] = [""] * page_count  # sparse, only filled for OCR'd pages
    zoom = dpi / 72

    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(page_count):
            page_label = page_num + 1  # 1-indexed

            if embedded_scores[page_num] >= quality_threshold:
                # Embedded text is good enough
                merged_pages.append(embedded_pages[page_num])
                source = "embedded"
            else:
                # Need OCR for this page
                page = doc[page_num]
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                ocr_text = extract_text_from_image(img)
                ocr_pages[page_num] = ocr_text

                ocr_score = text_quality_score(ocr_text)

                # Pick whichever is better (prefer embedded on tie)
                if ocr_score > embedded_scores[page_num]:
                    merged_pages.append(ocr_text)
                    source = "ocr"
                else:
                    merged_pages.append(embedded_pages[page_num])
                    source = "embedded"

            if progress_callback:
                progress_callback(page_label, page_count, source)

    # --- Build full-document strings with page markers ---
    def _join_pages(pages: list[str]) -> str:
        parts = []
        for i, text in enumerate(pages):
            if text.strip():
                parts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(parts)

    merged_text = _join_pages(merged_pages)
    full_embedded = _join_pages(embedded_pages)
    full_ocr = _join_pages(ocr_pages)

    return merged_text, full_embedded, full_ocr, page_count
