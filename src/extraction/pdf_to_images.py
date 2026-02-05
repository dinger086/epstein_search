"""PDF to image conversion using PyMuPDF."""

from pathlib import Path
from typing import Generator
from PIL import Image
import io

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def _check_pymupdf():
    """Verify PyMuPDF is available."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("pymupdf is required. Install with: pip install pymupdf")


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Get the number of pages in a PDF."""
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        return len(doc)


def convert_pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200,
    fmt: str = "PNG"
) -> Generator[tuple[int, Image.Image], None, None]:
    """
    Convert PDF pages to images.

    Yields tuples of (page_number, PIL.Image).
    Page numbers are 1-indexed.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 200)
        fmt: Image format (default PNG)

    Yields:
        Tuples of (page_number, image)
    """
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Calculate zoom factor from DPI (72 is default PDF DPI)
    zoom = dpi / 72

    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page to pixmap
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            yield page_num + 1, img  # 1-indexed


def convert_pdf_to_images_batch(
    pdf_path: str | Path,
    dpi: int = 200,
    batch_size: int = 10
) -> Generator[list[tuple[int, Image.Image]], None, None]:
    """
    Convert PDF pages to images in batches.

    Yields lists of (page_number, PIL.Image) tuples.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion
        batch_size: Number of pages per batch

    Yields:
        Lists of (page_number, image) tuples
    """
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = dpi / 72
    batch = []

    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            batch.append((page_num + 1, img))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def extract_text_from_pdf(pdf_path: str | Path) -> tuple[str, int]:
    """
    Extract embedded text from PDF (if available).

    This extracts text that's already in the PDF (not scanned images).
    Use OCR for scanned documents.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, page_count)
    """
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    texts = []
    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                texts.append(f"--- Page {page_num + 1} ---\n{text}")

    return "\n\n".join(texts), page_count


def has_embedded_text(pdf_path: str | Path) -> bool:
    """
    Check if PDF has extractable embedded text.

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if PDF has meaningful embedded text
    """
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False

    with fitz.open(str(pdf_path)) as doc:
        # Check first few pages for text
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text().strip()
            # If we find substantial text, it has embedded text
            if len(text) > 50:
                return True

    return False


def extract_images_from_pdf(
    pdf_path: str | Path,
    min_width: int = 100,
    min_height: int = 100
) -> Generator[tuple[int, int, Image.Image, dict], None, None]:
    """
    Extract embedded images from a PDF.

    Yields tuples of (page_number, image_index, PIL.Image, metadata).
    Page numbers are 1-indexed.

    Args:
        pdf_path: Path to PDF file
        min_width: Minimum image width to extract
        min_height: Minimum image height to extract

    Yields:
        Tuples of (page_number, image_index, image, metadata)
    """
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]  # Image XREF number

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Skip small images (likely icons/bullets)
                    if width < min_width or height < min_height:
                        continue

                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(image_bytes))

                    metadata = {
                        "width": width,
                        "height": height,
                        "ext": image_ext,
                        "colorspace": base_image.get("colorspace", ""),
                        "xref": xref,
                    }

                    yield page_num + 1, img_idx, img, metadata

                except Exception:
                    # Some images can't be extracted (e.g., inline images)
                    continue


def get_image_count(pdf_path: str | Path) -> int:
    """
    Count extractable images in a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of images
    """
    _check_pymupdf()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return 0

    count = 0
    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            count += len(page.get_images(full=True))

    return count
