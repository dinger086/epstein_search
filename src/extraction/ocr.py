"""OCR using Ollama's glm-ocr model."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image

from ..config import OCR_MODEL
from .pdf_to_images import convert_pdf_to_images, get_pdf_page_count

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def _check_ollama():
    """Verify Ollama is available."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama is required. Install with: pip install ollama")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


def extract_text_from_image(
    image: Image.Image | str | Path,
    model: str = None
) -> str:
    """
    Extract text from an image using glm-ocr.

    Args:
        image: PIL Image or path to image file
        model: Ollama model to use (default: glm-ocr)

    Returns:
        Extracted text
    """
    _check_ollama()
    model = model or OCR_MODEL

    # Convert to base64
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
    else:
        img_base64 = image_to_base64(image)

    # Call glm-ocr
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": "Text Recognition:",
            "images": [img_base64]
        }]
    )

    return response["message"]["content"]


def extract_text_from_pdf(
    pdf_path: str | Path,
    model: str = None,
    progress_callback: Callable[[int, int], None] = None
) -> tuple[str, int]:
    """
    Extract text from a PDF using OCR.

    Args:
        pdf_path: Path to PDF file
        model: Ollama model to use
        progress_callback: Optional callback(current_page, total_pages)

    Returns:
        Tuple of (extracted_text, page_count)
    """
    _check_ollama()
    model = model or OCR_MODEL

    pdf_path = Path(pdf_path)
    page_count = get_pdf_page_count(pdf_path)
    texts = []

    for page_num, image in convert_pdf_to_images(pdf_path):
        if progress_callback:
            progress_callback(page_num, page_count)

        # Extract text from this page
        text = extract_text_from_image(image, model=model)
        texts.append(f"--- Page {page_num} ---\n{text}")

    return "\n\n".join(texts), page_count


def extract_text_batch(
    images: list[Image.Image],
    model: str = None
) -> list[str]:
    """
    Extract text from multiple images.

    Args:
        images: List of PIL Images
        model: Ollama model to use

    Returns:
        List of extracted texts
    """
    _check_ollama()
    return [extract_text_from_image(img, model=model) for img in images]


def describe_image(
    image: Image.Image | str | Path,
    model: str = None,
    prompt: str = None
) -> str:
    """
    Generate a description of an image using a vision LLM.

    Args:
        image: PIL Image or path to image file
        model: Vision model to use (default: uses OCR_MODEL which should support vision)
        prompt: Custom prompt for description

    Returns:
        Image description
    """
    _check_ollama()
    model = model or OCR_MODEL

    # Convert to base64
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
    else:
        img_base64 = image_to_base64(image)

    default_prompt = """Describe this image in detail. Include:
- What type of document or image this is (photo, form, letter, chart, etc.)
- Any people visible (describe them without identifying)
- Key objects, locations, or scenes
- Any visible text (summarize, don't transcribe everything)
- Dates or time periods if apparent
- Any notable or unusual elements

Be concise but thorough."""

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt or default_prompt,
            "images": [img_base64]
        }]
    )

    return response["message"]["content"]


def describe_image_for_search(
    image: Image.Image | str | Path,
    model: str = None
) -> str:
    """
    Generate a search-optimized description of an image.

    Focuses on keywords and entities that would be useful for search.

    Args:
        image: PIL Image or path to image file
        model: Vision model to use

    Returns:
        Search-optimized description
    """
    prompt = """Analyze this image and provide a description optimized for search. Include:

1. Document type (photo, letter, form, receipt, chart, map, etc.)
2. People: roles/descriptions (e.g., "man in suit", "group of people at table")
3. Location clues (buildings, landmarks, signs, geography)
4. Time period clues (clothing, technology, document dates)
5. Key objects (vehicles, furniture, equipment)
6. Any readable text (names, dates, titles, headers)
7. Activities or events depicted

Format as a dense paragraph with key terms. Focus on searchable facts, not artistic interpretation."""

    return describe_image(image, model=model, prompt=prompt)
