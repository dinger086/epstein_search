"""Text extraction module."""

from .pdf_to_images import (
    convert_pdf_to_images,
    get_pdf_page_count,
    has_embedded_text,
    extract_text_from_pdf as extract_embedded_text,
    extract_images_from_pdf,
    get_image_count,
)
from .ocr import (
    extract_text_from_pdf,
    extract_text_from_image,
    describe_image,
    describe_image_for_search,
)

__all__ = [
    "convert_pdf_to_images",
    "get_pdf_page_count",
    "has_embedded_text",
    "extract_embedded_text",
    "extract_text_from_pdf",
    "extract_text_from_image",
    "extract_images_from_pdf",
    "get_image_count",
    "describe_image",
    "describe_image_for_search",
]
