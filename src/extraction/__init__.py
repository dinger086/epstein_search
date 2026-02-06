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
from .hybrid import extract_hybrid_text, text_quality_score
from .file_types import classify_file, ALL_EXTENSIONS, EXTENSION_MAP
from .spreadsheet import extract_text_from_spreadsheet
from .audio import transcribe_audio
from .video import extract_text_from_video
from .litigation import extract_text_from_litigation

# Registry: file_type -> extractor function
# Each extractor returns (text, metadata_dict)
EXTRACTORS = {
    "spreadsheet": extract_text_from_spreadsheet,
    "audio": transcribe_audio,
    "video": extract_text_from_video,
    "litigation": extract_text_from_litigation,
}

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
    "extract_hybrid_text",
    "text_quality_score",
    "classify_file",
    "ALL_EXTENSIONS",
    "EXTENSION_MAP",
    "extract_text_from_spreadsheet",
    "transcribe_audio",
    "extract_text_from_video",
    "extract_text_from_litigation",
    "EXTRACTORS",
]
