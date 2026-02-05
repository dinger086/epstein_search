"""Text chunking for RAG."""

import re
from ..config import CHUNK_SIZE, CHUNK_OVERLAP
from ..db import create_chunk, get_document

# Pattern to match page markers like "--- Page 1 ---"
PAGE_MARKER_PATTERN = re.compile(r'^---\s*Page\s+(\d+)\s*---\s*$', re.MULTILINE)


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence end
            for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break
            else:
                # Fall back to word boundary
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space + 1
        else:
            # We've reached the end - take the rest and stop
            end = len(text)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # If we've processed to the end, stop
        if end >= len(text):
            break

        # Move start, accounting for overlap
        # But ensure we always move forward to avoid infinite loops
        new_start = end - overlap
        if new_start <= start:
            new_start = end  # Skip overlap if chunk was too small
        start = new_start

    return chunks


def chunk_text_with_pages(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> list[tuple[str, int]]:
    """
    Split text into overlapping chunks, tracking page numbers.

    Args:
        text: Text to chunk (may contain "--- Page N ---" markers)
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks

    Returns:
        List of (chunk_text, page_number) tuples
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    if not text:
        return []

    # Find all page markers and their positions
    page_markers = []
    for match in PAGE_MARKER_PATTERN.finditer(text):
        page_markers.append((match.start(), int(match.group(1))))

    # Default to page 1 if no markers found
    if not page_markers:
        return [(chunk, 1) for chunk in chunk_text(text, chunk_size, overlap)]

    def get_page_at_position(pos: int) -> int:
        """Get the page number for a position in the text."""
        current_page = 1
        for marker_pos, page_num in page_markers:
            if marker_pos <= pos:
                current_page = page_num
            else:
                break
        return current_page

    # Chunk the text
    chunks = chunk_text(text, chunk_size, overlap)

    # Find each chunk's position and determine its page
    result = []
    search_start = 0
    for chunk in chunks:
        # Find where this chunk starts in the original text
        # (chunks are stripped, so we need to search for them)
        chunk_pos = text.find(chunk[:50], search_start)  # Use first 50 chars to find
        if chunk_pos == -1:
            chunk_pos = search_start

        page_num = get_page_at_position(chunk_pos)
        result.append((chunk, page_num))

        # Move search start forward to avoid finding same position
        search_start = chunk_pos + len(chunk) // 2

    return result


def chunk_document(doc_id: str, chunk_size: int = None, overlap: int = None) -> list[int]:
    """
    Chunk a document's text and store chunks in the database.

    Args:
        doc_id: Document ID
        chunk_size: Characters per chunk
        overlap: Overlap between chunks

    Returns:
        List of created chunk IDs
    """
    doc = get_document(doc_id)
    if not doc or not doc.get("extracted_text"):
        return []

    text = doc["extracted_text"]
    document_id = doc["id"]

    chunks_with_pages = chunk_text_with_pages(text, chunk_size, overlap)
    chunk_ids = []

    for idx, (chunk_text_content, page_num) in enumerate(chunks_with_pages):
        chunk_id = create_chunk(
            document_id=document_id,
            chunk_index=idx,
            text=chunk_text_content,
            page_number=page_num
        )
        chunk_ids.append(chunk_id)

    return chunk_ids
