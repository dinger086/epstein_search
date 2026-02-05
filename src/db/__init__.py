"""Database module."""

from .connection import get_connection, init_db
from .models import (
    create_document,
    get_document,
    get_documents_by_status,
    update_document_status,
    update_document_text,
    create_chunk,
    get_chunks_for_document,
    create_entity,
    get_entities_by_type,
    search_entities,
    create_image,
    get_image,
    get_images_for_document,
    update_image_description,
    search_images,
    get_images_without_descriptions,
    get_image_count,
)

__all__ = [
    "get_connection",
    "init_db",
    "create_document",
    "get_document",
    "get_documents_by_status",
    "update_document_status",
    "update_document_text",
    "create_chunk",
    "get_chunks_for_document",
    "create_entity",
    "get_entities_by_type",
    "search_entities",
    "create_image",
    "get_image",
    "get_images_for_document",
    "update_image_description",
    "search_images",
    "get_images_without_descriptions",
    "get_image_count",
]
