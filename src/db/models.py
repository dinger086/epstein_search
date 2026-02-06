"""Database model operations."""

import hashlib
from pathlib import Path
from typing import Optional
from .connection import get_connection, transaction


# Document operations

def create_document(
    doc_id: str,
    file_path: str,
    volume: str = "",
    page_count: int = 0,
    file_type: str = "pdf",
) -> int:
    """Create a new document record. Returns document ID."""
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (doc_id, file_path, volume, page_count, ocr_status, file_type)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (doc_id, file_path, volume, page_count, file_type)
        )
        return cursor.lastrowid


def get_document(doc_id: str) -> Optional[dict]:
    """Get document by doc_id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?",
        (doc_id,)
    ).fetchone()
    return dict(row) if row else None


def get_document_by_id(id: int) -> Optional[dict]:
    """Get document by internal ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?",
        (id,)
    ).fetchone()
    return dict(row) if row else None


def get_documents_by_status(status: str, limit: int = 100) -> list[dict]:
    """Get documents with a specific OCR status."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM documents WHERE ocr_status = ? LIMIT ?",
        (status, limit)
    ).fetchall()
    return [dict(row) for row in rows]


def iter_documents_by_status(status: str, batch_size: int = 100):
    """
    Iterate documents with a specific status in batches.
    Yields individual document dicts without loading all into memory.
    Only fetches columns needed for processing (excludes extracted_text).
    """
    conn = get_connection()
    last_id = 0

    while True:
        rows = conn.execute(
            """
            SELECT id, doc_id, file_path, volume, page_count, ocr_status, file_type
            FROM documents
            WHERE ocr_status = ? AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            (status, last_id, batch_size)
        ).fetchall()

        if not rows:
            break

        for row in rows:
            last_id = row["id"]
            yield dict(row)


def count_documents_by_status(status: str) -> int:
    """Count documents with a specific OCR status."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as count FROM documents WHERE ocr_status = ?",
        (status,)
    ).fetchone()
    return row["count"]


def count_documents_by_file_type() -> list[dict]:
    """Count documents grouped by file_type."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT COALESCE(file_type, 'pdf') as file_type,
               COUNT(*) as total,
               SUM(CASE WHEN ocr_status = 'completed' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN ocr_status = 'pending' THEN 1 ELSE 0 END) as pending,
               SUM(CASE WHEN ocr_status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM documents
        GROUP BY COALESCE(file_type, 'pdf')
        ORDER BY total DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def update_document_status(doc_id: str, status: str):
    """Update document OCR status."""
    with transaction() as conn:
        conn.execute(
            "UPDATE documents SET ocr_status = ? WHERE doc_id = ?",
            (status, doc_id)
        )


def update_document_text(
    doc_id: str,
    text: str,
    page_count: int = None,
    embedded_text: str = None,
    ocr_text: str = None,
    duration_seconds: float = None,
):
    """Update document extracted text and optional raw text archives."""
    with transaction() as conn:
        clauses = ["extracted_text = ?", "ocr_status = 'completed'"]
        params = [text]

        if page_count is not None:
            clauses.append("page_count = ?")
            params.append(page_count)
        if embedded_text is not None:
            clauses.append("embedded_text = ?")
            params.append(embedded_text)
        if ocr_text is not None:
            clauses.append("ocr_text = ?")
            params.append(ocr_text)
        if duration_seconds is not None:
            clauses.append("duration_seconds = ?")
            params.append(duration_seconds)

        params.append(doc_id)
        conn.execute(
            f"UPDATE documents SET {', '.join(clauses)} WHERE doc_id = ?",
            params,
        )


def get_all_documents(limit: int = None) -> list[dict]:
    """Get all documents."""
    conn = get_connection()
    if limit:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY id LIMIT ?",
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def get_document_count() -> int:
    """Get total document count."""
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()
    return row["count"]


# Chunk operations

def create_chunk(
    document_id: int,
    chunk_index: int,
    text: str,
    embedding_id: str = None,
    page_number: int = 1
) -> int:
    """Create a new chunk. Returns chunk ID."""
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, text, embedding_id, page_number)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, chunk_index, text, embedding_id, page_number)
        )
        return cursor.lastrowid


def update_chunk_embedding(chunk_id: int, embedding_id: str):
    """Update chunk embedding ID."""
    with transaction() as conn:
        conn.execute(
            "UPDATE chunks SET embedding_id = ? WHERE id = ?",
            (embedding_id, chunk_id)
        )


def get_chunks_for_document(document_id: int) -> list[dict]:
    """Get all chunks for a document."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (document_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_chunk_by_embedding_id(embedding_id: str) -> Optional[dict]:
    """Get chunk by embedding ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM chunks WHERE embedding_id = ?",
        (embedding_id,)
    ).fetchone()
    return dict(row) if row else None


def get_chunks_without_embeddings(limit: int = 100) -> list[dict]:
    """Get chunks that don't have embeddings yet."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT c.*, d.doc_id
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding_id IS NULL
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    return [dict(row) for row in rows]


# Entity operations

def create_entity(
    document_id: int,
    entity_type: str,
    entity_value: str,
    context: str = ""
) -> int:
    """Create a new entity. Returns entity ID."""
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO entities (document_id, entity_type, entity_value, context)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, entity_type, entity_value, context)
        )
        return cursor.lastrowid


def get_entities_by_type(entity_type: str, limit: int = 100) -> list[dict]:
    """Get entities by type with counts."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT entity_value, COUNT(*) as count
        FROM entities
        WHERE entity_type = ?
        GROUP BY entity_value
        ORDER BY count DESC
        LIMIT ?
        """,
        (entity_type, limit)
    ).fetchall()
    return [dict(row) for row in rows]


def search_entities(entity_value: str) -> list[dict]:
    """Search for entities by value (partial match)."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT e.*, d.doc_id
        FROM entities e
        JOIN documents d ON e.document_id = d.id
        WHERE e.entity_value LIKE ?
        ORDER BY d.doc_id
        """,
        (f"%{entity_value}%",)
    ).fetchall()
    return [dict(row) for row in rows]


def get_entities_for_document(document_id: int) -> list[dict]:
    """Get all entities for a document."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM entities WHERE document_id = ? ORDER BY entity_type",
        (document_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_documents_with_entity(entity_value: str) -> list[dict]:
    """Get all documents containing a specific entity."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT DISTINCT d.*
        FROM documents d
        JOIN entities e ON e.document_id = d.id
        WHERE e.entity_value = ?
        ORDER BY d.doc_id
        """,
        (entity_value,)
    ).fetchall()
    return [dict(row) for row in rows]


# Image operations

def create_image(
    document_id: int,
    page_number: int,
    image_index: int,
    width: int,
    height: int,
    file_path: str,
    description: str = "",
    embedding_id: str = None
) -> int:
    """Create a new image record. Returns image ID."""
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO images (document_id, page_number, image_index, width, height, file_path, description, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, page_number, image_index, width, height, file_path, description, embedding_id)
        )
        return cursor.lastrowid


def get_image(image_id: int) -> Optional[dict]:
    """Get image by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM images WHERE id = ?",
        (image_id,)
    ).fetchone()
    return dict(row) if row else None


def get_images_for_document(document_id: int) -> list[dict]:
    """Get all images for a document."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM images WHERE document_id = ? ORDER BY page_number, image_index",
        (document_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def update_image_description(image_id: int, description: str):
    """Update image description."""
    with transaction() as conn:
        conn.execute(
            "UPDATE images SET description = ? WHERE id = ?",
            (description, image_id)
        )


def update_image_embedding(image_id: int, embedding_id: str):
    """Update image embedding ID."""
    with transaction() as conn:
        conn.execute(
            "UPDATE images SET embedding_id = ? WHERE id = ?",
            (embedding_id, image_id)
        )


def search_images(query: str, limit: int = 20) -> list[dict]:
    """Search images by description using FTS."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT i.*, d.doc_id,
               snippet(images_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
        FROM images_fts
        JOIN images i ON i.id = images_fts.rowid
        JOIN documents d ON i.document_id = d.id
        WHERE images_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit)
    ).fetchall()
    return [dict(row) for row in rows]


def get_images_without_descriptions(limit: int = 100) -> list[dict]:
    """Get images that don't have descriptions yet."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT i.*, d.doc_id
        FROM images i
        JOIN documents d ON i.document_id = d.id
        WHERE i.description IS NULL OR i.description = ''
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_image_count() -> int:
    """Get total image count."""
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) as count FROM images").fetchone()
    return row["count"]


def get_all_images(limit: int = None) -> list[dict]:
    """Get all images."""
    conn = get_connection()
    if limit:
        rows = conn.execute(
            """
            SELECT i.*, d.doc_id
            FROM images i
            JOIN documents d ON i.document_id = d.id
            ORDER BY d.doc_id, i.page_number, i.image_index
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT i.*, d.doc_id
            FROM images i
            JOIN documents d ON i.document_id = d.id
            ORDER BY d.doc_id, i.page_number, i.image_index
            """
        ).fetchall()
    return [dict(row) for row in rows]


# File hash / duplicate detection operations

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file (streaming, memory-safe)."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def update_document_hash(doc_id: str, file_hash: str):
    """Store hash for a document."""
    with transaction() as conn:
        conn.execute(
            "UPDATE documents SET file_hash = ? WHERE doc_id = ?",
            (file_hash, doc_id)
        )


def get_duplicate_groups() -> list[dict]:
    """Get groups of documents sharing the same file hash.

    Returns list of dicts with keys: file_hash, count, documents.
    Each document has: doc_id, file_path, volume.
    """
    conn = get_connection()
    # Find hashes that appear more than once
    hash_rows = conn.execute(
        """
        SELECT file_hash, COUNT(*) as count
        FROM documents
        WHERE file_hash IS NOT NULL
        GROUP BY file_hash
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        """
    ).fetchall()

    groups = []
    for row in hash_rows:
        docs = conn.execute(
            """
            SELECT doc_id, file_path, volume, file_type
            FROM documents
            WHERE file_hash = ?
            ORDER BY doc_id
            """,
            (row["file_hash"],)
        ).fetchall()
        groups.append({
            "file_hash": row["file_hash"],
            "count": row["count"],
            "documents": [dict(d) for d in docs],
        })

    return groups


def get_documents_by_hash(file_hash: str) -> list[dict]:
    """Get all documents sharing a specific hash."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM documents WHERE file_hash = ? ORDER BY doc_id",
        (file_hash,)
    ).fetchall()
    return [dict(row) for row in rows]
