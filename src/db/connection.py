"""Database connection handling."""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

from ..config import SQLITE_DB_PATH, ensure_dirs


_connection: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Get or create database connection."""
    global _connection
    if _connection is None:
        ensure_dirs()
        _connection = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA foreign_keys = ON")
        # Limit memory usage - default cache can grow large
        _connection.execute("PRAGMA cache_size = -64000")  # 64MB max cache
        _connection.execute("PRAGMA temp_store = FILE")    # Use file for temp storage, not RAM
        _connection.execute("PRAGMA mmap_size = 0")        # Disable memory-mapped I/O
    return _connection


@contextmanager
def transaction():
    """Context manager for database transactions."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Initialize database schema."""
    conn = get_connection()

    # Core document table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            doc_id TEXT UNIQUE,
            file_path TEXT,
            volume TEXT,
            page_count INTEGER,
            extracted_text TEXT,
            ocr_status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Full-text search index
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            doc_id, extracted_text,
            content='documents',
            content_rowid='id'
        )
    """)

    # Triggers to keep FTS in sync
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, doc_id, extracted_text)
            VALUES (new.id, new.doc_id, new.extracted_text);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, doc_id, extracted_text)
            VALUES ('delete', old.id, old.doc_id, old.extracted_text);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, doc_id, extracted_text)
            VALUES ('delete', old.id, old.doc_id, old.extracted_text);
            INSERT INTO documents_fts(rowid, doc_id, extracted_text)
            VALUES (new.id, new.doc_id, new.extracted_text);
        END
    """)

    # Chunks for RAG
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            chunk_index INTEGER,
            text TEXT,
            embedding_id TEXT,
            page_number INTEGER DEFAULT 1
        )
    """)

    # Add page_number column if it doesn't exist (migration for existing DBs)
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER DEFAULT 1")
    except:
        pass  # Column already exists

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document
        ON chunks(document_id)
    """)

    # Index for status queries (important with millions of rows)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_status
        ON documents(ocr_status)
    """)

    # Extracted entities
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            entity_type TEXT,
            entity_value TEXT,
            context TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entities_type_value
        ON entities(entity_type, entity_value)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entities_document
        ON entities(document_id)
    """)

    # Extracted images
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            page_number INTEGER,
            image_index INTEGER,
            width INTEGER,
            height INTEGER,
            file_path TEXT,
            description TEXT,
            embedding_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_images_document
        ON images(document_id)
    """)

    # FTS for image descriptions
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
            description,
            content='images',
            content_rowid='id'
        )
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
            INSERT INTO images_fts(rowid, description)
            VALUES (new.id, new.description);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
            INSERT INTO images_fts(images_fts, rowid, description)
            VALUES ('delete', old.id, old.description);
            INSERT INTO images_fts(rowid, description)
            VALUES (new.id, new.description);
        END
    """)

    conn.commit()


def close_connection():
    """Close the database connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
