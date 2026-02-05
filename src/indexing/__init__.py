"""Indexing module for text chunking and embeddings."""

from .chunker import chunk_text, chunk_document
from .embeddings import get_embedding, get_embeddings
from .vector_store import VectorStore

__all__ = [
    "chunk_text",
    "chunk_document",
    "get_embedding",
    "get_embeddings",
    "VectorStore",
]
