"""Semantic search using vector embeddings."""

from ..config import TOP_K_RESULTS
from ..indexing.embeddings import get_embedding
from ..indexing.vector_store import VectorStore
from ..db.models import get_document_by_id, get_chunk_by_embedding_id


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def semantic_search(
    query: str,
    limit: int = None,
    include_chunks: bool = True
) -> list[dict]:
    """
    Search documents using semantic similarity.

    Args:
        query: Natural language query
        limit: Maximum results
        include_chunks: Include chunk text in results

    Returns:
        List of results with similarity scores
    """
    limit = limit or TOP_K_RESULTS

    # Get query embedding
    query_embedding = get_embedding(query)

    # Search vector store
    store = get_vector_store()
    results = store.query(
        query_embedding=query_embedding,
        n_results=limit,
        include=["documents", "metadatas", "distances"]
    )

    # Enrich results with document info
    enriched_results = []
    seen_docs = set()

    for i, embedding_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        distance = results["distances"][i] if results["distances"] else 0
        chunk_text = results["documents"][i] if results["documents"] else ""

        # Get chunk and document info
        document_id = metadata.get("document_id")
        if document_id:
            doc = get_document_by_id(int(document_id))
            if doc:
                doc_id = doc["doc_id"]

                # Track unique documents
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)

                result = {
                    "doc_id": doc_id,
                    "document_id": document_id,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "chunk_index": metadata.get("chunk_index", 0),
                    "page_number": metadata.get("page_number", 1),
                    "file_path": doc.get("file_path", ""),
                }

                if include_chunks:
                    result["chunk_text"] = chunk_text

                enriched_results.append(result)

    return enriched_results


def find_similar_documents(doc_id: str, limit: int = 10) -> list[dict]:
    """
    Find documents similar to a given document.

    Args:
        doc_id: Source document ID
        limit: Maximum results

    Returns:
        List of similar documents
    """
    from ..db.models import get_document, get_chunks_for_document

    doc = get_document(doc_id)
    if not doc:
        return []

    # Get chunks for this document
    chunks = get_chunks_for_document(doc["id"])
    if not chunks:
        # If no chunks, use full text
        text = doc.get("extracted_text", "")
        if not text:
            return []
        return semantic_search(text[:1000], limit=limit + 1)[1:]  # Exclude self

    # Use first chunk as representative
    first_chunk = chunks[0]["text"]
    results = semantic_search(first_chunk, limit=limit + 1)

    # Filter out the source document
    return [r for r in results if r["doc_id"] != doc_id][:limit]
