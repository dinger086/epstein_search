"""Vector store using ChromaDB."""

from pathlib import Path
from typing import Optional

from ..config import CHROMA_DB_PATH, ensure_dirs

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(self, collection_name: str = "documents", path: Path = None):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection
            path: Path to persist data (default: config CHROMA_DB_PATH)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        ensure_dirs()
        path = path or CHROMA_DB_PATH

        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] = None,
        metadatas: list[dict] = None
    ):
        """
        Add embeddings to the store.

        Args:
            ids: Unique IDs for each embedding
            embeddings: Embedding vectors
            documents: Optional document texts
            metadatas: Optional metadata dicts
        """
        kwargs = {
            "ids": ids,
            "embeddings": embeddings
        }
        if documents:
            kwargs["documents"] = documents
        if metadatas:
            kwargs["metadatas"] = metadatas

        self.collection.add(**kwargs)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict = None,
        include: list[str] = None
    ) -> dict:
        """
        Query similar embeddings.

        Args:
            query_embedding: Query vector
            n_results: Number of results
            where: Optional filter conditions
            include: Fields to include (default: ["documents", "metadatas", "distances"])

        Returns:
            Dict with ids, documents, metadatas, distances
        """
        include = include or ["documents", "metadatas", "distances"]

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": include
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Flatten the results (they come as nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            "metadatas": results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            "distances": results.get("distances", [[]])[0] if results.get("distances") else []
        }

    def get(self, ids: list[str]) -> dict:
        """
        Get embeddings by ID.

        Args:
            ids: List of IDs to retrieve

        Returns:
            Dict with ids, embeddings, documents, metadatas
        """
        return self.collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])

    def delete(self, ids: list[str]):
        """
        Delete embeddings by ID.

        Args:
            ids: List of IDs to delete
        """
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get total number of embeddings in the collection."""
        return self.collection.count()

    def update(
        self,
        ids: list[str],
        embeddings: list[list[float]] = None,
        documents: list[str] = None,
        metadatas: list[dict] = None
    ):
        """
        Update existing embeddings.

        Args:
            ids: IDs to update
            embeddings: New embedding vectors
            documents: New document texts
            metadatas: New metadata dicts
        """
        kwargs = {"ids": ids}
        if embeddings:
            kwargs["embeddings"] = embeddings
        if documents:
            kwargs["documents"] = documents
        if metadatas:
            kwargs["metadatas"] = metadatas

        self.collection.update(**kwargs)
