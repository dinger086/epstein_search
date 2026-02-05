"""Embedding generation using Ollama."""

from ..config import EMBEDDING_MODEL

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def _check_ollama():
    """Verify Ollama is available."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama is required. Install with: pip install ollama")


def get_embedding(text: str, model: str = None) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model: Embedding model name

    Returns:
        Embedding vector
    """
    _check_ollama()
    model = model or EMBEDDING_MODEL

    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def get_embeddings(texts: list[str], model: str = None) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (batch).

    Args:
        texts: List of texts to embed
        model: Embedding model name

    Returns:
        List of embedding vectors
    """
    _check_ollama()
    model = model or EMBEDDING_MODEL

    if not texts:
        return []

    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]


def embedding_dimension(model: str = None) -> int:
    """
    Get the dimension of embeddings for a model.

    Args:
        model: Embedding model name

    Returns:
        Embedding dimension
    """
    _check_ollama()
    model = model or EMBEDDING_MODEL

    # Generate a test embedding to get dimension
    test_embedding = get_embedding("test", model=model)
    return len(test_embedding)
