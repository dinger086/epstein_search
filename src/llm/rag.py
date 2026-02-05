"""RAG (Retrieval-Augmented Generation) pipeline."""

from ..config import LLM_MODEL, TOP_K_RESULTS
from ..search.hybrid import hybrid_search
from ..search.semantic import semantic_search
from ..db.models import get_document_by_id
from .ollama_client import chat


RAG_SYSTEM_PROMPT = """You are a research assistant analyzing documents from a large document collection.
Your task is to answer questions based on the provided context from these documents.

Guidelines:
- Only use information from the provided context
- If the context doesn't contain enough information to answer, say so
- Cite specific documents using their doc_id when making claims
- Be precise and factual
- If you see partial information, acknowledge what is known vs unknown"""


def format_context(results: list[dict], max_chars: int = 8000) -> str:
    """
    Format search results as context for the LLM.

    Args:
        results: Search results
        max_chars: Maximum context length

    Returns:
        Formatted context string
    """
    context_parts = []
    total_chars = 0

    for result in results:
        doc_id = result.get("doc_id", "unknown")
        chunk_text = result.get("chunk_text", "")

        if not chunk_text:
            # Try to get from document
            doc_db_id = result.get("document_id")
            if doc_db_id:
                doc = get_document_by_id(int(doc_db_id))
                if doc and doc.get("extracted_text"):
                    chunk_text = doc["extracted_text"][:500]

        if not chunk_text:
            continue

        entry = f"[Document: {doc_id}]\n{chunk_text}\n"

        if total_chars + len(entry) > max_chars:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n---\n".join(context_parts)


def ask(
    question: str,
    model: str = None,
    num_results: int = None,
    search_method: str = "hybrid"
) -> str:
    """
    Ask a question using RAG.

    Args:
        question: User question
        model: LLM model to use
        num_results: Number of documents to retrieve
        search_method: Search method (hybrid, semantic, keyword)

    Returns:
        Generated answer
    """
    model = model or LLM_MODEL
    num_results = num_results or TOP_K_RESULTS

    # Retrieve relevant documents
    if search_method == "semantic":
        results = semantic_search(question, limit=num_results)
    elif search_method == "keyword":
        from ..search.keyword import keyword_search
        results = keyword_search(question, limit=num_results)
    else:
        results = hybrid_search(question, limit=num_results)

    if not results:
        return "No relevant documents found for your question."

    # Format context
    context = format_context(results)

    # Generate answer
    answer = chat(
        prompt=f"Question: {question}",
        model=model,
        system=RAG_SYSTEM_PROMPT,
        context=context
    )

    return answer


def ask_with_sources(
    question: str,
    model: str = None,
    num_results: int = None,
    search_method: str = "hybrid"
) -> dict:
    """
    Ask a question and return answer with source documents.

    Args:
        question: User question
        model: LLM model to use
        num_results: Number of documents to retrieve
        search_method: Search method

    Returns:
        Dict with answer and sources
    """
    model = model or LLM_MODEL
    num_results = num_results or TOP_K_RESULTS

    # Retrieve relevant documents
    if search_method == "semantic":
        results = semantic_search(question, limit=num_results)
    elif search_method == "keyword":
        from ..search.keyword import keyword_search
        results = keyword_search(question, limit=num_results)
    else:
        results = hybrid_search(question, limit=num_results)

    if not results:
        return {
            "answer": "No relevant documents found for your question.",
            "sources": []
        }

    # Format context
    context = format_context(results)

    # Generate answer
    answer = chat(
        prompt=f"Question: {question}",
        model=model,
        system=RAG_SYSTEM_PROMPT,
        context=context
    )

    # Extract source info
    sources = []
    seen_docs = set()
    for result in results:
        doc_id = result.get("doc_id")
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "doc_id": doc_id,
                "file_path": result.get("file_path", ""),
                "relevance": result.get("similarity") or result.get("hybrid_score") or result.get("rrf_score", 0)
            })

    return {
        "answer": answer,
        "sources": sources,
        "context_used": context[:500] + "..." if len(context) > 500 else context
    }


def summarize_document(doc_id: str, model: str = None) -> str:
    """
    Generate a summary of a document.

    Args:
        doc_id: Document ID
        model: LLM model to use

    Returns:
        Document summary
    """
    from ..db.models import get_document

    doc = get_document(doc_id)
    if not doc or not doc.get("extracted_text"):
        return f"Document {doc_id} not found or has no text."

    text = doc["extracted_text"]

    # Truncate if too long
    max_text = 6000
    if len(text) > max_text:
        text = text[:max_text] + "\n\n[Text truncated...]"

    summary = chat(
        prompt="Please provide a concise summary of this document, highlighting key people, dates, and events mentioned.",
        model=model or LLM_MODEL,
        context=text
    )

    return summary
