"""Full-text search using SQLite FTS5."""

from ..db.connection import get_connection
from ..config import TOP_K_RESULTS


def keyword_search(
    query: str,
    limit: int = None,
    highlight: bool = True
) -> list[dict]:
    """
    Search documents using FTS5 full-text search.

    Supports:
    - Simple terms: "flight log"
    - Phrases: '"john doe"'
    - Boolean: "john AND doe", "john OR jane"
    - Prefix: "john*"
    - Exclude: "john NOT doe"

    Args:
        query: Search query
        limit: Maximum results
        highlight: Include highlighted snippets

    Returns:
        List of matching documents with scores
    """
    limit = limit or TOP_K_RESULTS
    conn = get_connection()

    if highlight:
        sql = """
            SELECT
                d.*,
                bm25(documents_fts) as score,
                snippet(documents_fts, 1, '<mark>', '</mark>', '...', 32) as snippet
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts)
            LIMIT ?
        """
    else:
        sql = """
            SELECT
                d.*,
                bm25(documents_fts) as score
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts)
            LIMIT ?
        """

    try:
        rows = conn.execute(sql, (query, limit)).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        # FTS5 query syntax errors
        if "fts5" in str(e).lower() or "syntax" in str(e).lower():
            # Fall back to simple phrase search
            escaped_query = f'"{query}"'
            rows = conn.execute(sql, (escaped_query, limit)).fetchall()
            return [dict(row) for row in rows]
        raise


def search_in_document(doc_id: str, query: str) -> list[dict]:
    """
    Search within a specific document.

    Args:
        doc_id: Document ID
        query: Search terms

    Returns:
        List of matching excerpts
    """
    conn = get_connection()

    sql = """
        SELECT
            snippet(documents_fts, 1, '<mark>', '</mark>', '...', 64) as snippet,
            bm25(documents_fts) as score
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ? AND d.doc_id = ?
        ORDER BY bm25(documents_fts)
    """

    # Combine doc_id filter with query
    fts_query = f'{query} AND doc_id:{doc_id}'

    try:
        rows = conn.execute(sql, (fts_query, doc_id)).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        # Simpler approach if complex query fails
        rows = conn.execute(
            """
            SELECT
                snippet(documents_fts, 1, '<mark>', '</mark>', '...', 64) as snippet,
                bm25(documents_fts) as score
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ? AND d.doc_id = ?
            """,
            (f'"{query}"', doc_id)
        ).fetchall()
        return [dict(row) for row in rows]


def suggest_completions(prefix: str, limit: int = 10) -> list[str]:
    """
    Get autocomplete suggestions based on prefix.

    Args:
        prefix: Search prefix
        limit: Maximum suggestions

    Returns:
        List of suggested terms
    """
    conn = get_connection()

    # Search for terms starting with prefix
    sql = """
        SELECT DISTINCT doc_id
        FROM documents_fts
        WHERE documents_fts MATCH ?
        LIMIT ?
    """

    try:
        rows = conn.execute(sql, (f'{prefix}*', limit)).fetchall()
        return [row["doc_id"] for row in rows]
    except Exception:
        return []
