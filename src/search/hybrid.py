"""Hybrid search combining keyword and semantic approaches."""

from ..config import TOP_K_RESULTS, HYBRID_KEYWORD_WEIGHT, HYBRID_SEMANTIC_WEIGHT
from .keyword import keyword_search
from .semantic import semantic_search


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60
) -> list[dict]:
    """
    Combine multiple ranked result lists using RRF.

    Args:
        results_lists: List of result lists (each with doc_id)
        k: RRF constant (default 60)

    Returns:
        Combined and re-ranked results
    """
    scores = {}
    docs = {}

    for results in results_lists:
        for rank, result in enumerate(results):
            doc_id = result["doc_id"]
            rrf_score = 1.0 / (k + rank + 1)

            if doc_id in scores:
                scores[doc_id] += rrf_score
            else:
                scores[doc_id] = rrf_score
                docs[doc_id] = result

    # Sort by combined RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for doc_id in sorted_doc_ids:
        doc = docs[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)

    return results


def weighted_combination(
    keyword_results: list[dict],
    semantic_results: list[dict],
    keyword_weight: float = None,
    semantic_weight: float = None
) -> list[dict]:
    """
    Combine results using weighted scores.

    Args:
        keyword_results: Results from keyword search
        semantic_results: Results from semantic search
        keyword_weight: Weight for keyword scores
        semantic_weight: Weight for semantic scores

    Returns:
        Combined and re-ranked results
    """
    keyword_weight = keyword_weight or HYBRID_KEYWORD_WEIGHT
    semantic_weight = semantic_weight or HYBRID_SEMANTIC_WEIGHT

    scores = {}
    docs = {}

    # Normalize keyword scores (BM25 scores are negative, lower is better)
    if keyword_results:
        min_score = min(r.get("score", 0) for r in keyword_results)
        max_score = max(r.get("score", 0) for r in keyword_results)
        score_range = max_score - min_score if max_score != min_score else 1

        for result in keyword_results:
            doc_id = result["doc_id"]
            # Normalize to 0-1 (inverted because BM25 is lower=better)
            raw_score = result.get("score", 0)
            normalized = 1 - ((raw_score - min_score) / score_range)
            scores[doc_id] = keyword_weight * normalized
            docs[doc_id] = result

    # Add semantic scores (similarity is 0-1, higher is better)
    for result in semantic_results:
        doc_id = result["doc_id"]
        similarity = result.get("similarity", 0)

        if doc_id in scores:
            scores[doc_id] += semantic_weight * similarity
        else:
            scores[doc_id] = semantic_weight * similarity
            docs[doc_id] = result

    # Sort by combined score
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for doc_id in sorted_doc_ids:
        doc = docs[doc_id].copy()
        doc["hybrid_score"] = scores[doc_id]
        results.append(doc)

    return results


def hybrid_search(
    query: str,
    limit: int = None,
    method: str = "rrf",
    keyword_weight: float = None,
    semantic_weight: float = None
) -> list[dict]:
    """
    Search using both keyword and semantic methods.

    Args:
        query: Search query
        limit: Maximum results
        method: Combination method ("rrf" or "weighted")
        keyword_weight: Weight for keyword results (weighted method)
        semantic_weight: Weight for semantic results (weighted method)

    Returns:
        Combined search results
    """
    limit = limit or TOP_K_RESULTS

    # Run both searches with extra results for better combination
    fetch_limit = limit * 2

    keyword_results = keyword_search(query, limit=fetch_limit)
    semantic_results = semantic_search(query, limit=fetch_limit)

    if method == "weighted":
        combined = weighted_combination(
            keyword_results,
            semantic_results,
            keyword_weight,
            semantic_weight
        )
    else:
        # Default to RRF
        combined = reciprocal_rank_fusion([keyword_results, semantic_results])

    return combined[:limit]


def filtered_search(
    query: str,
    keyword_filter: str = None,
    limit: int = None
) -> list[dict]:
    """
    Use keyword search to filter, then semantic search to rank.

    Args:
        query: Semantic query
        keyword_filter: Required keyword terms
        limit: Maximum results

    Returns:
        Filtered and ranked results
    """
    limit = limit or TOP_K_RESULTS

    if keyword_filter:
        # Get candidate documents from keyword search
        keyword_results = keyword_search(keyword_filter, limit=limit * 3)
        candidate_doc_ids = {r["doc_id"] for r in keyword_results}

        # Run semantic search
        semantic_results = semantic_search(query, limit=limit * 3)

        # Filter to only candidates
        filtered = [r for r in semantic_results if r["doc_id"] in candidate_doc_ids]
        return filtered[:limit]
    else:
        return semantic_search(query, limit=limit)
