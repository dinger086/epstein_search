"""Find connections between documents and entities."""

from collections import defaultdict

from ..db.connection import get_connection
from ..db.models import get_documents_with_entity, get_entities_for_document, get_document


def find_connections(entity_value: str) -> dict:
    """
    Find all documents mentioning an entity and related entities.

    Args:
        entity_value: Entity to search for

    Returns:
        Dict with documents and related entities
    """
    # Find documents containing this entity
    documents = get_documents_with_entity(entity_value)

    # Find other entities in those documents
    related_entities = defaultdict(set)
    document_ids = []

    for doc in documents:
        document_ids.append(doc["doc_id"])
        entities = get_entities_for_document(doc["id"])

        for entity in entities:
            if entity["entity_value"] != entity_value:
                related_entities[entity["entity_type"]].add(entity["entity_value"])

    return {
        "entity": entity_value,
        "document_count": len(documents),
        "documents": document_ids,
        "related_entities": {
            etype: list(values)
            for etype, values in related_entities.items()
        }
    }


def find_co_occurring_entities(
    entity_value: str,
    entity_type: str = None,
    limit: int = 20
) -> list[dict]:
    """
    Find entities that frequently appear with a given entity.

    Args:
        entity_value: Entity to find co-occurrences for
        entity_type: Filter to specific entity type
        limit: Maximum results

    Returns:
        List of co-occurring entities with counts
    """
    conn = get_connection()

    if entity_type:
        sql = """
            SELECT e2.entity_value, e2.entity_type, COUNT(*) as count
            FROM entities e1
            JOIN entities e2 ON e1.document_id = e2.document_id
            WHERE e1.entity_value = ?
              AND e2.entity_value != ?
              AND e2.entity_type = ?
            GROUP BY e2.entity_value, e2.entity_type
            ORDER BY count DESC
            LIMIT ?
        """
        rows = conn.execute(sql, (entity_value, entity_value, entity_type, limit)).fetchall()
    else:
        sql = """
            SELECT e2.entity_value, e2.entity_type, COUNT(*) as count
            FROM entities e1
            JOIN entities e2 ON e1.document_id = e2.document_id
            WHERE e1.entity_value = ?
              AND e2.entity_value != ?
            GROUP BY e2.entity_value, e2.entity_type
            ORDER BY count DESC
            LIMIT ?
        """
        rows = conn.execute(sql, (entity_value, entity_value, limit)).fetchall()

    return [dict(row) for row in rows]


def find_shared_connections(entity1: str, entity2: str) -> dict:
    """
    Find documents and entities shared between two entities.

    Args:
        entity1: First entity
        entity2: Second entity

    Returns:
        Dict with shared documents and entities
    """
    conn = get_connection()

    # Find documents containing both entities
    sql = """
        SELECT DISTINCT d.doc_id, d.id
        FROM documents d
        JOIN entities e1 ON e1.document_id = d.id
        JOIN entities e2 ON e2.document_id = d.id
        WHERE e1.entity_value = ? AND e2.entity_value = ?
    """
    rows = conn.execute(sql, (entity1, entity2)).fetchall()

    shared_docs = [row["doc_id"] for row in rows]
    shared_doc_ids = [row["id"] for row in rows]

    # Find other entities in those shared documents
    shared_entities = defaultdict(set)
    for doc_id in shared_doc_ids:
        entities = get_entities_for_document(doc_id)
        for entity in entities:
            if entity["entity_value"] not in (entity1, entity2):
                shared_entities[entity["entity_type"]].add(entity["entity_value"])

    return {
        "entity1": entity1,
        "entity2": entity2,
        "shared_documents": shared_docs,
        "shared_document_count": len(shared_docs),
        "shared_entities": {
            etype: list(values)
            for etype, values in shared_entities.items()
        }
    }


def build_entity_network(
    seed_entity: str,
    depth: int = 2,
    limit_per_level: int = 10
) -> dict:
    """
    Build a network of connected entities starting from a seed.

    Args:
        seed_entity: Starting entity
        depth: How many levels to explore
        limit_per_level: Max entities per level

    Returns:
        Dict representing the network
    """
    visited = {seed_entity}
    network = {
        "nodes": [{
            "id": seed_entity,
            "type": "seed",
            "level": 0
        }],
        "edges": []
    }

    current_level = [seed_entity]

    for level in range(1, depth + 1):
        next_level = []

        for entity in current_level:
            co_occurring = find_co_occurring_entities(
                entity,
                entity_type="PERSON",
                limit=limit_per_level
            )

            for co_entity in co_occurring:
                target = co_entity["entity_value"]
                count = co_entity["count"]

                # Add edge
                network["edges"].append({
                    "source": entity,
                    "target": target,
                    "weight": count
                })

                # Add node if not visited
                if target not in visited:
                    visited.add(target)
                    network["nodes"].append({
                        "id": target,
                        "type": co_entity["entity_type"],
                        "level": level
                    })
                    next_level.append(target)

        current_level = next_level[:limit_per_level]

    return network
