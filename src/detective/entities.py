"""Entity extraction from documents."""

import re
from typing import Callable

from ..db.models import (
    create_entity,
    get_document,
    get_entities_by_type,
    get_entities_for_document,
)
from ..config import LLM_MODEL

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# Common patterns for entity extraction
PATTERNS = {
    "DATE": [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or M/D/YY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',  # ISO format
    ],
    "PHONE": [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
    ],
    "EMAIL": [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ],
}


def extract_entities_regex(text: str) -> list[dict]:
    """
    Extract entities using regex patterns.

    Args:
        text: Text to extract from

    Returns:
        List of entity dicts with type, value, and context
    """
    entities = []

    for entity_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                entities.append({
                    "entity_type": entity_type,
                    "entity_value": match.group().strip(),
                    "context": context.strip()
                })

    return entities


def extract_entities_llm(text: str, model: str = None) -> list[dict]:
    """
    Extract entities using LLM.

    Args:
        text: Text to extract from
        model: LLM model to use

    Returns:
        List of entity dicts
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama is required for LLM extraction")

    model = model or LLM_MODEL

    prompt = """Extract named entities from the following text. For each entity found, provide:
- Type: PERSON, ORGANIZATION, LOCATION, or DATE
- Value: The exact text of the entity
- Context: A brief phrase showing how the entity appears

Format your response as a list, one entity per line:
TYPE: value | context

Text:
"""

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt + text[:3000]  # Limit text length
        }]
    )

    # Parse response
    entities = []
    for line in response["message"]["content"].split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue

        try:
            type_value, context = line.split("|", 1)
            if ":" in type_value:
                entity_type, value = type_value.split(":", 1)
                entity_type = entity_type.strip().upper()
                value = value.strip()

                if entity_type in ("PERSON", "ORGANIZATION", "LOCATION", "DATE", "ORG"):
                    if entity_type == "ORG":
                        entity_type = "ORGANIZATION"

                    entities.append({
                        "entity_type": entity_type,
                        "entity_value": value,
                        "context": context.strip()
                    })
        except ValueError:
            continue

    return entities


def extract_entities(
    text: str,
    use_llm: bool = False,
    model: str = None
) -> list[dict]:
    """
    Extract entities from text.

    Args:
        text: Text to extract from
        use_llm: Use LLM for extraction (slower but better for names)
        model: LLM model to use

    Returns:
        List of entity dicts
    """
    entities = extract_entities_regex(text)

    if use_llm:
        llm_entities = extract_entities_llm(text, model=model)
        # Merge, avoiding duplicates
        seen = {(e["entity_type"], e["entity_value"]) for e in entities}
        for entity in llm_entities:
            key = (entity["entity_type"], entity["entity_value"])
            if key not in seen:
                entities.append(entity)
                seen.add(key)

    return entities


def extract_entities_for_document(
    doc_id: str,
    use_llm: bool = False,
    model: str = None,
    progress_callback: Callable[[str], None] = None
) -> int:
    """
    Extract entities from a document and store them.

    Args:
        doc_id: Document ID
        use_llm: Use LLM for extraction
        model: LLM model to use
        progress_callback: Optional progress callback

    Returns:
        Number of entities extracted
    """
    doc = get_document(doc_id)
    if not doc or not doc.get("extracted_text"):
        return 0

    if progress_callback:
        progress_callback(f"Extracting entities from {doc_id}")

    text = doc["extracted_text"]
    document_id = doc["id"]

    # Check if entities already exist
    existing = get_entities_for_document(document_id)
    if existing:
        return len(existing)

    entities = extract_entities(text, use_llm=use_llm, model=model)

    for entity in entities:
        create_entity(
            document_id=document_id,
            entity_type=entity["entity_type"],
            entity_value=entity["entity_value"],
            context=entity["context"]
        )

    return len(entities)


def get_top_entities(entity_type: str, limit: int = 50) -> list[dict]:
    """
    Get most frequent entities of a type.

    Args:
        entity_type: Entity type (PERSON, ORGANIZATION, etc.)
        limit: Maximum results

    Returns:
        List of entities with counts
    """
    return get_entities_by_type(entity_type, limit=limit)
