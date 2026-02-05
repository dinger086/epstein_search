"""Timeline analysis and date-based search."""

import re
from datetime import datetime, date
from typing import Optional

from ..db.connection import get_connection
from ..db.models import get_document


# Date parsing patterns
DATE_PATTERNS = [
    (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%m/%d/%Y'),
    (r'(\d{1,2})/(\d{1,2})/(\d{2})', '%m/%d/%y'),
    (r'(\d{1,2})-(\d{1,2})-(\d{4})', '%m-%d-%Y'),
    (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
]

MONTH_NAMES = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9,
    'oct': 10, 'nov': 11, 'dec': 12
}


def parse_date(date_str: str) -> Optional[date]:
    """
    Parse a date string into a date object.

    Args:
        date_str: Date string in various formats

    Returns:
        date object or None if parsing fails
    """
    date_str = date_str.strip()

    # Try standard patterns
    for pattern, fmt in DATE_PATTERNS:
        match = re.match(pattern, date_str)
        if match:
            try:
                return datetime.strptime(match.group(), fmt).date()
            except ValueError:
                continue

    # Try month name patterns
    month_pattern = r'(?:(\w+)\s+(\d{1,2}),?\s+(\d{4}))|(?:(\d{1,2})\s+(\w+)\s+(\d{4}))'
    match = re.search(month_pattern, date_str, re.IGNORECASE)
    if match:
        groups = match.groups()
        try:
            if groups[0]:  # Month Day, Year
                month_name, day, year = groups[0], groups[1], groups[2]
                month = MONTH_NAMES.get(month_name.lower())
                if month:
                    return date(int(year), month, int(day))
            elif groups[3]:  # Day Month Year
                day, month_name, year = groups[3], groups[4], groups[5]
                month = MONTH_NAMES.get(month_name.lower())
                if month:
                    return date(int(year), month, int(day))
        except (ValueError, KeyError):
            pass

    return None


def extract_dates(text: str) -> list[dict]:
    """
    Extract all dates from text.

    Args:
        text: Text to extract dates from

    Returns:
        List of dicts with date_str, parsed_date, and context
    """
    dates = []

    # Combined pattern for various date formats
    patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_str = match.group()
            parsed = parse_date(date_str)

            # Get context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()

            dates.append({
                "date_str": date_str,
                "parsed_date": parsed,
                "context": context
            })

    # Sort by parsed date
    dates.sort(key=lambda x: x["parsed_date"] or date.min)
    return dates


def timeline_search(
    start_date: date | str = None,
    end_date: date | str = None,
    limit: int = 100
) -> list[dict]:
    """
    Find documents with dates in a range.

    Args:
        start_date: Start of date range
        end_date: End of date range
        limit: Maximum results

    Returns:
        List of documents with matching dates
    """
    conn = get_connection()

    # Parse string dates
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)

    # Get DATE entities
    sql = """
        SELECT e.entity_value, e.context, d.doc_id, d.id as document_id
        FROM entities e
        JOIN documents d ON e.document_id = d.id
        WHERE e.entity_type = 'DATE'
        ORDER BY d.doc_id
        LIMIT ?
    """

    rows = conn.execute(sql, (limit * 10,)).fetchall()  # Get extra to filter

    results = []
    for row in rows:
        parsed = parse_date(row["entity_value"])
        if parsed:
            # Filter by date range
            if start_date and parsed < start_date:
                continue
            if end_date and parsed > end_date:
                continue

            results.append({
                "doc_id": row["doc_id"],
                "document_id": row["document_id"],
                "date_str": row["entity_value"],
                "parsed_date": parsed,
                "context": row["context"]
            })

    # Sort by date
    results.sort(key=lambda x: x["parsed_date"])
    return results[:limit]


def get_date_distribution(
    start_year: int = None,
    end_year: int = None
) -> dict:
    """
    Get distribution of documents by year/month.

    Args:
        start_year: Start year filter
        end_year: End year filter

    Returns:
        Dict with year -> month -> count
    """
    conn = get_connection()

    sql = """
        SELECT e.entity_value
        FROM entities e
        WHERE e.entity_type = 'DATE'
    """

    rows = conn.execute(sql).fetchall()

    distribution = {}
    for row in rows:
        parsed = parse_date(row["entity_value"])
        if parsed:
            year = parsed.year
            month = parsed.month

            # Apply filters
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue

            if year not in distribution:
                distribution[year] = {}
            if month not in distribution[year]:
                distribution[year][month] = 0
            distribution[year][month] += 1

    return distribution
