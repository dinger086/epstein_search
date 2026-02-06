"""Litigation support file parsing."""

import re
from pathlib import Path

# Concordance delimiters
THORN = "\u00fe"       # þ - field delimiter
PILCROW = "\u00b6"     # ¶ - text qualifier

_ENCODINGS = ("utf-8", "latin-1", "cp1252", "utf-16")


def extract_text_from_litigation(path: str | Path) -> tuple[str, dict]:
    """Extract text from litigation support files.

    Handles .DAT (Concordance), .OPT (Opticon), and generic text
    formats (.DII, .LST, .LFP).

    Returns (text, metadata).
    """
    path = Path(path)
    ext = path.suffix.lower()

    raw = _read_with_encoding(path)
    if raw is None:
        return "", {"format": ext, "error": "could not decode file"}

    if ext == ".dat":
        return _parse_dat(raw, path)
    elif ext == ".opt":
        return _parse_opt(raw, path)
    else:
        return _parse_generic(raw, path)


def _read_with_encoding(path: Path) -> str | None:
    """Try multiple encodings to read a file."""
    for enc in _ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def _parse_dat(raw: str, path: Path) -> tuple[str, dict]:
    """Parse Concordance DAT format (thorn/pilcrow delimiters)."""
    lines = raw.splitlines()
    records = []
    header_fields = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Strip pilcrow qualifiers and split by thorn
        line = line.strip(PILCROW)
        fields = line.split(f"{PILCROW}{THORN}{PILCROW}")

        if i == 0:
            # First line is usually the header
            header_fields = [f.strip() for f in fields]
            continue

        if header_fields:
            parts = []
            for j, val in enumerate(fields):
                val = val.strip()
                if val:
                    field_name = header_fields[j] if j < len(header_fields) else f"FIELD_{j}"
                    parts.append(f"{field_name}: {val}")
            if parts:
                records.append("\n".join(parts))
        else:
            # No header, just output fields
            records.append("\t".join(f.strip() for f in fields))

    text = "\n\n".join(records)
    metadata = {"format": "concordance_dat", "record_count": len(records)}
    return text, metadata


def _parse_opt(raw: str, path: Path) -> tuple[str, dict]:
    """Parse Opticon OPT format (comma-delimited doc ID mappings)."""
    lines = raw.splitlines()
    records = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # OPT format: DocID,Volume,ImagePath,DocBreak,FolderBreak,PageCount,...
        parts = [p.strip() for p in line.split(",")]
        records.append("\t".join(parts))

    text = "\n".join(records)
    metadata = {"format": "opticon_opt", "record_count": len(records)}
    return text, metadata


def _parse_generic(raw: str, path: Path) -> tuple[str, dict]:
    """Generic text cleanup for .DII, .LST, .LFP files."""
    # Replace control characters (except newline/tab) with spaces
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    # Collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    line_count = len(cleaned.splitlines()) if cleaned else 0
    metadata = {"format": path.suffix.lower().lstrip("."), "line_count": line_count}
    return cleaned, metadata
