"""File type registry for non-PDF document processing."""

from pathlib import Path

# Extension -> file type mapping
EXTENSION_MAP: dict[str, str] = {
    # PDF
    ".pdf": "pdf",
    # Video
    ".mp4": "video",
    ".avi": "video",
    ".mov": "video",
    ".wmv": "video",
    ".mkv": "video",
    ".flv": "video",
    ".webm": "video",
    ".mpg": "video",
    ".mpeg": "video",
    ".m4v": "video",
    # Audio
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".wma": "audio",
    ".m4a": "audio",
    # Spreadsheet
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".csv": "spreadsheet",
    # Litigation support
    ".dat": "litigation",
    ".opt": "litigation",
    ".dii": "litigation",
    ".lst": "litigation",
    ".lfp": "litigation",
}

ALL_EXTENSIONS: set[str] = set(EXTENSION_MAP.keys())

FILE_TYPES: set[str] = set(EXTENSION_MAP.values())


def classify_file(path: str | Path) -> str:
    """Return the file type string for a given path, or 'unknown'."""
    ext = Path(path).suffix.lower()
    return EXTENSION_MAP.get(ext, "unknown")
