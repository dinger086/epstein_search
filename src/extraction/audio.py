"""Audio transcription using faster-whisper."""

from pathlib import Path

from ..config import WHISPER_MODEL_SIZE, WHISPER_LANGUAGE

# Lazy-loaded singleton model
_model = None


def _get_model():
    """Get or create the Whisper model (singleton, lazy-loaded)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
    return _model


def transcribe_audio(path: str | Path) -> tuple[str, dict]:
    """Transcribe an audio file using faster-whisper.

    Returns (text, metadata) where text has timestamps like [MM:SS],
    and metadata contains duration_seconds and language.
    """
    path = str(path)
    model = _get_model()

    segments, info = model.transcribe(
        path,
        language=WHISPER_LANGUAGE,
        vad_filter=True,
    )

    lines = []
    for segment in segments:
        minutes = int(segment.start // 60)
        seconds = int(segment.start % 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {segment.text.strip()}")

    text = "\n".join(lines)
    metadata = {
        "duration_seconds": round(info.duration, 1),
        "language": info.language,
    }
    return text, metadata
