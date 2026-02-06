"""Video text extraction via audio transcription."""

import subprocess
import tempfile
from pathlib import Path

from .audio import transcribe_audio


def extract_text_from_video(path: str | Path) -> tuple[str, dict]:
    """Extract text from a video by transcribing its audio track.

    Uses ffmpeg to extract audio to a temp WAV (16kHz mono),
    then passes it to transcribe_audio().

    Returns (text, metadata) with duration_seconds and language.
    """
    path = str(path)
    tmp_wav = None

    try:
        # Create temp WAV file
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()

        # Extract audio with ffmpeg: 16kHz mono WAV
        result = subprocess.run(
            [
                "ffmpeg", "-i", path,
                "-vn",                    # no video
                "-acodec", "pcm_s16le",   # 16-bit PCM
                "-ar", "16000",           # 16kHz
                "-ac", "1",               # mono
                "-y",                     # overwrite
                tmp_wav.name,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            # Check if the video has no audio track
            if "does not contain any stream" in result.stderr or \
               "Output file is empty" in result.stderr:
                return "", {"duration_seconds": 0, "language": ""}
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        # Check if the output file has content
        if Path(tmp_wav.name).stat().st_size < 100:
            return "", {"duration_seconds": 0, "language": ""}

        return transcribe_audio(tmp_wav.name)

    finally:
        if tmp_wav is not None:
            try:
                Path(tmp_wav.name).unlink(missing_ok=True)
            except OSError:
                pass
