from __future__ import annotations

import asyncio
import shutil
import wave
from pathlib import Path


class AudioConversionError(RuntimeError):
    """Raised when Telegram audio cannot be converted for the model."""


async def convert_to_model_wav(source: Path, destination: Path) -> None:
    """Convert audio to 16 kHz, mono, 16-bit PCM WAV using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioConversionError("ffmpeg is required to process audio messages")

    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise AudioConversionError(
            f"ffmpeg could not convert the audio: {detail or 'unknown error'}"
        )

    try:
        with wave.open(str(destination), "rb") as converted:
            valid = (
                converted.getframerate() == 16000
                and converted.getnchannels() == 1
                and converted.getsampwidth() == 2
                and converted.getcomptype() == "NONE"
            )
    except (OSError, wave.Error) as exc:
        raise AudioConversionError("ffmpeg produced an invalid WAV file") from exc
    if not valid:
        raise AudioConversionError(
            "ffmpeg did not produce 16 kHz mono 16-bit PCM WAV audio"
        )
