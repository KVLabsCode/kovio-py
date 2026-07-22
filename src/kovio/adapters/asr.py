"""Speech-to-text on the robot — local faster-whisper.

Push-to-talk transcription runs on-device so the raw mic audio never leaves the
robot: only the recognized text is sent to the cloud for a reply. The model is
loaded lazily on first use (a few seconds, one-time) and reused. On a dev
machine without faster-whisper installed we fall back to a no-op that returns
"" so the conversation path degrades quietly instead of crashing.

Input is raw 16 kHz mono 16-bit little-endian PCM (exactly the G1 mic multicast
format) — fed to the model as a float32 numpy array so no ffmpeg/PyAV decode is
needed.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

log = logging.getLogger("kovio.asr")


class AsrAdapter(ABC):
    """Transcribe 16 kHz mono 16-bit PCM to text."""

    @abstractmethod
    def transcribe(self, pcm16: bytes) -> str:
        """Return recognized text (may be "" for silence/failure)."""


class WhisperAsr(AsrAdapter):
    """Local faster-whisper. Model loaded on first transcribe(), then cached."""

    def __init__(self, model_name: str = "base.en", compute_type: str = "int8") -> None:
        self._model_name = model_name
        self._compute_type = compute_type
        self._model = None  # lazy

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.warning("[asr] faster_whisper not installed; transcription disabled")
            return False
        try:
            log.info("[asr] loading whisper model %s (%s)…",
                     self._model_name, self._compute_type)
            self._model = WhisperModel(
                self._model_name, device="cpu", compute_type=self._compute_type
            )
            log.info("[asr] model ready")
            return True
        except Exception:
            log.exception("[asr] model load failed; transcription disabled")
            return False

    def transcribe(self, pcm16: bytes) -> str:
        if not pcm16 or not self._ensure_model():
            return ""
        try:
            import numpy as np

            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            # vad_filter drops non-speech (mic idle-noise, room hum) inside the
            # clip so leading/trailing ambient can't produce hallucinated words.
            segments, _info = self._model.transcribe(
                audio, language="en", beam_size=1, vad_filter=True
            )
            text = " ".join(seg.text for seg in segments).strip()
            log.info("[asr] transcribed %d samples -> %r", len(audio), text[:80])
            return text
        except Exception:
            log.exception("[asr] transcription failed")
            return ""


class NullAsr(AsrAdapter):
    """No-op ASR (dev machines / conversation disabled)."""

    def transcribe(self, pcm16: bytes) -> str:
        log.info("[asr] NULL transcribe (%d bytes) -> ''", len(pcm16))
        return ""


def make_asr(name: str | None = None) -> AsrAdapter | None:
    """Construct the ASR adapter, or None when push-to-talk isn't enabled.

    Env:
      KOVIO_ASR         'whisper' to enable, 'none'/unset to disable
      KOVIO_ASR_MODEL   faster-whisper model (default 'base.en')
      KOVIO_ASR_COMPUTE ctranslate2 compute type (default 'int8')
    """
    name = name or os.getenv("KOVIO_ASR")
    if not name or name == "none":
        return None
    if name == "whisper":
        return WhisperAsr(
            model_name=os.getenv("KOVIO_ASR_MODEL", "base.en"),
            compute_type=os.getenv("KOVIO_ASR_COMPUTE", "int8"),
        )
    log.warning("[asr] unknown KOVIO_ASR=%r; disabling", name)
    return None
