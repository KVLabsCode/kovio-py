"""Microphone capture — the G1 mic stream, read passively.

The G1 publishes raw 16 kHz mono 16-bit PCM on a UDP multicast group (see the
G1 audio spec). We just JOIN and receive — no SDK client, no DDS request, so
this can never trip the robot into dev mode. ``record_utterance`` captures one
push-to-talk turn: it waits briefly for speech to begin, then records until a
short trailing silence (the person stopped talking) or a hard time cap.

Energy-based endpointing keeps it dependency-free (no webrtcvad); it's tuned for
"press Listen, say a sentence, done", not open-mic diarization.
"""
from __future__ import annotations

import audioop
import logging
import os
import socket
import struct
import time

log = logging.getLogger("kovio.mic")

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit


class MicCapture:
    """Receive the G1 mic multicast and record single utterances."""

    def __init__(
        self,
        group: str = "239.168.123.161",
        port: int = 5555,
        iface_ip: str = "192.168.123.164",
        silence_rms: int = 350,
    ) -> None:
        self._group = group
        self._port = port
        self._iface_ip = iface_ip
        # RMS at or below this counts as silence for endpointing. Measured on
        # this unit: true silence ~0, idle-noise blips to ~200, speech peaks
        # 3000+. 350 sits above the noise blips. whisper's own vad_filter is the
        # backstop, so this only needs to roughly bound the recording.
        self._silence_rms = silence_rms

    def _open(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", self._port))
        mreq = struct.pack(
            "4s4s", socket.inet_aton(self._group), socket.inet_aton(self._iface_ip)
        )
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        return s

    def record_utterance(
        self,
        max_seconds: float = 12.0,
        start_timeout: float = 6.0,
        trailing_silence: float = 1.2,
        min_speech_seconds: float = 0.3,
    ) -> bytes:
        """Capture one utterance as raw PCM16 bytes (may be b"" if nothing said).

        Records until ``trailing_silence`` of quiet follows detected speech, or
        ``max_seconds`` elapses. Gives up (returns b"") if no speech starts
        within ``start_timeout``.
        """
        try:
            sock = self._open()
        except OSError as e:
            log.warning("[mic] cannot open multicast %s:%s (%s)",
                        self._group, self._port, e)
            return b""
        sock.settimeout(0.5)
        pcm = bytearray()
        started = False
        speech_bytes = 0
        voiced_run = 0  # consecutive voiced chunks, to debounce noise blips
        t0 = time.time()
        last_voice = t0
        try:
            while True:
                now = time.time()
                if now - t0 > max_seconds:
                    break
                if not started and now - t0 > start_timeout:
                    log.info("[mic] no speech within %.1fs; giving up", start_timeout)
                    break
                try:
                    chunk = sock.recvfrom(65535)[0]
                except socket.timeout:
                    if started and now - last_voice > trailing_silence:
                        break
                    continue
                rms = audioop.rms(chunk, _BYTES_PER_SAMPLE) if chunk else 0
                voiced = rms > self._silence_rms
                voiced_run = voiced_run + 1 if voiced else 0
                if voiced:
                    last_voice = now
                    # Require two consecutive voiced chunks to begin, so a lone
                    # noise blip near the threshold can't open the recording.
                    if not started and voiced_run >= 2:
                        started = True
                        log.info("[mic] speech detected (rms=%d); recording", rms)
                if started:
                    pcm += chunk
                    if voiced:
                        speech_bytes += len(chunk)
                    if now - last_voice > trailing_silence:
                        break
        finally:
            try:
                mreq = struct.pack(
                    "4s4s", socket.inet_aton(self._group),
                    socket.inet_aton(self._iface_ip),
                )
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
            except OSError:
                pass
            sock.close()

        speech_s = speech_bytes / (_SAMPLE_RATE * _BYTES_PER_SAMPLE)
        if speech_s < min_speech_seconds:
            log.info("[mic] too little speech (%.2fs); discarding", speech_s)
            return b""
        log.info("[mic] captured %.1fs (%d bytes)",
                 len(pcm) / (_SAMPLE_RATE * _BYTES_PER_SAMPLE), len(pcm))
        return bytes(pcm)


def make_mic_capture() -> "MicCapture | None":
    """Construct the mic capture, or None when push-to-talk isn't enabled.

    Enabled whenever KOVIO_ASR is set (mic + ASR go together). Env:
      KOVIO_MIC_GROUP    multicast group (default 239.168.123.161)
      KOVIO_MIC_PORT     multicast port (default 5555)
      KOVIO_MIC_IFACE_IP local iface IP to join from (default 192.168.123.164)
      KOVIO_MIC_SILENCE  RMS silence threshold (default 200)
    """
    if not os.getenv("KOVIO_ASR") or os.getenv("KOVIO_ASR") == "none":
        return None
    return MicCapture(
        group=os.getenv("KOVIO_MIC_GROUP", "239.168.123.161"),
        port=int(os.getenv("KOVIO_MIC_PORT", "5555")),
        iface_ip=os.getenv("KOVIO_MIC_IFACE_IP", "192.168.123.164"),
        silence_rms=int(os.getenv("KOVIO_MIC_SILENCE", "350")),
    )
