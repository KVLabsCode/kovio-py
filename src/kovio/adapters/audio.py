"""Audio adapters — how the SDK speaks (text-to-speech) on the robot.

On the Unitree G1 the voice service lives on the main-control computer and is
driven over DDS. Only the compiled ``kovio_tts`` binary (built against
unitree_sdk2 **2.0.2**) can reach it: that is the sole SDK build whose
``unitree_api`` message types match G1 firmware v1.5.3 — the Python SDK's types
don't match, so every Python audio call returns error 3102. So the default
adapter shells out to that binary, mirroring how ChromiumKioskAdapter shells
out to Chromium. On a dev machine (or any robot without the binary) we fall
back to a no-op logger so nothing crashes.

Reused by the dashboard "type text to speak" feature (via SessionStreamer) and,
later, the conversational loop (rt/audio_msg ASR -> LLM -> speak()).
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from abc import ABC, abstractmethod

log = logging.getLogger("kovio.audio")

_WATCH_TIMEOUT_S = 30.0


class AudioAdapter(ABC):
    """Speak text on the robot."""

    @abstractmethod
    def speak(self, text: str, volume: int | None = None) -> None:
        """Speak ``text`` (English TTS). ``volume`` 0-100, or None for default."""

    # --- factory ---

    @staticmethod
    def logger() -> "AudioAdapter":
        """No-op adapter that just logs. Useful on dev machines / no speaker."""
        return LoggingAudioAdapter()


class KovioTtsAdapter(AudioAdapter):
    """Speak via the compiled ``kovio_tts`` binary (G1 firmware-matched SDK 2.0.2).

    Fire-and-forget: ``TtsMaker`` is synthesized+played server-side on the robot,
    so we ``Popen`` the binary and let a small watcher thread log its exit code
    without blocking the caller — the 5s session poll must stay responsive. A
    non-zero exit (the binary surfaces 3102) means the robot isn't in Normal
    mode or the voice service is down; we log it and move on.
    """

    def __init__(
        self,
        binary: str,
        iface: str = "eth0",
        lib_dir: str | None = None,
        default_volume: int = 85,
    ) -> None:
        self._binary = binary
        self._iface = iface
        self._default_volume = _clamp_volume(default_volume)
        # Bundled cyclonedds .so lives next to the binary's install; make it
        # resolvable without requiring a system-wide ldconfig entry.
        self._env = dict(os.environ)
        if lib_dir:
            existing = self._env.get("LD_LIBRARY_PATH", "")
            self._env["LD_LIBRARY_PATH"] = (
                f"{lib_dir}:{existing}" if existing else lib_dir
            )

    def speak(self, text: str, volume: int | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        vol = self._default_volume if volume is None else _clamp_volume(volume)
        argv = [self._binary, self._iface, text, str(vol)]
        try:
            proc = subprocess.Popen(
                argv,
                env=self._env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as e:
            log.warning("[audio] could not launch kovio_tts (%s): %s", self._binary, e)
            return
        log.info("[audio] speaking (%d chars, vol=%d, pid=%s)", len(text), vol, proc.pid)
        threading.Thread(
            target=self._watch, args=(proc, text), name="kovio-tts-watch", daemon=True
        ).start()

    def _watch(self, proc: subprocess.Popen, text: str) -> None:
        try:
            out, _ = proc.communicate(timeout=_WATCH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.warning("[audio] kovio_tts timed out after %.0fs; killed", _WATCH_TIMEOUT_S)
            return
        if proc.returncode == 0:
            log.info("[audio] spoke ok: %r", text[:60])
        else:
            # Non-zero: most commonly 3102 (robot not in Normal mode / voice
            # service down). Surface the binary's stdout to make that obvious.
            log.warning(
                "[audio] kovio_tts exit=%s output=%s",
                proc.returncode,
                (out or "").strip()[:200],
            )


class LoggingAudioAdapter(AudioAdapter):
    """No-op audio. Just logs. Use on dev machines or robots without a speaker."""

    def speak(self, text: str, volume: int | None = None) -> None:
        log.info("[audio] SPEAK (vol=%s): %s", volume, text)


class JblAudioPlayer:
    """Play pre-rendered WAV (the cloud greeting) out a Bluetooth speaker.

    Distinct from AudioAdapter.speak(): the greeting voice is synthesized in the
    cloud (ElevenLabs), so here we just push finished WAV bytes to a PulseAudio
    sink via ``paplay``. The sink is the JBL's A2DP sink, e.g.
    ``bluez_sink.78_66_F3_81_16_D2.a2dp_sink``.

    A2DP sinks suspend when idle and can clip the first ~200ms on resume, so we
    play a short burst of silence first to wake the sink before the greeting.
    Fire-and-forget with a watcher thread, mirroring KovioTtsAdapter, so the 5s
    session poll never blocks on playback.
    """

    def __init__(self, sink: str, default_volume: int = 85) -> None:
        self._sink = sink
        self._default_volume = _clamp_volume(default_volume)

    def play_wav(self, data: bytes, volume: int | None = None) -> None:
        if not data:
            return
        vol = self._default_volume if volume is None else _clamp_volume(volume)
        # paplay --volume is linear 0..65536.
        pa_vol = str(int(round(vol / 100 * 65536)))
        try:
            self._wake_sink()
            proc = subprocess.Popen(
                ["paplay", "--device", self._sink, "--volume", pa_vol],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            log.warning("[audio] could not launch paplay (sink=%s): %s", self._sink, e)
            return
        log.info("[audio] playing greeting (%d bytes, vol=%d) -> %s",
                 len(data), vol, self._sink)
        threading.Thread(
            target=self._watch, args=(proc, data), name="kovio-jbl-play", daemon=True
        ).start()

    def _watch(self, proc: subprocess.Popen, data: bytes) -> None:
        try:
            _out, err = proc.communicate(input=data, timeout=_WATCH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.warning("[audio] paplay timed out after %.0fs; killed", _WATCH_TIMEOUT_S)
            return
        if proc.returncode != 0:
            log.warning(
                "[audio] paplay exit=%s err=%s",
                proc.returncode, (err or b"").decode("utf-8", "replace").strip()[:200],
            )

    def _wake_sink(self) -> None:
        """Push ~0.3s of silence so the suspended A2DP sink is awake before the
        real audio — otherwise the first word can be clipped on resume."""
        # 0.3s of 16-bit mono silence at 22.05kHz, wrapped as WAV.
        frames = b"\x00\x00" * int(22050 * 0.3)
        import io as _io
        import wave as _wave

        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(frames)
        try:
            subprocess.run(
                ["paplay", "--device", self._sink],
                input=buf.getvalue(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            pass  # best-effort wake; the greeting still plays


def _clamp_volume(value: int) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 85


def make_audio_adapter(name: str | None = None) -> AudioAdapter:
    """Construct the audio adapter.

    Selection order: explicit ``name`` -> ``KOVIO_AUDIO`` env -> auto (kovio_tts
    when ``KOVIO_TTS_BIN`` points at a runnable binary, else the logger).
    Recognized names: 'kovio_tts', 'logger', 'none'.

    Env:
      KOVIO_TTS_BIN     path to the compiled kovio_tts binary (required for TTS)
      KOVIO_TTS_IFACE   wired DDS interface (default 'eth0')
      KOVIO_TTS_LIBDIR  dir holding the bundled cyclonedds .so (LD_LIBRARY_PATH)
      KOVIO_TTS_VOLUME  default volume 0-100 (default 85)
    """
    name = name or os.getenv("KOVIO_AUDIO")
    binary = os.getenv("KOVIO_TTS_BIN")
    iface = os.getenv("KOVIO_TTS_IFACE", "eth0")
    lib_dir = os.getenv("KOVIO_TTS_LIBDIR")
    default_volume = _clamp_volume(os.getenv("KOVIO_TTS_VOLUME", "85"))

    if name in ("logger", "none"):
        return LoggingAudioAdapter()

    if name == "kovio_tts" or (name is None and binary):
        if not binary:
            raise ValueError("KOVIO_AUDIO=kovio_tts but KOVIO_TTS_BIN is not set.")
        if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            log.warning(
                "[audio] kovio_tts binary not found/executable at %s; using logger",
                binary,
            )
            return LoggingAudioAdapter()
        return KovioTtsAdapter(
            binary=binary, iface=iface, lib_dir=lib_dir, default_volume=default_volume
        )

    # No name and no binary configured: dev machine / camera-only robot.
    return LoggingAudioAdapter()


def make_audio_player() -> "JblAudioPlayer | None":
    """Construct the Bluetooth greeting player, or None if not configured.

    Env:
      KOVIO_JBL_SINK    PulseAudio sink name of the paired speaker, e.g.
                        ``bluez_sink.78_66_F3_81_16_D2.a2dp_sink``. Unset =>
                        greeting audio is simply not played (feature disabled).
      KOVIO_JBL_VOLUME  default playback volume 0-100 (default 85)
    """
    sink = os.getenv("KOVIO_JBL_SINK")
    if not sink:
        return None
    volume = _clamp_volume(os.getenv("KOVIO_JBL_VOLUME", "85"))
    return JblAudioPlayer(sink=sink, default_volume=volume)
