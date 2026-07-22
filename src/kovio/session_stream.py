"""SessionStreamer — live-view uploader for admin sessions.

Polls ``GET /session/v1/current`` with the fleet key every few seconds; while
an admin has a session open for this robot, JPEG-encodes the perception
adapter's latest color frame and POSTs it to the in-RAM relay at
``POST /session/v1/frame``. Frames are only ever the latest single JPEG in
cloud process RAM — nothing is recorded on disk or in the database, and when
no session is open this thread does nothing but the light 5s poll.

Runs alongside CloudEventSink with the same key/url and the same
never-crash-the-agent posture: every network error is swallowed and retried on
the next tick.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from urllib import error, parse, request

from .cloud import _DEFAULT_TIMEOUT, _get_json

log = logging.getLogger("kovio.session")


class SessionStreamer:
    """Background thread: poll for an open session, stream frames while one is."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        robot_id: str,
        frame_source: Callable[[], "object | None"] | None = None,
        poll_interval_seconds: float = 5.0,
        jpeg_quality: int = 70,
        timeout: float = _DEFAULT_TIMEOUT,
        audience_engine=None,
        speaker=None,
        player=None,
        mic=None,
        asr=None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.robot_id = robot_id
        self._frame_source = frame_source
        self.poll_interval = poll_interval_seconds
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        # V2: while a session is open, the AudienceEngine's moments (passerby /
        # dwell / close_approach) are drained and uploaded alongside the frames,
        # with a sensor-health snapshot so the dashboard can show DEGRADED
        # instead of a silent zero when a sensor dies mid-session.
        self._engine = audience_engine
        # Dashboard-driven TTS: the /current poll may carry a speak_text +
        # speak_nonce; we hand the text to this AudioAdapter and de-dupe on the
        # nonce so the 5s poll never repeats the same utterance.
        self._speaker = speaker
        # Greeting-on-Go: the /current poll may instead carry a speak_audio_url
        # (a cloud-rendered ElevenLabs WAV). When present we fetch it and play it
        # out the Bluetooth speaker via this player, in preference to onboard
        # TTS. Shares the same nonce de-dup as speak_text.
        self._player = player
        self._last_speak_nonce: str | None = None
        # Push-to-talk: when the /current poll carries a listen_nonce we capture
        # one mic utterance (mic), transcribe it on-device (asr), POST the text
        # to /utterance, and play the reply WAV out the speaker (player). The
        # turn runs on its own thread so the 5s poll never blocks on it.
        self._mic = mic
        self._asr = asr
        self._last_listen_nonce: str | None = None
        self._listening = False

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False  # last known session state, for edge logging

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="kovio-session-stream"
        )
        self._thread.start()
        log.info("session.stream.started (poll=%.0fs)", self.poll_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- internals ----

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("session.stream.tick_failed")
            self._stop.wait(self.poll_interval)

    def _tick(self) -> None:
        qs = parse.urlencode({"robot_id": self.robot_id})
        status, payload = _get_json(
            f"{self.api_url}/session/v1/current?{qs}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        # Dashboard TTS rides the same poll — dispatch before the frame work so
        # a speak command lands even on a tick where there's no frame to encode.
        self._maybe_speak(payload)
        # Push-to-talk conversation rides the same poll too.
        self._maybe_listen(payload)

        active = bool(payload and payload.get("active"))
        if active != self._active:
            log.info("session %s", "OPEN — streaming frames" if active else "closed")
            self._active = active
            if self._engine is not None:
                if active:
                    # Fresh session = fresh identity: track ids restart at 1 and
                    # nothing links people across sessions (privacy posture).
                    self._engine.set_encounter_cap(
                        payload.get("encounter_cap_seconds")
                    )
                    self._engine.arm()
                else:
                    self._engine.disarm()
        if not active:
            return

        if self._engine is not None:
            self._post_moments()

        jpeg = self._encode_latest()
        if jpeg is None:
            return
        self._post_frame(jpeg)

    def _maybe_speak(self, payload: "dict | None") -> None:
        """Speak a dashboard-issued utterance from the /current payload, once.

        The cloud surfaces ``speak_text`` + ``speak_nonce`` (and optional
        ``speak_volume``) only while a session is open and a message is pending.
        We de-dupe on the nonce so the recurring 5s poll speaks each message
        exactly once. Never raises — a bad speaker/payload must not stall the
        session loop."""
        if not payload:
            return
        nonce = payload.get("speak_nonce")
        if not nonce or nonce == self._last_speak_nonce:
            return

        # Prefer a cloud-rendered greeting WAV (natural voice, Bluetooth speaker)
        # over onboard text TTS. Both de-dupe on the same nonce.
        audio_url = payload.get("speak_audio_url")
        if audio_url and self._player is not None:
            self._last_speak_nonce = nonce
            try:
                data = self._fetch_audio(audio_url)
                if data:
                    self._player.play_wav(data, payload.get("speak_volume"))
            except Exception:
                log.exception("session.play_audio_failed")
            return

        text = payload.get("speak_text")
        if not text or self._speaker is None:
            return
        self._last_speak_nonce = nonce
        try:
            self._speaker.speak(text, payload.get("speak_volume"))
        except Exception:
            log.exception("session.speak_failed")

    def _maybe_listen(self, payload: "dict | None") -> None:
        """Handle a push-to-talk listening window from the /current payload.

        On each new ``listen_nonce`` we run one conversation turn on a worker
        thread (capture -> transcribe -> reply -> play) so the poll keeps
        running. Only one turn at a time; de-dupes on the nonce. Never raises."""
        if self._mic is None or self._asr is None or not payload:
            return
        nonce = payload.get("listen_nonce")
        if not nonce or nonce == self._last_listen_nonce or self._listening:
            return
        self._last_listen_nonce = nonce
        self._listening = True
        threading.Thread(
            target=self._run_listen_turn, args=(nonce,),
            name="kovio-listen", daemon=True,
        ).start()

    def _run_listen_turn(self, nonce: str) -> None:
        try:
            pcm = self._mic.record_utterance()
            if not pcm:
                return
            text = self._asr.transcribe(pcm)
            if not text:
                return
            wav = self._post_utterance(text, nonce)
            if wav and self._player is not None:
                self._player.play_wav(wav)
        except Exception:
            log.exception("session.listen_turn_failed")
        finally:
            self._listening = False

    def _post_utterance(self, text: str, nonce: str) -> bytes | None:
        """POST recognized text to /utterance; return the reply WAV bytes."""
        import json

        qs = parse.urlencode({"robot_id": self.robot_id})
        body = json.dumps({"text": text, "nonce": nonce}).encode("utf-8")
        req = request.Request(
            f"{self.api_url}/session/v1/utterance?{qs}", data=body, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                reply = resp.headers.get("X-Reply-Text")
                if reply:
                    log.info("session.reply: %r", reply[:80])
                return resp.read()
        except error.HTTPError as e:
            # 409 = session closed; 503 = reply generation off/errored.
            if e.code not in (409, 503):
                log.warning("session.utterance.http_error status=%s", e.code)
        except (error.URLError, TimeoutError, OSError) as e:
            log.warning("session.utterance.network_error %s", e)
        return None

    def _fetch_audio(self, path: str) -> bytes | None:
        """GET the pending greeting WAV (fleet-key auth). ``path`` is the
        relative URL the cloud handed us on the poll."""
        url = path if path.startswith("http") else f"{self.api_url}{path}"
        req = request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except error.HTTPError as e:
            # 404 = greeting expired/superseded between the poll and the fetch.
            if e.code != 404:
                log.warning("session.audio.http_error status=%s", e.code)
        except (error.URLError, TimeoutError, OSError) as e:
            log.warning("session.audio.network_error %s", e)
        return None

    def _encode_latest(self) -> bytes | None:
        if self._frame_source is None:
            return None
        frame = self._frame_source()
        if frame is None:
            return None
        try:
            import cv2
        except ImportError:
            log.warning("session.stream: cv2 unavailable; cannot encode frames")
            return None
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return None
        return buf.tobytes()

    def _post_moments(self) -> None:
        """Upload drained audience moments + sensor health. Posted every tick
        while a session is open (even with zero moments) so the dashboard's
        sensor-health row stays live."""
        import json

        moments = self._engine.drain()
        body = json.dumps(
            {"moments": moments, "sensor": self._engine.health()}
        ).encode("utf-8")
        qs = parse.urlencode({"robot_id": self.robot_id})
        req = request.Request(
            f"{self.api_url}/session/v1/moments?{qs}", data=body, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with request.urlopen(req, timeout=self.timeout):
                pass
        except error.HTTPError as e:
            # 409 = session closed between poll and post; normal, next tick idles.
            if e.code != 409:
                log.warning("session.moments.http_error status=%s", e.code)
                self._engine.requeue(moments)
        except (error.URLError, TimeoutError, OSError) as e:
            log.warning("session.moments.network_error %s", e)
            self._engine.requeue(moments)

    def _post_frame(self, jpeg: bytes) -> None:
        qs = parse.urlencode({"robot_id": self.robot_id})
        req = request.Request(
            f"{self.api_url}/session/v1/frame?{qs}", data=jpeg, method="POST"
        )
        req.add_header("Content-Type", "image/jpeg")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with request.urlopen(req, timeout=self.timeout):
                pass
        except error.HTTPError as e:
            # 409 = session closed between poll and post; normal, next tick idles.
            if e.code != 409:
                log.warning("session.frame.http_error status=%s", e.code)
        except (error.URLError, TimeoutError, OSError) as e:
            log.warning("session.frame.network_error %s", e)
