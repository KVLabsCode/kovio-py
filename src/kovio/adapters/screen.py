"""Screen adapters — how the SDK displays creatives.

The default for a Pi 7" touchscreen is ChromiumKioskAdapter, which spawns
Chromium in fullscreen kiosk mode pointing at the creative URL. On a dev
machine without Chromium, fall back to ScreenAdapter.logger(). For laptop
demos there's BrowserScreenAdapter, which serves the robot screen as a web
page you open yourself (no kiosk process to manage).
"""
from __future__ import annotations

import json
import logging
import socketserver
import sqlite3
import subprocess
import threading
from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..platform import find_chromium
from ..types import AdEvent

log = logging.getLogger("kovio.screen")


class ScreenAdapter(ABC):
    """Display a creative on the robot's screen."""

    @abstractmethod
    def display(self, url_or_path: str) -> None:
        """Display the given creative (URL or local file path)."""

    @abstractmethod
    def clear(self) -> None:
        """Clear the screen."""

    # --- factories ---

    @staticmethod
    def pi_touchscreen() -> "ScreenAdapter":
        """Default adapter for the official Pi 7\" touchscreen + Chromium."""
        return ChromiumKioskAdapter()

    @staticmethod
    def logger() -> "ScreenAdapter":
        """No-op adapter that just logs. Useful on dev machines."""
        return LoggingScreenAdapter()

    @staticmethod
    def browser(
        db_path: str | Path = "kovio.db",
        robot_id: str = "",
        port: int = 8001,
    ) -> "ScreenAdapter":
        """Serve the robot screen as a web page on http://localhost:<port>.

        Browser-based stand-in for the Pi kiosk: instead of spawning Chromium,
        it renders the live display state in a page you open yourself, and
        records taps as engagement events. Used by `kovio demo`.
        """
        return BrowserScreenAdapter(db_path=db_path, robot_id=robot_id, port=port)


class ChromiumKioskAdapter(ScreenAdapter):
    """Display HTML creatives via Chromium in kiosk mode.

    Browser binary location comes from platform.find_chromium() — the single
    source of truth across the codebase. Construction fails fast with a clear
    error if no Chromium-family browser is on PATH.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._chromium = find_chromium()
        if not self._chromium:
            raise SystemExit(
                "No Chromium-family browser found on PATH.\n"
                "Pi:  sudo apt install chromium-browser\n"
                "Jetson / Ubuntu: sudo apt install chromium\n"
                "macOS: install Google Chrome from https://chrome.google.com/\n"
                "Or run with KOVIO_SCREEN=logger to log creative URLs instead."
            )

    def display(self, url_or_path: str) -> None:
        self.clear()
        self._proc = subprocess.Popen(
            [
                self._chromium,
                "--kiosk",
                "--noerrdialogs",
                "--disable-infobars",
                "--no-first-run",
                url_or_path,
            ]
        )
        log.info("[screen] displaying %s (pid=%s)", url_or_path, self._proc.pid)

    def clear(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


class LoggingScreenAdapter(ScreenAdapter):
    """No-op screen. Just logs. Use on dev machines without a display."""

    def display(self, url_or_path: str) -> None:
        log.info("[screen] DISPLAY %s", url_or_path)

    def clear(self) -> None:
        log.info("[screen] CLEAR")


class BrowserScreenAdapter(ScreenAdapter):
    """Serve the current creative as a web page on localhost.

    A tiny stdlib HTTP server (no extra deps) renders the robot's screen in
    your own browser: a breathing 'kovio' wordmark while idle, and the live
    creative — with a save QR and an engage hint — when one is up. `display()`
    and `clear()` just flip the state the page polls. Taps over a live creative
    POST to /api/engage, which appends an `engagement` event to the same SQLite
    log the agent writes (so the cloud sink uploads it like any other event).
    """

    def __init__(
        self,
        db_path: str | Path = "kovio.db",
        robot_id: str = "",
        port: int = 8001,
        host: str = "127.0.0.1",
        save_url: str | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._robot_id = robot_id
        self._port = port
        self._host = host
        self._save_url = save_url or f"https://app.kovio.dev/save?r={robot_id or 'demo'}"

        self._lock = threading.Lock()
        self._showing = False
        self._creative_url: str | None = None
        self._qr_svg = _render_qr_svg(self._save_url)

        self._httpd: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None
        self._start_server()

    # --- ScreenAdapter interface ---

    def display(self, url_or_path: str) -> None:
        with self._lock:
            self._showing = True
            self._creative_url = url_or_path
        log.info("[screen] serving creative %s at http://%s:%d", url_or_path, self._host, self._port)

    def clear(self) -> None:
        with self._lock:
            self._showing = False
        log.info("[screen] idle (no creative)")

    # --- state the HTTP handler reads ---

    def _state_json(self) -> dict:
        with self._lock:
            showing = self._showing
            creative = self._creative_url
        # Local file:// creatives are proxied through /creative so the iframe
        # can load them same-origin; http(s) creatives are framed directly.
        if showing and creative:
            src = "/creative" if creative.startswith("file:") else creative
        else:
            src = None
        return {
            "showing": showing,
            "creative_src": src,
            "robot_id": self._robot_id,
            "qr_svg": self._qr_svg,
        }

    def _current_json(self) -> dict:
        """What's on screen right now, enriched with the live campaign (if any).

        `state` is "playing" / "idle"; when playing, `campaign_id` / `advertiser`
        come from the most recent `ad_played` event. They stay null for the
        bundled default creative (its play event carries no campaign), which is
        exactly how you tell a real cloud campaign from the local fallback.
        """
        with self._lock:
            showing = self._showing
            creative = self._creative_url
        campaign_id = advertiser = None
        if showing:
            try:
                conn = sqlite3.connect(str(self._db_path))
                row = conn.execute(
                    "SELECT payload_json FROM events WHERE event_type='ad_played' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                conn.close()
                if row:
                    p = json.loads(row[0])
                    campaign_id = p.get("campaign_id")
                    advertiser = p.get("advertiser")
            except sqlite3.Error:
                pass
        return {
            "state": "playing" if showing else "idle",
            "creative_url": creative if showing else None,
            "campaign_id": campaign_id,
            "advertiser": advertiser,
        }

    def _creative_bytes(self) -> bytes | None:
        """Read the current local creative's HTML for the /creative proxy."""
        with self._lock:
            creative = self._creative_url
        if not creative or not creative.startswith("file:"):
            return None
        path = Path(unquote(urlparse(creative).path))
        try:
            return path.read_bytes()
        except OSError as e:
            log.warning("[screen] could not read creative %s: %s", path, e)
            return None

    def _record_engagement(self) -> None:
        """Append an engagement event to the local log (cloud-syncs like the rest)."""
        with self._lock:
            creative = self._creative_url
        evt = AdEvent(
            event_type="engagement",
            payload={"creative_url": creative, "source": "browser_tap"},
            robot_id=self._robot_id,
        )
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            # Mirror _EventDB's schema — tolerate the table not existing yet.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id     TEXT PRIMARY KEY,
                    timestamp    REAL NOT NULL,
                    event_type   TEXT NOT NULL,
                    robot_id     TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO events (event_id, timestamp, event_type, robot_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (evt.event_id, evt.timestamp, evt.event_type, evt.robot_id, json.dumps(evt.payload)),
            )
            conn.commit()
            conn.close()
            log.info("[screen] engagement recorded (creative=%s)", creative)
        except sqlite3.Error as e:
            log.warning("[screen] failed to record engagement: %s", e)

    # --- server lifecycle ---

    def _start_server(self) -> None:
        adapter = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
                route = urlparse(self.path).path
                if route == "/":
                    from .browser_screen_page import PAGE_HTML
                    self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
                elif route == "/api/state":
                    self._send(200, json.dumps(adapter._state_json()).encode(), "application/json")
                elif route == "/api/current":
                    self._send(200, json.dumps(adapter._current_json()).encode(), "application/json")
                elif route == "/creative":
                    body = adapter._creative_bytes()
                    if body is None:
                        self._send(404, b"no creative", "text/plain")
                    else:
                        self._send(200, body, "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
                if urlparse(self.path).path == "/api/engage":
                    adapter._record_engagement()
                    self._send(200, b'{"ok":true}', "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def log_message(self, fmt: str, *args) -> None:  # quiet the default stderr spam
                log.debug("[screen.http] " + fmt, *args)

        try:
            self._httpd = _ThreadingHTTPServer((self._host, self._port), Handler)
        except OSError as e:
            raise SystemExit(
                f"Could not bind the demo screen to {self._host}:{self._port} ({e}).\n"
                f"Is another `kovio demo` already running? Free the port or pass a different one."
            )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="kovio-browser-screen", daemon=True
        )
        self._thread.start()
        log.info("[screen] browser screen live → http://%s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _render_qr_svg(data: str) -> str:
    """Return a self-contained SVG QR for `data`, or a styled placeholder.

    Uses the optional `qrcode` library (pulled in by the `dev` extra). If it
    isn't installed we degrade to a labelled placeholder rather than failing —
    the demo still runs, the QR just isn't scannable.
    """
    try:
        import qrcode
        import qrcode.image.svg as svg  # pure-Python SVG factory (no Pillow needed)
    except ImportError:
        log.info("[screen] qrcode not installed; showing a QR placeholder "
                 "(pip install 'kovio[dev]' for a scannable code)")
        return (
            '<svg viewBox="0 0 124 124" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="124" height="124" rx="8" fill="#f2ecdc"/>'
            '<text x="62" y="58" font-size="10" fill="#8a7f6f" text-anchor="middle" '
            'font-family="monospace">scan to</text>'
            '<text x="62" y="72" font-size="10" fill="#8a7f6f" text-anchor="middle" '
            'font-family="monospace">save</text></svg>'
        )
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(image_factory=svg.SvgPathImage)
    return img.to_string(encoding="unicode")


def make_screen_adapter(name: str | None = None) -> "ScreenAdapter":
    """Construct the screen adapter by name.

    If name is None, falls back to KOVIO_SCREEN env var, then to the platform
    default. Recognized names: 'chromium', 'logger', 'none' (alias for logger).
    """
    from ..platform import (
        detect_platform,
        screen_adapter_from_env,
        default_screen,
    )

    if name is None:
        name = screen_adapter_from_env() or default_screen(detect_platform())

    if name == "chromium":
        return ChromiumKioskAdapter()
    if name in ("logger", "none"):
        return LoggingScreenAdapter()
    raise ValueError(
        f"Unknown screen adapter: {name!r}. Valid: chromium, logger, none."
    )
