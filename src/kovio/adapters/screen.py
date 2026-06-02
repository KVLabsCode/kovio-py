"""Screen adapters — how the SDK displays creatives.

The default for a Pi 7" touchscreen is ChromiumKioskAdapter, which spawns
Chromium in fullscreen kiosk mode pointing at the creative URL. On a dev
machine without Chromium, fall back to ScreenAdapter.logger().
"""
from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod

from ..platform import find_chromium

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
