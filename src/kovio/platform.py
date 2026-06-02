"""Platform & hardware detection.

Used at SDK startup to pick sensible defaults for which adapters to load.
The user can always override via environment variables or explicit
KovioAgent construction.
"""
from __future__ import annotations

import logging
import os
import platform as _platform
import shutil
from enum import Enum
from pathlib import Path

log = logging.getLogger("kovio.platform")


class Platform(str, Enum):
    JETSON = "jetson"
    RASPBERRY_PI = "raspberry_pi"
    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    UNKNOWN = "unknown"

    @property
    def is_arm_sbc(self) -> bool:
        return self in (Platform.JETSON, Platform.RASPBERRY_PI)


def detect_platform() -> Platform:
    """Return the platform we're running on.

    Detection order: device-tree model file (most reliable for SBCs),
    then platform.system() + platform.machine() heuristics.
    """
    # ARM single-board computer detection via device-tree model file
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            model = model_path.read_text(errors="ignore").lower()
            if "jetson" in model or "nvidia" in model or "tegra" in model:
                return Platform.JETSON
            if "raspberry pi" in model:
                return Platform.RASPBERRY_PI
        except (OSError, UnicodeDecodeError) as e:
            log.debug("Could not read /proc/device-tree/model: %s", e)

    system = _platform.system().lower()
    if system == "darwin":
        return Platform.MACOS
    if system == "linux":
        return Platform.LINUX
    if system == "windows":
        return Platform.WINDOWS
    return Platform.UNKNOWN


def find_chromium() -> str | None:
    """Find a Chromium-family browser. Returns full path or None.

    Single source of truth for browser binary location — every adapter that
    needs to spawn a browser must call this rather than probing PATH itself.
    """
    candidates = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


def perception_provider_from_env() -> str | None:
    """If KOVIO_PERCEPTION is set, return its value; else None.

    Used to let the user override platform-based defaults without code changes.
    Recognized values: 'mock', 'orbbec', 'realsense'.
    """
    return os.environ.get("KOVIO_PERCEPTION", "").strip().lower() or None


def screen_adapter_from_env() -> str | None:
    """If KOVIO_SCREEN is set, return its value; else None.

    Recognized values: 'chromium', 'logger', 'none'.
    """
    return os.environ.get("KOVIO_SCREEN", "").strip().lower() or None


def default_perception(plat: Platform) -> str:
    """Default perception adapter for a given platform."""
    if plat == Platform.JETSON:
        return "realsense"
    if plat == Platform.RASPBERRY_PI:
        return "orbbec"
    return "mock"  # MacOS, generic Linux, Windows — no camera assumed


def default_screen(plat: Platform) -> str:
    """Default screen adapter for a given platform."""
    if plat.is_arm_sbc and find_chromium():
        return "chromium"
    return "logger"  # log only — for dev work on machines without a screen plugged in
