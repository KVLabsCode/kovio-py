"""Mock perception — synthesizes SceneState events on a schedule.

Used for development on machines without a camera, and for end-to-end
testing of the full pipeline (selector -> screen -> event log -> cloud sync)
without needing physical hardware.

Unlike StubPerceptionAdapter (random scenes), this adapter loops a fixed,
scripted sequence so demos are deterministic and reproducible.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from itertools import cycle

from ..types import SceneState
from .perception import PerceptionAdapter

log = logging.getLogger("kovio.perception.mock")

DEFAULT_DEMO_SCRIPT = [
    # Each entry: SceneState kwargs (timestamp set by adapter)
    {"person_count": 0, "attended_count": 0, "mean_distance_m": None},
    {"person_count": 0, "attended_count": 0, "mean_distance_m": None},
    {"person_count": 1, "attended_count": 0, "mean_distance_m": 3.5},
    {"person_count": 1, "attended_count": 1, "mean_distance_m": 2.1},
    {"person_count": 2, "attended_count": 1, "mean_distance_m": 1.8},
    {"person_count": 2, "attended_count": 2, "mean_distance_m": 1.5},
    {"person_count": 1, "attended_count": 1, "mean_distance_m": 1.4},
    {"person_count": 1, "attended_count": 0, "mean_distance_m": 2.8},
    {"person_count": 0, "attended_count": 0, "mean_distance_m": None},
    {"person_count": 0, "attended_count": 0, "mean_distance_m": None},
    {"person_count": 3, "attended_count": 2, "mean_distance_m": 2.5},
    {"person_count": 0, "attended_count": 0, "mean_distance_m": None},
]


class MockPerceptionAdapter(PerceptionAdapter):
    """Emits scripted SceneState events at a fixed cadence."""

    def __init__(self, script: list[dict] | None = None, cadence_seconds: float = 2.0):
        self._script = script or DEFAULT_DEMO_SCRIPT
        self._cadence = cadence_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_scene: Callable[[SceneState], None]) -> None:
        if self._thread is not None:
            log.warning("MockPerceptionAdapter already started")
            return

        self._stop.clear()

        def _run():
            log.info("MockPerceptionAdapter started (cadence=%.2fs)", self._cadence)
            for kwargs in cycle(self._script):
                if self._stop.is_set():
                    break
                scene = SceneState(**kwargs)
                try:
                    on_scene(scene)
                except Exception:
                    log.exception("on_scene callback raised")
                self._stop.wait(self._cadence)
            log.info("MockPerceptionAdapter stopped")

        self._thread = threading.Thread(target=_run, name="kovio-mock-perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
