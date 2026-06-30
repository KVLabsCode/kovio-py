"""Perception adapters — how the SDK observes the audience.

v0 ships a stub adapter that emits synthetic SceneState events. Real
adapters wrapping a 3D depth camera + Hailo person detection land in v0.2.
"""
from __future__ import annotations

import logging
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..types import SceneState

log = logging.getLogger("kovio.perception")


class PerceptionAdapter(ABC):
    """Source of SceneState events."""

    @abstractmethod
    def start(self, on_scene: Callable[[SceneState], None]) -> None:
        """Begin emitting scene states. Calls on_scene for each."""

    @abstractmethod
    def stop(self) -> None:
        """Stop emitting."""

    # --- factories ---

    @staticmethod
    def stub(rate_hz: float = 1.0) -> "PerceptionAdapter":
        """Synthetic perception — random scenes at rate_hz. No camera needed."""
        return StubPerceptionAdapter(rate_hz)


def make_perception_adapter(name: str | None = None) -> "PerceptionAdapter":
    """Construct the perception adapter by name.

    If name is None, falls back to KOVIO_PERCEPTION env var, then to
    the platform default. Recognized names: 'mock', 'orbbec', 'realsense',
    'rich' (depth-camera + lidar fusion with interaction metrics).
    """
    from ..platform import (
        detect_platform,
        perception_provider_from_env,
        default_perception,
    )

    if name is None:
        name = perception_provider_from_env() or default_perception(detect_platform())

    if name == "mock":
        from .mock_perception import MockPerceptionAdapter
        return MockPerceptionAdapter()
    if name == "orbbec":
        from .orbbec_perception import OrbbecPerceptionAdapter
        return OrbbecPerceptionAdapter()
    if name == "realsense":
        from .realsense_perception import RealSensePerceptionAdapter
        return RealSensePerceptionAdapter()
    if name in ("rich", "realsense_rich", "fusion"):
        from .rich_perception import RichPerceptionAdapter
        return RichPerceptionAdapter()
    raise ValueError(
        f"Unknown perception adapter: {name!r}. "
        f"Valid: mock, orbbec, realsense, rich."
    )


class StubPerceptionAdapter(PerceptionAdapter):
    """Emit random scene states at a fixed rate. No hardware required."""

    def __init__(self, rate_hz: float = 1.0) -> None:
        self.rate_hz = rate_hz
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, on_scene: Callable[[SceneState], None]) -> None:
        log.info("[perception] starting stub at %.2f Hz", self.rate_hz)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_scene,), daemon=True, name="kovio-stub"
        )
        self._thread.start()

    def _run(self, on_scene: Callable[[SceneState], None]) -> None:
        while not self._stop.is_set():
            n = random.randint(0, 5)
            attended = random.randint(0, n) if n else 0
            scene = SceneState(
                person_count=n,
                attended_count=attended,
                mean_distance_m=round(random.uniform(0.5, 4.0), 2) if n else None,
            )
            try:
                on_scene(scene)
            except Exception:
                log.exception("on_scene callback raised")
            self._stop.wait(1.0 / self.rate_hz)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
