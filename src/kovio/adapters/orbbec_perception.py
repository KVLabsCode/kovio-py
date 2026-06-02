"""OrbbecPerceptionAdapter — real perception from an Astra-class depth camera.

Currently supports the OpenNI 2 backend, which covers the classic Astra
family: Astra, Astra Pro, Astra S, Astra Mini, Astra Pro Plus.

Newer cameras (Astra+, Astra 2, Astra Mini Pro, Femto, Gemini) need the
modern Orbbec SDK via `pyorbbecsdk`. That backend lands in v0.3.

Install requirements (Pi 5):
  - OpenNI 2 built and installed (see scripts/setup_astra_pi.sh)
  - pip install "kovio[astra]"   # pulls openni + numpy
  - OPENNI2_REDIST env var pointing at the OpenNI 2 redist directory
    (or pass openni2_path= to the constructor)

For person counting we currently do simple depth-mask thresholding: count
pixels in the "person depth range" and divide by typical-person area.
Replace with a real person detector (YOLOv8n on the Hailo HAT) in v0.3.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

from ..types import SceneState
from .perception import PerceptionAdapter

log = logging.getLogger("kovio.perception.orbbec")

# Rough pixel area of a standing adult at ~2m in a 640x480 depth frame.
# Used to convert mask area into a person count. Calibrate per deployment.
_PERSON_PIXEL_AREA = 4000


class OrbbecPerceptionAdapter(PerceptionAdapter):
    """Person counting via depth thresholding on an Orbbec Astra camera."""

    def __init__(
        self,
        min_depth_m: float = 0.6,
        max_depth_m: float = 5.0,
        rate_hz: float = 2.0,
        openni2_path: str | None = None,
    ) -> None:
        self.min_depth_mm = int(min_depth_m * 1000)
        self.max_depth_mm = int(max_depth_m * 1000)
        self.rate_hz = rate_hz
        self.openni2_path = openni2_path or os.environ.get("OPENNI2_REDIST") or ""

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._device = None
        self._stream = None

    def start(self, on_scene: Callable[[SceneState], None]) -> None:
        try:
            self._init_camera()
        except Exception:
            log.exception("Astra init failed — adapter will not emit scenes")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(on_scene,),
            daemon=True,
            name="kovio-astra",
        )
        self._thread.start()

    def _init_camera(self) -> None:
        # Local imports so the SDK itself doesn't require openni/numpy.
        from openni import openni2  # type: ignore[import-not-found]

        if self.openni2_path:
            log.info("initializing OpenNI 2 from %s", self.openni2_path)
            openni2.initialize(self.openni2_path)
        else:
            log.info("initializing OpenNI 2 (default search path)")
            openni2.initialize()

        self._device = openni2.Device.open_any()
        info = self._device.get_device_info()
        name = info.name.decode() if isinstance(info.name, bytes) else info.name
        log.info("opened Astra: %s", name)

        self._stream = self._device.create_depth_stream()
        self._stream.start()

    def _run(self, on_scene: Callable[[SceneState], None]) -> None:
        import numpy as np  # type: ignore[import-not-found]

        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self._stream.read_frame()
                buf = frame.get_buffer_as_uint16()
                img = np.frombuffer(buf, dtype=np.uint16).reshape(
                    frame.height, frame.width
                )
            except Exception:
                log.exception("depth frame read failed")
                self._stop.wait(period)
                continue

            # Build a person-range mask. Depth is in millimeters.
            mask = (img >= self.min_depth_mm) & (img <= self.max_depth_mm)
            valid_px = int(mask.sum())
            person_count = max(0, round(valid_px / _PERSON_PIXEL_AREA))

            if valid_px > 0:
                mean_depth_m = float(img[mask].mean()) / 1000.0
            else:
                mean_depth_m = None

            # Gaze isn't measurable from depth alone — needs RGB + a head-pose
            # model. v0.2 reports 0 attended; real gaze arrives in v0.3 with
            # the RGB+Hailo pipeline.
            scene = SceneState(
                person_count=person_count,
                attended_count=0,
                mean_distance_m=mean_depth_m,
            )

            try:
                on_scene(scene)
            except Exception:
                log.exception("on_scene callback raised")

            elapsed = time.time() - t0
            self._stop.wait(max(0.0, period - elapsed))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None
        try:
            from openni import openni2  # type: ignore[import-not-found]
            openni2.unload()
        except Exception:
            pass
        self._device = None
