"""RealSense perception — D455 depth+RGB camera with on-device person detection.

Target hardware: Intel RealSense D435i / D455 (USB 3.0).
Works on any host with pyrealsense2 installed (Pi 5, Jetson Orin, Linux x86, macOS).

The depth stream is used for per-person distance estimation. The RGB stream is
fed into a small YOLOv8n model (onnxruntime) for person detection. On Jetson,
we can later add a TensorRT-optimized variant; for now, onnxruntime gives us
cross-platform compatibility with acceptable performance (~10 FPS on Pi 5,
~30+ FPS on Jetson Orin Nano Super).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..types import SceneState
from .perception import PerceptionAdapter

log = logging.getLogger("kovio.perception.realsense")

YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
YOLO_MODEL_CACHE = Path.home() / ".cache" / "kovio" / "models" / "yolov8n.onnx"

# Class index 0 in COCO is "person"
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4


def _ensure_model() -> Path:
    """Download YOLOv8n ONNX model if not cached. Returns local path."""
    if YOLO_MODEL_CACHE.exists():
        return YOLO_MODEL_CACHE
    YOLO_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading YOLOv8n model to %s ...", YOLO_MODEL_CACHE)
    import urllib.request
    urllib.request.urlretrieve(YOLO_MODEL_URL, YOLO_MODEL_CACHE)
    return YOLO_MODEL_CACHE


class RealSensePerceptionAdapter(PerceptionAdapter):
    """D435i / D455 + YOLOv8n person detection.

    Args:
        cadence_seconds: how often to emit a SceneState (default 1.0s).
        width / height: capture resolution (default 640x480 — fast enough for any host).
        attention_threshold_m: people closer than this are considered "attended".
            Default 2.0m — adjust based on your robot's screen size.
    """

    def __init__(
        self,
        cadence_seconds: float = 1.0,
        width: int = 640,
        height: int = 480,
        attention_threshold_m: float = 2.0,
    ):
        self._cadence = cadence_seconds
        self._width = width
        self._height = height
        self._attention_threshold = attention_threshold_m
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_scene: Callable[[SceneState], None]) -> None:
        if self._thread is not None:
            log.warning("RealSensePerceptionAdapter already started")
            return

        try:
            import pyrealsense2 as rs
            import numpy as np
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit(
                "RealSensePerceptionAdapter requires pyrealsense2, numpy, and onnxruntime.\n"
                "Install with: pip install 'kovio[jetson]' (or [pi] if using on Pi).\n"
                f"Missing: {e}"
            )

        self._stop.clear()

        model_path = _ensure_model()
        session = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())
        input_name = session.get_inputs()[0].name

        # Configure RealSense pipeline — RGB and depth aligned
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, 30)
        align = rs.align(rs.stream.color)
        pipeline.start(config)
        log.info("RealSense pipeline started (%dx%d)", self._width, self._height)

        def _preprocess(bgr):
            import cv2
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (640, 640))
            tensor = resized.astype("float32") / 255.0
            tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
            tensor = tensor[None, :, :, :]      # batch
            return tensor

        def _detect_people(bgr, depth_frame) -> tuple[int, int, float | None]:
            """Return (person_count, attended_count, mean_distance_m)."""
            tensor = _preprocess(bgr)
            outputs = session.run(None, {input_name: tensor})
            preds = outputs[0][0]  # (84, num_predictions) — YOLOv8 output

            person_boxes = []
            for pred in preds.T:  # YOLOv8 output is transposed
                cls_scores = pred[4:]
                cls = int(cls_scores.argmax())
                conf = float(cls_scores.max())
                if cls == PERSON_CLASS_ID and conf >= CONF_THRESHOLD:
                    cx, cy = float(pred[0]), float(pred[1])
                    # Scale back to original image coords
                    px = int(cx / 640 * self._width)
                    py = int(cy / 640 * self._height)
                    person_boxes.append((px, py))

            person_count = len(person_boxes)
            distances = []
            attended = 0
            for (px, py) in person_boxes:
                if 0 <= px < self._width and 0 <= py < self._height:
                    d = depth_frame.get_distance(px, py)
                    if d > 0:
                        distances.append(d)
                        if d <= self._attention_threshold:
                            attended += 1
            mean_d = (sum(distances) / len(distances)) if distances else None
            return person_count, attended, mean_d

        def _run():
            log.info("RealSensePerceptionAdapter started (cadence=%.2fs)", self._cadence)
            last_emit = 0.0
            try:
                while not self._stop.is_set():
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    aligned = align.process(frames)
                    color = aligned.get_color_frame()
                    depth = aligned.get_depth_frame()
                    if not color or not depth:
                        continue

                    now = time.time()
                    if now - last_emit < self._cadence:
                        continue
                    last_emit = now

                    bgr = np.asanyarray(color.get_data())
                    try:
                        pc, ac, md = _detect_people(bgr, depth)
                    except Exception:
                        log.exception("Detection failed; emitting empty scene")
                        pc, ac, md = 0, 0, None

                    scene = SceneState(person_count=pc, attended_count=ac, mean_distance_m=md)
                    try:
                        on_scene(scene)
                    except Exception:
                        log.exception("on_scene callback raised")
            finally:
                pipeline.stop()
                log.info("RealSense pipeline stopped")

        self._thread = threading.Thread(target=_run, name="kovio-realsense-perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
