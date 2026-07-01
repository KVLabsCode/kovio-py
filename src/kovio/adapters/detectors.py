"""On-device detectors: people, phones, pose, and a gaze proxy.

These wrap the ML models that turn a camera frame into the geometry the tracker
and gesture classifier reason over. The heavy libraries (onnxruntime, opencv)
are imported lazily so the SDK core stays dependency-free and importable on any
host; construct a detector only on a robot with the ``[jetson]`` extra.

Two pieces are intentionally pure (numpy-only) and unit-tested, because they are
where bugs hide: ``parse_yolo_output`` (decode + confidence filter) and
``associate_phones`` (which person is holding the phone). The model I/O around
them is thin.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("kovio.perception.detectors")

PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67  # COCO
NOSE, L_EYE, R_EYE = 0, 1, 2  # COCO-17 head keypoints used by facing_camera

_MODELS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
    "yolov8n-pose": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.onnx",
}
_CACHE = Path.home() / ".cache" / "kovio" / "models"


def ensure_model(name: str) -> Path:
    """Return a local path to ``name``.onnx, downloading once if needed."""
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}; known: {list(_MODELS)}")
    dst = _CACHE / f"{name}.onnx"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    log.info("downloading %s -> %s", name, dst)
    urllib.request.urlretrieve(_MODELS[name], dst)
    return dst


@dataclass
class Box:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class PersonObservation:
    """Everything we derived about one person in one frame."""

    box: Box
    distance_m: float | None = None
    looking: bool = False
    has_phone: bool = False
    keypoints: list | None = None  # COCO-17 [(x,y,conf), ...] when pose ran
    forward_left: bool = False
    forward_right: bool = False


# --------------------------------------------------------------------------- #
# Pure helpers (numpy only) — unit tested.
# --------------------------------------------------------------------------- #

def parse_yolo_output(preds, conf_threshold, img_w, img_h, input_size=640):
    """Decode a YOLOv8 detection head into pixel-space ``Box`` list.

    ``preds`` is the model's ``(4 + num_classes, num_anchors)`` array (already
    squeezed of the batch dim). Boxes are letterbox-free (assumes the frame was
    resized straight to ``input_size`` x ``input_size``), scaled back to the
    original image. No NMS here — callers apply :func:`nms` if they need it.
    """
    import numpy as np

    preds = np.asarray(preds)
    if preds.ndim != 2:
        raise ValueError(f"expected 2D preds, got shape {preds.shape}")
    p = preds.T  # (num_anchors, 4 + num_classes)
    cls_scores = p[:, 4:]
    cls = cls_scores.argmax(axis=1)
    conf = cls_scores.max(axis=1)
    keep = conf >= conf_threshold

    sx = img_w / input_size
    sy = img_h / input_size
    boxes: list[Box] = []
    for cx, cy, w, h, c, k in zip(
        p[keep, 0], p[keep, 1], p[keep, 2], p[keep, 3], conf[keep], cls[keep]
    ):
        boxes.append(
            Box(
                cls=int(k),
                conf=float(c),
                x1=float((cx - w / 2) * sx),
                y1=float((cy - h / 2) * sy),
                x2=float((cx + w / 2) * sx),
                y2=float((cy + h / 2) * sy),
            )
        )
    return boxes


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[Box], iou_threshold: float = 0.5) -> list[Box]:
    """Greedy per-class non-max suppression."""
    out: list[Box] = []
    for b in sorted(boxes, key=lambda x: x.conf, reverse=True):
        if all(iou(b, k) <= iou_threshold for k in out if k.cls == b.cls):
            out.append(b)
    return out


def facing_camera(keypoints, conf_min: float = 0.3) -> bool:
    """Keypoint-based 'looking at the screen' proxy — pure, no extra model.

    When a head is turned toward the camera both eyes are visible and the nose
    sits between them; in profile one eye drops out (low keypoint confidence) and
    the nose slides past an eye. We require confident nose + both eyes and the
    nose horizontally between the eyes. Cheaper and steadier than a face cascade
    when pose keypoints are already on hand. It is a proxy for attention, not
    identity or true gaze vector.
    """
    if keypoints is None or len(keypoints) <= R_EYE:
        return False
    nose = keypoints[NOSE]
    le, re = keypoints[L_EYE], keypoints[R_EYE]
    if nose is None or le is None or re is None:
        return False
    if min(nose[2], le[2], re[2]) < conf_min:
        return False
    lo, hi = min(le[0], re[0]), max(le[0], re[0])
    if hi - lo < 1e-3:
        return False
    return lo <= nose[0] <= hi


def _contains_center(person: Box, phone: Box) -> bool:
    return person.x1 <= phone.cx <= person.x2 and person.y1 <= phone.cy <= person.y2


def associate_phones(persons: list[Box], phones: list[Box]) -> tuple[int, set[int]]:
    """Map detected phones to the people holding them.

    A phone belongs to the person whose box contains the phone's centre; if none
    contains it, the nearest person centre within a phone-diagonal claims it.
    Returns ``(phones_out, {indices of persons holding a phone})``. Counting the
    *people* (not raw phone boxes) is what the "phone-out" metric wants.
    """
    holders: set[int] = set()
    counted = 0
    for ph in phones:
        owner = None
        for i, pe in enumerate(persons):
            if _contains_center(pe, ph):
                owner = i
                break
        if owner is None and persons:
            diag = ((ph.x2 - ph.x1) ** 2 + (ph.y2 - ph.y1) ** 2) ** 0.5
            best_i, best_d = None, float("inf")
            for i, pe in enumerate(persons):
                d = ((pe.cx - ph.cx) ** 2 + (pe.cy - ph.cy) ** 2) ** 0.5
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None and best_d <= max(diag * 3.0, 1.0):
                owner = best_i
        if owner is not None:
            counted += 1
            holders.add(owner)
    return counted, holders


# --------------------------------------------------------------------------- #
# Model wrappers (lazy heavy deps) — thin glue around the pure helpers.
# --------------------------------------------------------------------------- #

class YoloDetector:
    """YOLOv8 detection wrapper restricted to a set of COCO classes."""

    def __init__(self, model: str = "yolov8n", conf: float = 0.5, classes=None):
        import onnxruntime as ort  # noqa: lazy

        self.conf = conf
        self.classes = set(classes) if classes is not None else None
        self._session = ort.InferenceSession(
            str(ensure_model(model)), providers=ort.get_available_providers()
        )
        self._input = self._session.get_inputs()[0].name

    def detect(self, bgr) -> list[Box]:
        import cv2
        import numpy as np

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640)).astype("float32") / 255.0
        tensor = resized.transpose(2, 0, 1)[None]
        out = self._session.run(None, {self._input: tensor})[0][0]
        boxes = parse_yolo_output(np.asarray(out), self.conf, w, h)
        if self.classes is not None:
            boxes = [b for b in boxes if b.cls in self.classes]
        return nms(boxes)


class GazeEstimator:
    """Cheap 'is this person looking at the screen' proxy.

    A frontal face visible inside the upper region of the person box is a strong
    signal the head is turned toward the camera (hence the screen). This is a
    proxy, not true gaze estimation — documented as such — but it is fast and
    runs on the existing RGB frame with no extra model. Falls back to "unknown"
    (``False``) rather than guessing when OpenCV's cascade data is unavailable.
    """

    def __init__(self) -> None:
        self._cascade = None
        try:
            import cv2

            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self._cascade = cascade
            else:
                log.warning("frontal-face cascade missing; gaze disabled")
        except Exception:
            log.warning("opencv unavailable; gaze disabled")

    @property
    def available(self) -> bool:
        return self._cascade is not None

    def looking(self, bgr, person: Box) -> bool:
        if self._cascade is None:
            return False
        import cv2

        x1, y1 = max(0, int(person.x1)), max(0, int(person.y1))
        x2 = min(bgr.shape[1], int(person.x2))
        # Upper 45% of the body box is where the head is.
        y2 = min(bgr.shape[0], int(person.y1 + 0.45 * (person.y2 - person.y1)))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return False
        crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(crop, 1.2, 4, minSize=(24, 24))
        return len(faces) > 0


class PoseDetector:
    """YOLOv8-pose wrapper -> per-person COCO-17 keypoints."""

    def __init__(self, model: str = "yolov8n-pose", conf: float = 0.5):
        import onnxruntime as ort  # noqa: lazy

        self.conf = conf
        self._session = ort.InferenceSession(
            str(ensure_model(model)), providers=ort.get_available_providers()
        )
        self._input = self._session.get_inputs()[0].name

    def detect(self, bgr) -> list[tuple]:
        """Return [(Box, keypoints[(x,y,conf)*17]), ...]."""
        import cv2
        import numpy as np

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640)).astype("float32") / 255.0
        tensor = resized.transpose(2, 0, 1)[None]
        out = np.asarray(self._session.run(None, {self._input: tensor})[0][0])  # (56, N)
        p = out.T  # (N, 56): 4 box + 1 person-conf + 51 kpts
        sx, sy = w / 640.0, h / 640.0
        results = []
        for row in p:
            conf = float(row[4])
            if conf < self.conf:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            box = Box(
                PERSON_CLASS_ID, conf,
                float((cx - bw / 2) * sx), float((cy - bh / 2) * sy),
                float((cx + bw / 2) * sx), float((cy + bh / 2) * sy),
            )
            kpts = []
            for k in range(17):
                kx, ky, kc = row[5 + 3 * k], row[6 + 3 * k], row[7 + 3 * k]
                kpts.append((float(kx * sx), float(ky * sy), float(kc)))
            results.append((box, kpts))
        # NMS on the person boxes, carrying keypoints along.
        kept = nms([b for b, _ in results])
        keep_ids = {id(b) for b in kept}
        return [(b, kp) for b, kp in results if id(b) in keep_ids]
