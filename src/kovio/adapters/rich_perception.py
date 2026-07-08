"""Rich fused perception — depth camera + lidar -> enriched SceneState.

This is the adapter the always-on robot service runs. It fuses everything the
sensors can derive on-device, with no frame ever leaving the robot:

  * people & distance & attention  (RealSense RGB-D + YOLO/pose)
  * who is looking, and dwell       (tracker + keypoint gaze proxy)
  * phone-out                       (YOLO 'cell phone' -> nearest person)
  * physical interactions           (pose -> GestureClassifier, depth gives the
                                      forward hint for handshake/fist-bump)
  * crowd / nearest / approach       (Livox lidar, wide field of view)

Every capability is independently toggleable and degrades gracefully: missing
pose model -> no gestures; no lidar -> no crowd fields; etc. The originals
(person_count, attended_count, mean_distance_m) are always filled so this is a
drop-in replacement for RealSensePerceptionAdapter.

Heavy deps (pyrealsense2, onnxruntime, opencv) are imported lazily inside
start() so importing this module never drags them in.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from ..types import Interaction, InteractionKind, SceneState
from .gestures import GestureClassifier, L_WRIST, R_WRIST
from .perception import PerceptionAdapter
from .tracker import CentroidTracker, Detection

log = logging.getLogger("kovio.perception.rich")

PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67
_FORWARD_MARGIN_M = 0.15  # wrist this much closer than the body = reaching out


class RichPerceptionAdapter(PerceptionAdapter):
    """Depth-camera + lidar fusion emitting enriched SceneState + interactions.

    Args:
        cadence_seconds: how often to emit a SceneState.
        attention_threshold_m: people closer than this count as "attended".
        enable_phone / enable_gestures / enable_gaze / enable_lidar: feature gates.
        gaze_dwell_seconds: sustained-gaze threshold for a ``gaze_dwell`` event.
        interaction_cooldown_s: min gap between repeat interactions of the same
            kind from the same tracked person (debounces per-frame gestures).
        network_interface: NIC the Go2 lidar DDS is on (eth0 on the robot).
    """

    def __init__(
        self,
        cadence_seconds: float = 1.0,
        width: int = 640,
        height: int = 480,
        attention_threshold_m: float = 2.0,
        enable_phone: bool = True,
        enable_gestures: bool = True,
        enable_gaze: bool = True,
        enable_lidar: bool = True,
        gaze_dwell_seconds: float = 1.5,
        interaction_cooldown_s: float = 2.0,
        lidar_radius_m: float = 4.0,
        network_interface: str = "eth0",
    ) -> None:
        self._cadence = cadence_seconds
        self._width = width
        self._height = height
        self._attn = attention_threshold_m
        self._enable_phone = enable_phone
        self._enable_gestures = enable_gestures
        self._enable_gaze = enable_gaze
        self._enable_lidar = enable_lidar
        self._cooldown = interaction_cooldown_s
        self._lidar_radius = lidar_radius_m
        self._iface = network_interface

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Latest color frame, for the admin session live view. Guarded copy —
        # the capture loop overwrites it every tick.
        self._frame_lock = threading.Lock()
        self._latest_bgr = None
        # min_hits=2: a detection must persist across two ticks to count as a
        # person, filtering single-tick detector flicker on background clutter.
        self._tracker = CentroidTracker(
            gaze_dwell_seconds=gaze_dwell_seconds, min_hits=2
        )
        self._gestures = GestureClassifier()
        self._last_fired: dict[tuple[int, str], float] = {}
        self._phone_state: dict[int, bool] = {}

    # ------------------------------------------------------------------ #

    def start(self, on_scene: Callable[[SceneState], None]) -> None:
        if self._thread is not None:
            log.warning("RichPerceptionAdapter already started")
            return
        try:
            import numpy as np  # noqa: F401
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit(
                "RichPerceptionAdapter needs pyrealsense2 + onnxruntime + opencv + numpy.\n"
                "Install with: pip install 'kovio[jetson]'\n"
                f"Missing: {e}"
            )
        from . import detectors as det

        # Detection stack. Pose gives persons+keypoints (gestures, gaze); a
        # detection model adds phones. We avoid running YOLO twice for persons.
        pose = None
        person_det = None
        phone_det = None
        if self._enable_gestures:
            try:
                pose = det.PoseDetector()
            except Exception:
                log.exception("pose model unavailable; gestures+pose-gaze disabled")
        if pose is None:
            person_det = det.YoloDetector(classes={PERSON_CLASS_ID})
        if self._enable_phone:
            phone_det = det.YoloDetector(classes={CELL_PHONE_CLASS_ID})
        gaze_cascade = det.GazeEstimator() if (self._enable_gaze and pose is None) else None

        lidar = None
        if self._enable_lidar:
            try:
                lidar = det_lidar = self._make_lidar()
                if not det_lidar.available:
                    log.info("lidar present-but-unavailable; crowd metrics omitted")
            except Exception:
                log.exception("lidar init failed; crowd metrics omitted")

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(on_scene, rs, det, pose, person_det, phone_det, gaze_cascade, lidar),
            name="kovio-rich-perception",
            daemon=True,
        )
        self._thread.start()

    def _make_lidar(self):
        from .lidar import LidarSource

        return LidarSource(radius_m=self._lidar_radius, network_interface=self._iface)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def latest_frame_bgr(self):
        """Most recent color frame (BGR ndarray copy), or None before first
        capture. Feeds the admin session live view; never blocks the loop."""
        with self._frame_lock:
            return self._latest_bgr

    # ------------------------------------------------------------------ #

    def _run(self, on_scene, rs, det, pose, person_det, phone_det, gaze_cascade, lidar):
        import numpy as np

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, 30)
        align = rs.align(rs.stream.color)
        pipeline.start(config)
        log.info("rich perception started (cadence=%.2fs)", self._cadence)

        last_emit = 0.0
        misses = 0
        try:
            while not self._stop.is_set():
                # A transient frame timeout must NOT kill perception. Count
                # consecutive misses; after a run of them, rebuild the pipeline
                # (recovers a hung/dropped RealSense) instead of dying.
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    misses = 0
                except RuntimeError as e:
                    misses += 1
                    log.warning("frame wait failed (%d in a row): %s", misses, e)
                    if misses >= 5:
                        log.error("no frames after %d tries; restarting pipeline", misses)
                        try:
                            pipeline.stop()
                        except Exception:
                            pass
                        try:
                            pipeline.start(config)
                            misses = 0
                        except Exception:
                            log.exception("pipeline restart failed; retrying")
                            time.sleep(2.0)
                    continue
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
                with self._frame_lock:
                    self._latest_bgr = bgr.copy()
                try:
                    scene = self._build_scene(
                        bgr, depth, now, det, pose, person_det,
                        phone_det, gaze_cascade, lidar,
                    )
                except Exception:
                    log.exception("perception failed; emitting empty scene")
                    scene = SceneState(0, 0, None)
                try:
                    on_scene(scene)
                except Exception:
                    log.exception("on_scene callback raised")
        finally:
            pipeline.stop()
            log.info("rich perception stopped")

    def _depth_at(self, depth, px, py) -> float | None:
        if 0 <= px < self._width and 0 <= py < self._height:
            d = depth.get_distance(int(px), int(py))
            return d if d > 0 else None
        return None

    def _build_scene(self, bgr, depth, now, det, pose, person_det,
                     phone_det, gaze_cascade, lidar) -> SceneState:
        # 1) People (+ keypoints when pose is active).
        people = []  # list of (box, keypoints|None)
        if pose is not None:
            people = pose.detect(bgr)
        else:
            people = [(b, None) for b in person_det.detect(bgr)]
        person_boxes = [b for b, _ in people]

        # 2) Phones -> which person holds one.
        holders: set[int] = set()
        phones_out = 0
        if phone_det is not None and person_boxes:
            phone_boxes = phone_det.detect(bgr)
            phones_out, holders = det.associate_phones(person_boxes, phone_boxes)

        # 3) Per-person geometry: distance, gaze, forward-reach hints.
        detections: list[Detection] = []
        distances: list[float] = []
        attended = 0
        for idx, (box, kp) in enumerate(people):
            dist = self._depth_at(depth, box.cx, box.cy)
            if dist is not None:
                distances.append(dist)
                if dist <= self._attn:
                    attended += 1
            looking = False
            if self._enable_gaze:
                if kp is not None:
                    looking = det.facing_camera(kp)
                elif gaze_cascade is not None:
                    looking = gaze_cascade.looking(bgr, box)
            detections.append(
                Detection(cx=box.cx, cy=box.cy, distance_m=dist, looking=looking)
            )

        # 4) Track -> stable ids, dwell, sustained-gaze events. Counts come from
        # CONFIRMED tracks (seen >= min_hits frames), so a chair/lamp that only
        # trips the detector now and then never becomes a "person".
        tracks = self._tracker.update(detections, now)
        confirmed = self._tracker.confirmed_tracks()
        confirmed_ids = {t.track_id for t in confirmed}
        mean_dwell = self._tracker.mean_dwell_seconds()
        looked_count = sum(1 for t in confirmed if t.looking)

        # Map each detection index to its track id by nearest centroid.
        track_for_idx = self._match_tracks_to_people(person_boxes, tracks)

        interactions: list[Interaction] = []

        # 5) Gestures (pose only).
        if pose is not None and self._enable_gestures:
            for idx, (box, kp) in enumerate(people):
                tid = track_for_idx.get(idx)
                if tid is None or kp is None or tid not in confirmed_ids:
                    continue
                fl, fr = self._forward_hints(depth, kp, self._depth_at(depth, box.cx, box.cy))
                for hit in self._gestures.classify(tid, kp, now, fl, fr):
                    if self._allow(tid, hit.kind, now):
                        interactions.append(
                            Interaction(hit.kind, hit.confidence, tid,
                                        self._depth_at(depth, box.cx, box.cy))
                        )

        # 6) Phone-out interaction on the rising edge per track.
        if phone_det is not None:
            live_ids = {t.track_id for t in tracks}
            holder_ids = {track_for_idx.get(i) for i in holders}
            holder_ids.discard(None)
            holder_ids &= confirmed_ids  # ignore phantom "phone holders"
            for tid in holder_ids:
                if not self._phone_state.get(tid, False) and self._allow(tid, InteractionKind.PHONE_OUT, now):
                    interactions.append(Interaction(InteractionKind.PHONE_OUT, 0.9, tid))
            self._phone_state = {tid: (tid in holder_ids) for tid in live_ids}

        # 7) Sustained-gaze events (confirmed people only).
        for t in self._tracker.new_gaze_dwell_tracks():
            if t.track_id in confirmed_ids:
                interactions.append(
                    Interaction(InteractionKind.GAZE_DWELL, 1.0, t.track_id, t.distance_m)
                )

        # 8) Lidar crowd context (+ per-person blips and unique-entry "passed").
        people_nearby = crowd_density = nearest = bearing = None
        lidar_people = lidar_passed = None
        if lidar is not None:
            reading = lidar.read()
            if reading is not None:
                people_nearby = reading.people_nearby
                crowd_density = reading.crowd_density
                nearest = reading.nearest_distance_m
                bearing = reading.approach_bearing_deg
                lidar_people = reading.people
                # take_passed drains the entries accumulated since the last tick,
                # so summing lidar_passed in the cloud yields unique bodies seen.
                lidar_passed = lidar.take_passed()

        # Emit confirmed-people metrics: a flickering false positive never
        # reaches min_hits, so it contributes to none of these.
        conf_dists = [t.distance_m for t in confirmed if t.distance_m is not None]
        mean_d = sum(conf_dists) / len(conf_dists) if conf_dists else None
        attended_confirmed = sum(
            1 for t in confirmed if t.distance_m is not None and t.distance_m <= self._attn
        )
        return SceneState(
            person_count=len(confirmed),
            attended_count=attended_confirmed,
            mean_distance_m=mean_d,
            people_nearby=people_nearby,
            crowd_density=crowd_density,
            nearest_distance_m=nearest,
            approach_bearing_deg=bearing,
            lidar_people=lidar_people,
            lidar_passed=lidar_passed,
            looked_count=looked_count,
            mean_dwell_s=round(mean_dwell, 2) if mean_dwell is not None else None,
            interactions=tuple(interactions),
            timestamp=now,
        )

    # --- helpers ---

    def _forward_hints(self, depth, kp, body_dist):
        """Is each wrist reaching toward the robot (closer than the torso)?"""
        if body_dist is None:
            return False, False

        def reaching(i):
            if i >= len(kp) or kp[i] is None or kp[i][2] < 0.3:
                return False
            wd = self._depth_at(depth, kp[i][0], kp[i][1])
            return wd is not None and wd < body_dist - _FORWARD_MARGIN_M

        return reaching(L_WRIST), reaching(R_WRIST)

    @staticmethod
    def _match_tracks_to_people(person_boxes, tracks) -> dict[int, int]:
        """idx-of-person-box -> track_id, by nearest centroid (1:1 greedy)."""
        out: dict[int, int] = {}
        used: set[int] = set()
        for idx, b in enumerate(person_boxes):
            best_tid, best_d = None, float("inf")
            for t in tracks:
                if t.track_id in used:
                    continue
                d = ((t.cx - b.cx) ** 2 + (t.cy - b.cy) ** 2) ** 0.5
                if d < best_d:
                    best_tid, best_d = t.track_id, d
            if best_tid is not None:
                out[idx] = best_tid
                used.add(best_tid)
        return out

    def _allow(self, track_id: int, kind: str, now: float) -> bool:
        key = (track_id, kind)
        if now - self._last_fired.get(key, -1e9) < self._cooldown:
            return False
        self._last_fired[key] = now
        return True
