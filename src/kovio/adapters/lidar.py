"""Lidar crowd & proximity analysis for the Unitree Go2 (Livox L1).

The depth camera sees a narrow cone in front of the screen; the lidar sees a
wide field all around the robot. That makes it the right sensor for "how many
people are *around* the robot", who is closest, and from which direction they
are approaching — context the camera can't supply.

The math (``analyze_pointcloud``) is pure and unit-tested: filter to a height
band and radius, cluster the survivors into person-sized blobs, and read off
count, density, nearest range, and the bearing of the nearest body. The driver
glue (``LidarSource``) is thin and lazy — it reads the Go2's DDS ``rt/utlidar/
cloud`` topic when the unitree_sdk2 python bindings are present, or a Livox
stream, and degrades to ``None`` (no lidar metrics) everywhere else.

Frame convention: robot-body frame, +x forward, +y left, +z up (metres).
Bearing is degrees clockwise from straight ahead: 0 = front, +90 = right.
"""
from __future__ import annotations

import logging
import math
import os
import struct
import threading
import time
from dataclasses import dataclass

log = logging.getLogger("kovio.perception.lidar")


@dataclass(frozen=True)
class CrowdReading:
    people_nearby: int
    crowd_density: float          # people per m^2 within the radius
    nearest_distance_m: float | None
    approach_bearing_deg: float | None
    # Per-person polar positions for the live radar: (range_m, bearing_deg),
    # bearing 0=front, +=right (clockwise), nearest first. Empty when no bodies.
    people: tuple[tuple[float, float], ...] = ()


def _dbscan_2d(points, eps, min_points):
    """Minimal DBSCAN over 2D points. Returns a list of clusters (point lists)."""
    n = len(points)
    eps2 = eps * eps
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= eps2:
                neighbors[i].append(j)
                neighbors[j].append(i)

    labels = [-1] * n          # -1 = unvisited/noise
    clusters: list[list] = []
    for i in range(n):
        if labels[i] != -1:
            continue
        if len(neighbors[i]) + 1 < min_points:
            continue  # not a core point; leave as noise for now
        cid = len(clusters)
        clusters.append([])
        stack = [i]
        labels[i] = cid
        while stack:
            p = stack.pop()
            clusters[cid].append(points[p])
            if len(neighbors[p]) + 1 >= min_points:  # core -> expand
                for q in neighbors[p]:
                    if labels[q] == -1:
                        labels[q] = cid
                        stack.append(q)
    return clusters


def analyze_pointcloud(
    points,
    radius_m: float = 4.0,
    eps_m: float = 0.5,
    min_points: int = 3,
    z_min: float = 0.2,
    z_max: float = 2.0,
    max_person_width_m: float = 1.0,
) -> CrowdReading:
    """Turn a raw pointcloud into a crowd reading. Pure; no I/O.

    ``points`` is any iterable of (x, y, z) in robot-body metres.
    """
    r2 = radius_m * radius_m
    flat = []
    for p in points:
        x, y, z = p[0], p[1], p[2]
        if z_min <= z <= z_max and (x * x + y * y) <= r2:
            flat.append((x, y))
    if not flat:
        return CrowdReading(0, 0.0, None, None)

    people = []  # (range, x, y)
    for cl in _dbscan_2d(flat, eps_m, min_points):
        xs = [p[0] for p in cl]
        ys = [p[1] for p in cl]
        width = max(max(xs) - min(xs), max(ys) - min(ys))
        if width > max_person_width_m:
            continue  # too wide to be one person (wall, furniture)
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        people.append((math.hypot(cx, cy), cx, cy))

    count = len(people)
    density = count / (math.pi * r2) if r2 > 0 else 0.0
    if not people:
        return CrowdReading(0, round(density, 4), None, None)

    people.sort(key=lambda t: t[0])
    # Polar position of every body (nearest first) for the radar; +y is left so
    # negate to get clockwise bearing (0=front, +90=right).
    positions = tuple(
        (round(rng, 2), round(math.degrees(math.atan2(-cy, cx)), 1))
        for rng, cx, cy in people
    )
    rng, nx, ny = people[0]
    bearing = math.degrees(math.atan2(-ny, nx))  # +y is left -> negate for CW
    return CrowdReading(
        count, round(density, 4), round(rng, 2), round(bearing, 1), positions
    )


def _polar_to_xy(positions) -> list[tuple[float, float]]:
    """(range_m, bearing_deg CW from front) -> body-frame (x fwd, y left)."""
    out = []
    for rng, bearing in positions:
        a = math.radians(bearing)
        out.append((rng * math.cos(a), -rng * math.sin(a)))  # bearing is CW
    return out


def count_new_entries(prev_xy, curr_xy, gate_m: float = 0.8) -> int:
    """How many ``curr_xy`` bodies are NEW vs ``prev_xy`` (greedy nearest match).

    A body that matches a previous one within ``gate_m`` is the same person who
    was already in the field; an unmatched body just entered. Pure so the
    "people passed by" count can be unit-tested without a lidar. This is the
    lidar analogue of the tracker's unique-reach fix: each person is counted
    once on entry, never once per frame.
    """
    used: set[int] = set()
    new = 0
    for cx, cy in curr_xy:
        best, best_d = None, gate_m
        for j, (px, py) in enumerate(prev_xy):
            if j in used:
                continue
            d = math.hypot(cx - px, cy - py)
            if d <= best_d:
                best, best_d = j, d
        if best is None:
            new += 1
        else:
            used.add(best)
    return new


class LidarSource:
    """Pulls pointclouds off the robot and exposes the latest crowd reading.

    Two DDS backends are attached concurrently and whichever actually DELIVERS a
    cloud wins (the source then locks onto it, so counts never double):

    * ``ros2_livox`` — the Livox ROS 2 topic ``/livox/lidar`` (DDS name
      ``rt/livox/lidar``), read straight off cyclonedds with **best-effort** QoS
      to match the sensor-data publisher. This is how the Unitree **G1** exposes
      its Livox MID-360; ``rclpy`` is *not* required (and isn't importable in the
      SDK's Python anyway), so we speak DDS directly via the ``cyclonedds``
      bindings that ship with ``unitree_sdk2py``.
    * ``unitree_dds`` — the Unitree firmware cloud (``rt/utlidar/cloud_livox_mid360``,
      or the legacy Go2 ``rt/utlidar/cloud``) via ``unitree_sdk2py``'s subscriber.

    Construction NEVER raises on a host without a lidar — ``available`` reports
    False and ``read()`` returns None so the fused adapter simply omits lidar
    metrics. Init is also RETRIED lazily on every ``read()``: a backend that
    couldn't start yet (e.g. ``eth0`` had no address at boot, or the Livox driver
    wasn't up) is re-attempted, so the lidar recovers WITHOUT a process restart —
    the fix for the boot-time race that used to disable it for the whole run.

    Env overrides: ``KOVIO_LIDAR_BACKEND`` (``auto`` | ``ros2`` | ``unitree_dds``
    | ``off``), ``KOVIO_LIDAR_TOPIC`` (unitree DDS topic), ``KOVIO_LIDAR_ROS2_TOPIC``
    (default ``rt/livox/lidar``), ``KOVIO_LIDAR_NET_IFACE`` (default ``eth0``).
    """

    def __init__(
        self,
        radius_m: float = 4.0,
        network_interface: str = "eth0",
        topic: str = "rt/utlidar/cloud_livox_mid360",
        entry_gate_m: float = 0.8,
        ros2_topic: str = "rt/livox/lidar,rt/utlidar/cloud_livox_mid360",
        retry_seconds: float = 5.0,
        engine=None,
    ) -> None:
        self.radius_m = radius_m
        # Optional AudienceEngine: when set it does the clustering/tracking per
        # cloud (the V2 moments) and this source derives its CrowdReading from
        # the engine's clusters instead of running analyze_pointcloud twice.
        self._engine = engine
        self._entry_gate_m = entry_gate_m
        self._retry_s = retry_seconds
        self._latest: CrowdReading | None = None
        # unique "people passed by" accounting (frame-to-frame body matching).
        self._prev_xy: list[tuple[float, float]] = []
        self._passed_accum = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # The backend that first delivered a cloud; once set, others are ignored.
        self._active_backend: str | None = None

        self._iface = os.environ.get("KOVIO_LIDAR_NET_IFACE", network_interface)
        self._dds_topic = os.environ.get("KOVIO_LIDAR_TOPIC", topic)
        # The cyclonedds reader tries several candidate topics (best-effort matches
        # both the livox_ros_driver2 output AND the firmware's native cloud, which
        # unitree_sdk2py's own subscriber fails to type-match).
        self._ros2_topics = [
            t.strip()
            for t in os.environ.get("KOVIO_LIDAR_ROS2_TOPIC", ros2_topic).split(",")
            if t.strip()
        ]
        self._backend_pref = os.environ.get("KOVIO_LIDAR_BACKEND", "auto").lower()

        self._dds_sub = None
        self._ros2_readers: list = []
        self._started = {"unitree_dds": False, "ros2_livox": False}
        # The Unitree SDK owns the process-wide cyclonedds domain (and binds it to
        # the robot NIC); when present we bring it up FIRST and let both readers
        # join domain 0, rather than racing a bare participant onto it.
        self._factory_ready = False
        self._last_attempt = 0.0
        self._ensure_started(force=True)

    # ---- public surface ------------------------------------------------- #

    @property
    def available(self) -> bool:
        """True once at least one backend has a live listener attached."""
        return any(self._started.values())

    @property
    def backend(self) -> str | None:
        """The backend delivering data (once one does), else the first attached
        listener, else None when no lidar could be attached on this host."""
        if self._active_backend:
            return self._active_backend
        return next((b for b, up in self._started.items() if up), None)

    @property
    def topic(self) -> str:
        """The topic this source is receiving on (or listening on if not yet
        locked). Reported by ``kovio doctor``."""
        ros2 = ", ".join(self._ros2_topics)
        if self._active_backend == "ros2_livox":
            return ros2
        if self._active_backend == "unitree_dds":
            return self._dds_topic
        listening = []
        if self._started.get("ros2_livox"):
            listening.append(ros2)
        if self._started.get("unitree_dds"):
            listening.append(self._dds_topic)
        return ", ".join(listening) or self._dds_topic

    def read(self) -> CrowdReading | None:
        """Most recent crowd reading, or None if no lidar is delivering yet.

        Also drives lazy (re)connection, so a backend that wasn't ready at
        construction time is retried here until it starts producing clouds.
        """
        self._ensure_started()
        return self._latest

    def take_passed(self) -> int:
        """Unique bodies that ENTERED the lidar field since the last call, then
        reset. Summed in the cloud into a cumulative "people passed by" count.

        With an engine attached this is the tracker's movement-gated passerby
        count (immune to cluster flicker); the frame-matching accumulator is
        the engine-less fallback."""
        if self._engine is not None:
            return self._engine.take_passed()
        with self._lock:
            n, self._passed_accum = self._passed_accum, 0
        return n

    def close(self) -> None:
        self._stop.set()

    # ---- backend lifecycle ---------------------------------------------- #

    @staticmethod
    def _unitree_importable() -> bool:
        import importlib.util

        return importlib.util.find_spec("unitree_sdk2py") is not None

    def _ensure_started(self, force: bool = False) -> None:
        """Attempt to start any not-yet-running backend, rate-limited to one try
        per ``retry_seconds`` so a persistently-absent lidar costs almost nothing.

        Ordering matters: when the Unitree SDK is present it must initialise the
        shared cyclonedds domain (bound to the robot NIC) BEFORE any bare
        participant, or the domain gets created on the wrong interface and the
        Unitree subscriber can never attach. So we gate the ROS 2 reader on the
        factory being ready, and both come up together once ``eth0`` has an IP.
        """
        pref = self._backend_pref
        if pref == "off":
            return
        want_ros2 = pref in ("auto", "ros2", "ros2_livox")
        want_dds = pref in ("auto", "unitree_dds", "dds")
        if (self._started["ros2_livox"] or not want_ros2) and (
            self._started["unitree_dds"] or not want_dds
        ):
            return
        now = time.monotonic()
        if not force and now - self._last_attempt < self._retry_s:
            return
        self._last_attempt = now

        # Explicit "ros2" skips the Unitree factory and uses a bare participant.
        use_factory = want_dds and pref != "ros2" and self._unitree_importable()
        if use_factory and not self._factory_ready:
            try:
                self._init_factory()
                self._factory_ready = True
            except Exception as e:  # noqa: BLE001 - eth0 not up yet, retry next tick
                log.info("lidar: DDS domain not ready on %s (%s)", self._iface, e)
                # Don't start a bare ROS 2 participant while the Unitree SDK still
                # owns domain 0 — wait so both attach to the same NIC together.
                return

        if want_dds and self._factory_ready and not self._started["unitree_dds"]:
            try:
                self._start_unitree_dds_sub()
                self._started["unitree_dds"] = True
                log.info("lidar: unitree_dds backend listening on %s", self._dds_topic)
            except Exception as e:  # noqa: BLE001
                log.info("lidar: unitree_dds backend unavailable (%s)", e)
        if want_ros2 and not self._started["ros2_livox"]:
            try:
                self._start_ros2_livox()
                self._started["ros2_livox"] = True
                log.info(
                    "lidar: ros2_livox backend listening on %s",
                    ", ".join(self._ros2_topics),
                )
            except Exception as e:  # noqa: BLE001 - no ros2/cyclonedds here is fine
                log.info("lidar: ros2_livox backend unavailable (%s)", e)

    def _init_factory(self) -> None:
        """Bring up the Unitree SDK's shared cyclonedds domain, bound to the NIC."""
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore

        ChannelFactoryInitialize(0, self._iface)

    def _start_unitree_dds_sub(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # type: ignore

        self._dds_sub = ChannelSubscriber(self._dds_topic, PointCloud2_)
        self._dds_sub.Init(lambda msg: self._ingest("unitree_dds", msg), 10)

    def _start_ros2_livox(self) -> None:
        """Subscribe to the Livox ROS 2 cloud straight off cyclonedds (no rclpy),
        with best-effort QoS so a ROS 2 sensor-data publisher actually matches."""
        from cyclonedds.domain import DomainParticipant  # type: ignore
        from cyclonedds.qos import Policy, Qos  # type: ignore
        from cyclonedds.sub import DataReader  # type: ignore
        from cyclonedds.topic import Topic  # type: ignore
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # type: ignore

        qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.History.KeepLast(4),
            Policy.Durability.Volatile,
        )
        participant = DomainParticipant(0)
        # Hold refs so they aren't GC'd out from under the readers.
        self._ros2_participant = participant
        self._ros2_topic_objs = []
        self._ros2_readers = []
        for name in self._ros2_topics:
            topic = Topic(participant, name, PointCloud2_, qos=qos)
            self._ros2_topic_objs.append(topic)
            self._ros2_readers.append(DataReader(participant, topic, qos=qos))
        threading.Thread(
            target=self._ros2_poll_loop, name="kovio-lidar-ros2", daemon=True
        ).start()

    def _ros2_poll_loop(self) -> None:
        while not self._stop.is_set():
            for reader in self._ros2_readers:
                try:
                    samples = reader.take(N=8)
                except Exception:
                    samples = []
                for s in samples or []:
                    # cyclonedds may hand back the message itself or wrap it in a
                    # sample; probe for a PointCloud2 field before unwrapping,
                    # because the message's own ``data`` (the point bytes) shadows
                    # a sample wrapper's ``data`` attribute.
                    msg = s if hasattr(s, "point_step") else getattr(s, "data", s)
                    self._ingest("ros2_livox", msg)
            time.sleep(0.05)

    # ---- shared ingest -------------------------------------------------- #

    def _ingest(self, backend_name: str, msg) -> None:
        """Decode a PointCloud2 from either backend into the latest reading.

        The first backend to deliver a cloud LOCKS the source (``_active_backend``)
        so a second backend that's also up can't double-count "passed" bodies.
        """
        with self._lock:
            if self._active_backend is None:
                self._active_backend = backend_name
                log.info("lidar: receiving clouds via %s", backend_name)
            elif self._active_backend != backend_name:
                return
        try:
            pts = self._decode_pointcloud2(msg)
            if self._engine is not None:
                clusters = self._engine.ingest_points(pts)
                reading = self._reading_from_clusters(clusters)
            else:
                reading = analyze_pointcloud(pts, radius_m=self.radius_m)
            curr_xy = _polar_to_xy(reading.people)
            with self._lock:
                self._passed_accum += count_new_entries(
                    self._prev_xy, curr_xy, self._entry_gate_m
                )
                self._prev_xy = curr_xy
                self._latest = reading
        except Exception:
            log.exception("lidar: failed to process cloud")

    def _reading_from_clusters(self, clusters) -> CrowdReading:
        """CrowdReading from the engine's clusters (already nearest-first)."""
        within = [c for c in clusters if c.range_m <= self.radius_m]
        density = len(within) / (math.pi * self.radius_m**2)
        if not within:
            return CrowdReading(0, round(density, 4), None, None)
        positions = tuple(
            (round(c.range_m, 2), round(c.bearing_deg, 1)) for c in within
        )
        return CrowdReading(
            len(within),
            round(density, 4),
            positions[0][0],
            positions[0][1],
            positions,
        )

    @staticmethod
    def _decode_pointcloud2(msg):
        """Decode a ROS-style PointCloud2 (x,y,z float32) to Nx3 (x,y,z).

        Reads the actual x/y/z field offsets so it handles both the Unitree
        cloud and the Livox cloud (which carry intensity/tag/etc. after xyz).
        numpy path handles the MID-360's ~200k pts/s; the pure-python loop is a
        fallback so hosts without numpy still get crowd metrics."""
        step = msg.point_step
        data = bytes(msg.data)
        off = {f.name: f.offset for f in msg.fields} if getattr(msg, "fields", None) else {}
        ox, oy, oz = off.get("x", 0), off.get("y", 4), off.get("z", 8)
        n = len(data) // step if step else 0
        try:
            import numpy as np

            buf = np.frombuffer(data, dtype=np.uint8, count=n * step).reshape(n, step)

            def col(o):
                return buf[:, o : o + 4].copy().view(np.float32).ravel()

            return np.stack([col(ox), col(oy), col(oz)], axis=1)
        except ImportError:
            pass
        out = []
        for i in range(n):
            base = i * step
            x = struct.unpack_from("<f", data, base + ox)[0]
            y = struct.unpack_from("<f", data, base + oy)[0]
            z = struct.unpack_from("<f", data, base + oz)[0]
            out.append((x, y, z))
        return out
