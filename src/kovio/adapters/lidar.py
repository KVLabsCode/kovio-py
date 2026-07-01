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

    Backends, tried in order: the Unitree DDS pointcloud topic (via the
    ``unitree_sdk2py`` bindings) and a Livox stream. Construction never raises
    on a host without a lidar — ``available`` reports False and ``read()``
    returns None so the fused adapter simply omits lidar metrics.

    The default topic is the Livox MID-360 cloud the Unitree firmware publishes
    (``rt/utlidar/cloud_livox_mid360``); the legacy Go2 ``rt/utlidar/cloud`` can
    be selected explicitly. Override with the ``KOVIO_LIDAR_TOPIC`` env var.
    """

    def __init__(
        self,
        radius_m: float = 4.0,
        network_interface: str = "eth0",
        topic: str = "rt/utlidar/cloud_livox_mid360",
        entry_gate_m: float = 0.8,
    ) -> None:
        import os

        self.radius_m = radius_m
        self._entry_gate_m = entry_gate_m
        self._latest: CrowdReading | None = None
        # unique "people passed by" accounting (frame-to-frame body matching).
        self._prev_xy: list[tuple[float, float]] = []
        self._passed_accum = 0
        self._sub = None
        self._backend = None
        # Resolved topic is kept so diagnostics (`kovio doctor`) can report which
        # DDS topic this source is (or would be) listening on.
        self._topic = os.environ.get("KOVIO_LIDAR_TOPIC", topic)
        try:
            self._init_unitree_dds(network_interface, self._topic)
            self._backend = "unitree_dds"
        except Exception as e:  # noqa: BLE001 - any failure means "no lidar here"
            log.info("lidar: unitree DDS backend unavailable (%s)", e)

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> str | None:
        """The active backend name (e.g. ``"unitree_dds"``), or None when no
        lidar could be attached on this host."""
        return self._backend

    @property
    def topic(self) -> str:
        """The DDS pointcloud topic this source subscribes to."""
        return self._topic

    def _init_unitree_dds(self, iface: str, topic: str) -> None:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # type: ignore

        ChannelFactoryInitialize(0, iface)
        self._sub = ChannelSubscriber(topic, PointCloud2_)
        self._sub.Init(self._on_cloud, 10)

    def _on_cloud(self, msg) -> None:
        try:
            pts = self._decode_pointcloud2(msg)
            reading = analyze_pointcloud(pts, radius_m=self.radius_m)
            curr_xy = _polar_to_xy(reading.people)
            self._passed_accum += count_new_entries(
                self._prev_xy, curr_xy, self._entry_gate_m
            )
            self._prev_xy = curr_xy
            self._latest = reading
        except Exception:
            log.exception("lidar: failed to process cloud")

    @staticmethod
    def _decode_pointcloud2(msg) -> list:
        """Decode a ROS-style PointCloud2 (x,y,z float32 fields) to a list."""
        import struct

        step = msg.point_step
        data = bytes(msg.data)
        out = []
        for off in range(0, len(data), step):
            x, y, z = struct.unpack_from("<fff", data, off)
            out.append((x, y, z))
        return out

    def read(self) -> CrowdReading | None:
        """Most recent crowd reading, or None if no lidar is attached."""
        return self._latest

    def take_passed(self) -> int:
        """Unique bodies that ENTERED the lidar field since the last call, then
        reset. Summed in the cloud into a cumulative "people passed by" count."""
        n, self._passed_accum = self._passed_accum, 0
        return n
