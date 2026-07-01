"""Lidar crowd-analysis tests — synthetic pointclouds, pure math."""
import math
import struct

import pytest

from kovio.adapters.lidar import analyze_pointcloud, CrowdReading


def _blob(cx, cy, n=6, z=1.0, spread=0.1):
    """A small cluster of n points around (cx, cy) at standing height."""
    out = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        out.append((cx + spread * math.cos(ang), cy + spread * math.sin(ang), z))
    return out


def test_empty_cloud():
    r = analyze_pointcloud([])
    assert r == CrowdReading(0, 0.0, None, None)


def test_two_people_counted():
    pts = _blob(2.0, 0.0) + _blob(1.0, 1.5)
    r = analyze_pointcloud(pts, radius_m=4.0, eps_m=0.5, min_points=3)
    assert r.people_nearby == 2
    assert r.crowd_density == round(2 / (math.pi * 16), 4)


def test_ground_and_ceiling_filtered():
    ground = [(1.0, 0.0, 0.0)] * 10        # z below z_min
    ceiling = [(1.0, 0.0, 3.0)] * 10       # z above z_max
    person = _blob(1.5, 0.0)
    r = analyze_pointcloud(ground + ceiling + person)
    assert r.people_nearby == 1


def test_out_of_radius_excluded():
    near = _blob(1.0, 0.0)
    far = _blob(10.0, 0.0)                  # beyond radius
    r = analyze_pointcloud(near + far, radius_m=4.0)
    assert r.people_nearby == 1


def test_wide_object_rejected_as_furniture():
    # A 3 m wide wall of points — exceeds max_person_width.
    wall = [(2.0, y, 1.0) for y in [-1.5 + 0.1 * i for i in range(31)]]
    r = analyze_pointcloud(wall, eps_m=0.5, min_points=3, max_person_width_m=1.0)
    assert r.people_nearby == 0


def test_nearest_and_bearing_front():
    pts = _blob(2.0, 0.0)                   # straight ahead
    r = analyze_pointcloud(pts)
    assert abs(r.nearest_distance_m - 2.0) < 0.2
    assert abs(r.approach_bearing_deg) < 5  # ~0 deg = front


def test_bearing_right_is_positive():
    # Body on the robot's right (+x forward, -y right) -> positive bearing.
    pts = _blob(1.0, -1.0)
    r = analyze_pointcloud(pts)
    assert r.approach_bearing_deg > 0
    # nearer body wins bearing
    pts2 = _blob(1.0, -1.0) + _blob(3.0, 2.0)
    r2 = analyze_pointcloud(pts2)
    assert r2.people_nearby == 2 and r2.approach_bearing_deg > 0


def test_people_positions_emitted_nearest_first():
    # Two bodies: one ahead-left far, one right near. people[] is nearest-first
    # polar (range, bearing CW), so the near one (positive bearing) leads.
    pts = _blob(3.0, 1.0) + _blob(1.0, -1.0)
    r = analyze_pointcloud(pts)
    assert r.people_nearby == 2
    assert len(r.people) == 2
    near_rng, near_bear = r.people[0]
    assert abs(near_rng - math.hypot(1.0, 1.0)) < 0.2
    assert near_bear > 0                      # body on the right -> +bearing
    assert r.people[1][0] > r.people[0][0]    # sorted by range


def test_empty_reading_has_no_people():
    assert analyze_pointcloud([]).people == ()


def test_count_new_entries_pure():
    from kovio.adapters.lidar import count_new_entries

    # nobody before, two now -> two entries
    assert count_new_entries([], [(1.0, 0.0), (2.0, 1.0)]) == 2
    # same two roughly where they were -> nobody new
    assert count_new_entries([(1.0, 0.0), (2.0, 1.0)], [(1.05, 0.0), (2.0, 1.1)]) == 0
    # one stayed, one left, one fresh arrival -> exactly one entry
    assert count_new_entries([(1.0, 0.0), (5.0, 5.0)], [(1.0, 0.0), (0.0, -2.0)]) == 1
    # a jump beyond the gate reads as a new body, not the same person moved
    assert count_new_entries([(1.0, 0.0)], [(4.0, 0.0)], gate_m=0.8) == 1


class _Field:
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset


class _FakePC2:
    """A minimal ROS PointCloud2: x,y,z float32 then an intensity float32."""

    def __init__(self, points):
        self.point_step = 16
        self.fields = [_Field("x", 0), _Field("y", 4), _Field("z", 8),
                       _Field("intensity", 12)]
        buf = bytearray()
        for x, y, z in points:
            buf += struct.pack("<ffff", x, y, z, 1.0)
        self.data = bytes(buf)


def test_lidar_backend_off_is_inert(monkeypatch):
    """With the backend disabled, construction never touches DDS and every field
    reports 'no lidar' — the safe degrade `kovio doctor` relies on."""
    from kovio.adapters.lidar import LidarSource

    monkeypatch.setenv("KOVIO_LIDAR_BACKEND", "off")
    monkeypatch.setenv("KOVIO_LIDAR_TOPIC", "rt/utlidar/cloud")
    src = LidarSource(network_interface="does-not-exist0")

    assert src.available is False
    assert src.backend is None
    assert src.topic == "rt/utlidar/cloud"     # env override, nothing attached
    assert src.read() is None


def test_lidar_decode_pointcloud2_uses_field_offsets():
    """Decoding reads the real x/y/z offsets, so a Livox cloud with trailing
    intensity/tag columns still yields the right (x,y,z)."""
    from kovio.adapters.lidar import LidarSource

    pts = LidarSource._decode_pointcloud2(_FakePC2([(1.0, 2.0, 3.0), (-1.0, 0.5, 0.2)]))
    assert len(pts) == 2
    assert pytest.approx(pts[0]) == (1.0, 2.0, 3.0)
    assert pytest.approx(pts[1]) == (-1.0, 0.5, 0.2)


def test_lidar_ingest_locks_backend_and_counts(monkeypatch):
    """A cloud from either backend produces a reading; the first backend to
    deliver LOCKS the source so a second one can't double-count 'passed'."""
    from kovio.adapters.lidar import LidarSource

    monkeypatch.setenv("KOVIO_LIDAR_BACKEND", "off")
    src = LidarSource(radius_m=4.0)
    assert src.read() is None

    src._ingest("ros2_livox", _FakePC2(_blob(1.0, 0.5) + _blob(2.5, -1.0)))
    reading = src.read()
    assert reading is not None and reading.people_nearby == 2
    assert src.backend == "ros2_livox"
    assert src.take_passed() == 2               # two bodies entered the field

    # once locked to ros2_livox, a unitree_dds cloud is ignored (no double count)
    src._ingest("unitree_dds", _FakePC2(_blob(1.2, 0.0)))
    assert src.backend == "ros2_livox"
    assert src.take_passed() == 0
