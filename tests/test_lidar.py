"""Lidar crowd-analysis tests — synthetic pointclouds, pure math."""
import math

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


def test_lidar_source_reports_backend_and_topic(monkeypatch):
    """`kovio doctor` reads these to flag a dead lidar before a live demo:
    construction never raises on a host without a lidar, `available`/`backend`
    reflect that, `topic` honours the env override, and `read()` starts empty."""
    from kovio.adapters.lidar import LidarSource

    monkeypatch.setenv("KOVIO_LIDAR_TOPIC", "rt/utlidar/cloud")
    src = LidarSource(network_interface="does-not-exist0")

    assert src.topic == "rt/utlidar/cloud"          # env override wins
    assert src.available == (src.backend is not None)
    assert src.read() is None                        # no cloud yet -> no frame
