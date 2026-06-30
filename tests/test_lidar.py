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
