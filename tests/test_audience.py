"""Audience engine tests — synthetic clouds/centroids, no hardware.

Covers the three V2 moments (passerby / dwell / close_approach), the
static-clutter guards, and the encounter-cap de-duplication that makes reach a
UNIQUE count.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from kovio.adapters.audience import (
    AudienceEngine,
    AudienceTracker,
    BackgroundModel,
    Cluster,
    extract_clusters,
)


def _blob3d(cx, cy, n=40, z=1.0, spread=0.12):
    pts = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        r = spread * (0.3 + 0.7 * ((i * 7) % 10) / 10)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), z + 0.3 * math.sin(i)))
    return pts


def _clusters_at(*centres):
    return [Cluster(cx, cy, 40, 0.4) for cx, cy in centres]


# --------------------------------------------------------------- extraction --

def test_extract_finds_person_sized_blob():
    pts = _blob3d(2.0, 0.5)
    cl = extract_clusters(np.array(pts, dtype=np.float32))
    assert len(cl) == 1
    assert abs(cl[0].cx - 2.0) < 0.15 and abs(cl[0].cy - 0.5) < 0.15


def test_extract_rejects_wall_and_floor():
    wall = [(x, 2.0, 1.0) for x in np.arange(-3, 3, 0.05)]      # 6 m wide
    floor = [(x, y, 0.02) for x in np.arange(0.5, 2, 0.3) for y in np.arange(-1, 1, 0.3)]
    cl = extract_clusters(np.array(wall + floor, dtype=np.float32))
    assert cl == []


def test_extract_empty_and_out_of_band():
    assert extract_clusters(np.zeros((0, 3), dtype=np.float32)) == []
    high = np.array(_blob3d(1.5, 0.0, z=3.0), dtype=np.float32)  # above head band
    assert extract_clusters(high) == []


def test_two_legs_merge_into_one_body():
    left = _blob3d(2.0, 0.75, n=25, spread=0.08)
    right = _blob3d(2.0, 0.25, n=25, spread=0.08)  # 0.5 m apart: mid-stride
    cl = extract_clusters(np.array(left + right, dtype=np.float32))
    assert len(cl) == 1
    assert abs(cl[0].cy - 0.5) < 0.12  # centroid lands between the legs


def test_two_separate_people_stay_separate():
    a = _blob3d(2.0, 1.0, n=25, spread=0.08)
    b = _blob3d(2.0, -1.0, n=25, spread=0.08)  # 2 m apart
    cl = extract_clusters(np.array(a + b, dtype=np.float32))
    assert len(cl) == 2


def test_background_model_absorbs_static_object():
    bg = BackgroundModel(alpha=0.2, threshold=0.7)  # fast alpha for the test
    cells = {(10, 10), (10, 11)}
    static = set()
    for _ in range(30):
        static = bg.update(cells)
    assert static == cells
    # A fresh cell is not background.
    assert (5, 5) not in bg.update(cells | {(5, 5)})


# ----------------------------------------------------------------- passerby --

def test_walker_counts_once():
    tr = AudienceTracker()
    t, moments = 0.0, []
    # Walk from (4, 1) to (-4, 1) through the 3 m radius at ~1.3 m/s, 10 Hz.
    for i in range(60):
        x = 4.0 - i * 0.13
        moments += tr.update(_clusters_at((x, 1.0)), t)
        t += 0.1
    kinds = [m.kind for m in moments]
    assert kinds.count("passerby") == 1
    m = next(m for m in moments if m.kind == "passerby")
    assert m.payload["closest_m"] <= 3.0
    assert m.track_id == 1


def test_static_pillar_never_counts():
    tr = AudienceTracker()
    t, moments = 0.0, []
    for _ in range(300):  # 30 s of a motionless "person-sized" pillar at 2 m
        moments += tr.update(_clusters_at((2.0, 0.0)), t)
        t += 0.1
    assert moments == []


def test_jittering_static_object_never_counts():
    # Regression: centroid jitter accrues path length (~0.5 m/s observed live)
    # but no net displacement — it must not become a passerby or dwell.
    tr = AudienceTracker()
    t, moments = 0.0, []
    for i in range(600):  # 60 s of a jittery blob well inside the dwell zone
        jx = 0.06 * math.sin(i * 2.1)
        jy = 0.06 * math.cos(i * 1.3)
        moments += tr.update(_clusters_at((1.2 + jx, 0.1 + jy)), t)
        t += 0.1
    assert moments == []


def test_double_pass_within_cap_is_one_reach():
    tr = AudienceTracker(encounter_cap_s=300.0)
    t, moments = 0.0, []
    for i in range(40):  # pass one: walks in and out
        x = 4.0 - i * 0.2
        moments += tr.update(_clusters_at((x, 1.0)), t)
        t += 0.1
    for _ in range(30):  # 3 s gone (track dies)
        moments += tr.update([], t)
        t += 0.1
    for i in range(40):  # pass two: re-enters near where they left
        x = -4.0 + i * 0.2
        moments += tr.update(_clusters_at((x, 1.0)), t)
        t += 0.1
    assert [m.kind for m in moments].count("passerby") == 1


def test_reentry_after_cap_counts_again():
    tr = AudienceTracker(encounter_cap_s=5.0)
    t, moments = 0.0, []
    for i in range(40):
        moments += tr.update(_clusters_at((4.0 - i * 0.2, 1.0)), t)
        t += 0.1
    for _ in range(80):  # 8 s > 5 s cap
        moments += tr.update([], t)
        t += 0.1
    for i in range(40):
        moments += tr.update(_clusters_at((-4.0 + i * 0.2, 1.0)), t)
        t += 0.1
    assert [m.kind for m in moments].count("passerby") == 2


def test_two_people_two_reaches():
    tr = AudienceTracker()
    t, moments = 0.0, []
    for i in range(60):
        x = 4.0 - i * 0.13
        moments += tr.update(_clusters_at((x, 1.0), (x, -1.5)), t)
        t += 0.1
    assert [m.kind for m in moments].count("passerby") == 2
    ids = {m.track_id for m in moments if m.kind == "passerby"}
    assert len(ids) == 2


def _rotate(centres, deg):
    a = math.radians(deg)
    return [
        (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))
        for x, y in centres
    ]


def test_robot_turn_does_not_animate_the_furniture():
    # Ego-motion regression: the G1 balance-steps/turns, displacing every
    # static object in body frame at once. That must never mint passersby.
    tr = AudienceTracker()
    scene = [(2.0, 0.5), (1.2, -0.8), (3.0, 1.5), (2.5, -1.8), (3.5, 0.2)]
    t, moments = 0.0, []
    for _ in range(50):  # 5 s static
        moments += tr.update(_clusters_at(*scene), t)
        t += 0.1
    for step in range(20):  # robot turns 40° over 2 s (2°/frame, coherent)
        moments += tr.update(_clusters_at(*_rotate(scene, 2.0 * (step + 1))), t)
        t += 0.1
    for _ in range(100):  # 10 s settled again
        moments += tr.update(_clusters_at(*_rotate(scene, 40.0)), t)
        t += 0.1
    assert moments == []


def test_walker_still_counts_among_static_scene():
    # One real mover among several statics must not be masked by the guard.
    tr = AudienceTracker()
    statics = [(2.0, 1.5), (3.0, -1.5), (3.5, 0.8)]
    t, moments = 0.0, []
    for i in range(60):
        walker = (4.0 - i * 0.13, -0.5)
        moments += tr.update(_clusters_at(*statics, walker), t)
        t += 0.1
    assert [m.kind for m in moments].count("passerby") == 1


# -------------------------------------------------------------------- dwell --

def _walk_in_and_stand(tr, stand_s, t0=0.0, stand_at=(1.0, 0.3)):
    """Walk in from 4 m then stand at ``stand_at`` for ``stand_s`` seconds."""
    t, moments = t0, []
    for i in range(25):  # 2.5 s walk-in: 4 m -> ~1 m (also passes eligibility)
        x = 4.0 - i * 0.125
        moments += tr.update(_clusters_at((x, stand_at[1])), t)
        t += 0.1
    for _ in range(int(stand_s * 10)):
        moments += tr.update(_clusters_at(stand_at), t)
        t += 0.1
    return moments, t


def test_dwell_tiers_progress():
    tr = AudienceTracker()
    moments, _ = _walk_in_and_stand(tr, 13.0)
    dwells = [m for m in moments if m.kind == "dwell"]
    assert [d.payload["tier"] for d in dwells] == ["paused", "engaged", "deep"]
    # All tiers are the SAME person as the reach.
    assert {d.track_id for d in dwells} == {1}
    assert [m.kind for m in moments].count("passerby") == 1


def test_short_pause_is_not_dwell():
    tr = AudienceTracker()
    moments, _ = _walk_in_and_stand(tr, 2.0)  # < 3 s
    assert [m.kind for m in moments if m.kind == "dwell"] == []


def test_dwell_needs_to_be_close():
    tr = AudienceTracker()
    t, moments = 0.0, []
    for i in range(25):
        moments += tr.update(_clusters_at((4.0 - i * 0.08, 0.3)), t)
        t += 0.1
    for _ in range(80):  # stands 8 s but at ~2 m — outside the 1.5 m zone
        moments += tr.update(_clusters_at((2.0, 0.3)), t)
        t += 0.1
    assert [m for m in moments if m.kind == "dwell"] == []


# ----------------------------------------------------------- close approach --

def _engine_with_walkin(now):
    eng = AudienceEngine(warmup_s=0.0)
    eng.arm()
    t = now
    for i in range(25):  # person walks to 0.8 m in front
        pts = np.array(_blob3d(4.0 - i * 0.13, 0.0), dtype=np.float32)
        eng.ingest_points(pts, now=t)
        t += 0.1
    return eng, t


def test_close_approach_emits_with_lidar_coincidence():
    eng, t = _engine_with_walkin(1000.0)
    # Depth sees something at 0.7 m while a lidar track is in the cone.
    for _ in range(20):
        eng.set_camera(0.7, (0.8,), now=t)
        pts = np.array(_blob3d(0.8, 0.0), dtype=np.float32)
        eng.ingest_points(pts, now=t)
        t += 0.1
    for _ in range(5):  # backs away -> episode ends
        eng.set_camera(2.5, (2.4,), now=t)
        pts = np.array(_blob3d(2.4, 0.0), dtype=np.float32)
        eng.ingest_points(pts, now=t)
        t += 0.1
    kinds = [m["kind"] for m in eng.drain()]
    assert "close_approach" in kinds


def test_no_close_approach_without_lidar_track():
    eng = AudienceEngine(warmup_s=0.0)
    eng.arm()
    t = 2000.0
    for _ in range(30):  # hand over the lens: depth < 1 m, but lidar sees nobody
        eng.set_camera(0.4, (), now=t)
        eng.ingest_points(np.zeros((0, 3), dtype=np.float32), now=t)
        t += 0.1
    assert [m["kind"] for m in eng.drain()] == []


# ------------------------------------------------------------------- engine --

def test_engine_reset_on_arm_and_health():
    eng, t = _engine_with_walkin(3000.0)
    wire = eng.drain()
    assert [m["kind"] for m in wire] == ["passerby"]
    assert wire[0]["track_id"] == 1
    h = eng.health(now=t)
    assert h["lidar_ok"] and h["lidar_hz"] > 5
    assert h["depth_ok"] is False  # camera never reported

    eng.arm()  # new session: ids restart, pending cleared
    t2 = t + 100
    for i in range(25):
        pts = np.array(_blob3d(4.0 - i * 0.13, 0.0), dtype=np.float32)
        eng.ingest_points(pts, now=t2)
        t2 += 0.1
    wire = eng.drain()
    assert [m["kind"] for m in wire] == ["passerby"]
    assert wire[0]["track_id"] == 1  # session-scoped ids


def test_disarmed_engine_queues_nothing():
    eng = AudienceEngine(warmup_s=0.0)
    t = 4000.0
    for i in range(25):
        pts = np.array(_blob3d(4.0 - i * 0.13, 0.0), dtype=np.float32)
        eng.ingest_points(pts, now=t)
        t += 0.1
    assert eng.drain() == []


def test_take_passed_counts_movers_not_flicker():
    eng = AudienceEngine(warmup_s=0.0)  # NOT armed — passed counting is always on
    t = 6000.0
    # A static blob that flickers in and out never counts…
    for i in range(80):
        pts = (
            np.array(_blob3d(2.0, 1.0), dtype=np.float32)
            if i % 3 != 0
            else np.zeros((0, 3), dtype=np.float32)
        )
        eng.ingest_points(pts, now=t)
        t += 0.1
    assert eng.take_passed() == 0
    # …a walker counts exactly once.
    for i in range(30):
        pts = np.array(_blob3d(4.0 - i * 0.13, 0.0), dtype=np.float32)
        eng.ingest_points(pts, now=t)
        t += 0.1
    assert eng.take_passed() == 1
    assert eng.take_passed() == 0  # drained


def test_requeue_preserves_order_and_ids():
    eng, _ = _engine_with_walkin(5000.0)
    wire = eng.drain()
    eng.requeue(wire)
    again = eng.drain()
    assert again == wire
