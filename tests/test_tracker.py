"""Tracker unit tests — pure logic, no hardware, no heavy deps.

These exercise the association, dwell, gaze-accumulation, and aging behaviour
that the dwell/unique-count metrics all depend on.
"""
from kovio.adapters.tracker import CentroidTracker, Detection


def test_stable_id_across_frames():
    tr = CentroidTracker(max_distance_px=50)
    a = tr.update([Detection(100, 100)], now=0.0)
    assert len(a) == 1
    tid = a[0].track_id
    # Person drifts a little each frame — same id.
    b = tr.update([Detection(110, 105)], now=0.5)
    c = tr.update([Detection(120, 110)], now=1.0)
    assert b[0].track_id == tid
    assert c[0].track_id == tid


def test_new_person_gets_new_id():
    tr = CentroidTracker(max_distance_px=50)
    tr.update([Detection(100, 100)], now=0.0)
    # A detection far away cannot be the same track -> new id.
    live = tr.update([Detection(100, 100), Detection(400, 400)], now=0.5)
    ids = sorted(t.track_id for t in live)
    assert ids == [1, 2]


def test_dwell_accumulates():
    tr = CentroidTracker()
    tr.update([Detection(100, 100)], now=0.0)
    tr.update([Detection(100, 100)], now=3.0)
    live = tr.update([Detection(100, 100)], now=5.0)
    assert abs(live[0].dwell_seconds - 5.0) < 1e-6
    assert abs((tr.mean_dwell_seconds() or 0) - 5.0) < 1e-6


def test_gaze_seconds_only_count_while_looking():
    tr = CentroidTracker(gaze_dwell_seconds=1.0)
    tr.update([Detection(100, 100, looking=False)], now=0.0)
    tr.update([Detection(100, 100, looking=True)], now=1.0)   # +1s looking
    tr.update([Detection(100, 100, looking=True)], now=2.0)   # +1s looking
    live = tr.update([Detection(100, 100, looking=False)], now=3.0)  # no add
    assert abs(live[0].looking_seconds - 2.0) < 1e-6


def test_gaze_dwell_event_fires_once():
    tr = CentroidTracker(gaze_dwell_seconds=1.5)
    tr.update([Detection(100, 100, looking=True)], now=0.0)
    tr.update([Detection(100, 100, looking=True)], now=1.0)   # 1.0s — under threshold
    assert tr.new_gaze_dwell_tracks() == []
    tr.update([Detection(100, 100, looking=True)], now=2.0)   # 2.0s — crosses 1.5
    fired = tr.new_gaze_dwell_tracks()
    assert len(fired) == 1
    # Does not re-fire on subsequent frames.
    tr.update([Detection(100, 100, looking=True)], now=3.0)
    assert tr.new_gaze_dwell_tracks() == []


def test_track_ages_out_after_miss_budget():
    tr = CentroidTracker(max_missed=2)
    tr.update([Detection(100, 100)], now=0.0)
    tr.update([], now=0.1)   # miss 1
    tr.update([], now=0.2)   # miss 2
    live = tr.update([], now=0.3)  # miss 3 -> dropped
    assert live == []
    # A reappearance now gets a fresh id.
    again = tr.update([Detection(100, 100)], now=0.4)
    assert again[0].track_id == 2
