"""Lightweight multi-object tracker — the foundation for dwell & unique counts.

A frame-by-frame people detector gives us *counts*; a tracker turns those into
*journeys*. Once a person carries a stable id across frames we can measure how
long they dwell, how long they hold their gaze on the screen, and avoid
double-counting the same person as a fresh "interaction" every tick.

This is deliberately dependency-light (pure Python + the stdlib): a greedy
nearest-centroid associator with a max match radius and a miss budget. It runs
comfortably at camera frame rate on a Jetson Orin alongside YOLO. Identity is
*never* persisted or exported — track ids are ephemeral integers that reset on
restart and never leave the robot.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class Detection:
    """One person seen in a single frame."""

    cx: float
    cy: float
    distance_m: float | None = None
    looking: bool = False  # gaze estimated on-screen this frame


@dataclass
class Track:
    """A person followed across frames. Identity-free; id is ephemeral."""

    track_id: int
    cx: float
    cy: float
    first_seen: float
    last_seen: float
    distance_m: float | None = None
    looking: bool = False
    looking_seconds: float = 0.0
    _missed: int = 0
    _gaze_event_fired: bool = field(default=False, repr=False)

    @property
    def dwell_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class CentroidTracker:
    """Greedy nearest-centroid tracker.

    Args:
        max_distance_px: a detection beyond this from a track's last centroid
            cannot be the same person (prevents id swaps across the frame).
        max_missed: drop a track after this many consecutive frames unseen.
        gaze_dwell_seconds: looking_seconds threshold at which a track is
            reported (once) as having a sustained gaze.
    """

    def __init__(
        self,
        max_distance_px: float = 120.0,
        max_missed: int = 5,
        gaze_dwell_seconds: float = 1.5,
    ) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed = max_missed
        self.gaze_dwell_seconds = gaze_dwell_seconds
        self._tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)
        self._last_now: float | None = None

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        """Associate ``detections`` to existing tracks; return live tracks.

        Pure function of (current tracks, detections, now) — no I/O. Call once
        per processed frame with a monotonic-ish wall clock.
        """
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now

        unmatched = set(range(len(detections)))
        # Greedy match: shortest pairwise distance first, each side used once.
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self._tracks.items():
            for di in unmatched:
                d = detections[di]
                dist = ((tr.cx - d.cx) ** 2 + (tr.cy - d.cy) ** 2) ** 0.5
                if dist <= self.max_distance_px:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks: set[int] = set()
        for _dist, tid, di in pairs:
            if tid in matched_tracks or di not in unmatched:
                continue
            tr = self._tracks[tid]
            det = detections[di]
            tr.cx, tr.cy = det.cx, det.cy
            tr.distance_m = det.distance_m
            tr.looking = det.looking
            tr.last_seen = now
            tr._missed = 0
            if det.looking:
                tr.looking_seconds += dt
            matched_tracks.add(tid)
            unmatched.discard(di)

        # Age out unmatched tracks.
        for tid, tr in list(self._tracks.items()):
            if tid not in matched_tracks:
                tr._missed += 1
                if tr._missed > self.max_missed:
                    del self._tracks[tid]

        # Spawn tracks for leftover detections.
        for di in unmatched:
            det = detections[di]
            tid = next(self._ids)
            self._tracks[tid] = Track(
                track_id=tid,
                cx=det.cx,
                cy=det.cy,
                first_seen=now,
                last_seen=now,
                distance_m=det.distance_m,
                looking=det.looking,
                looking_seconds=dt if det.looking else 0.0,
            )

        return [t for t in self._tracks.values() if t._missed == 0]

    def mean_dwell_seconds(self) -> float | None:
        """Mean dwell across currently-live tracks (None if no one is tracked)."""
        live = [t for t in self._tracks.values() if t._missed == 0]
        if not live:
            return None
        return sum(t.dwell_seconds for t in live) / len(live)

    def new_gaze_dwell_tracks(self) -> list[Track]:
        """Tracks that *just* crossed the sustained-gaze threshold (fires once).

        Lets the adapter emit a single ``gaze_dwell`` interaction per person
        rather than one every frame they keep looking.
        """
        out: list[Track] = []
        for t in self._tracks.values():
            if (
                not t._gaze_event_fired
                and t.looking_seconds >= self.gaze_dwell_seconds
            ):
                t._gaze_event_fired = True
                out.append(t)
        return out
