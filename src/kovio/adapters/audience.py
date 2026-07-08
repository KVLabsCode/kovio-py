"""Audience metrics engine — the V2 perception pipeline (LiDAR + depth fusion).

Turns raw MID-360 point clouds into the three body/proximity moments the
session dashboard reports, all keyed off a shared, session-scoped ``track_id``:

* ``passerby``       — a human-sized cluster entered the 3 m radius and
                       persisted ≥1 s (LiDAR primary; the honest top-of-funnel).
* ``dwell``          — a tracked cluster stayed within ~1.5 m for ≥3 s, tiered
                       ``paused`` (≥3 s) / ``engaged`` (≥6 s) / ``deep`` (≥12 s),
                       with the depth camera CONFIRMING the near body when it can.
* ``close_approach`` — the nearest depth point dropped below ~1 m with a
                       coincident LiDAR cluster on the same bearing (depth
                       primary; guards against a hand/bag at the lens).

Everything face-free by design: the camera is mounted low and sees legs, not
faces. No frame, no embedding, nothing re-identifying ever leaves the device —
moments are counts, distances and durations only, and track ids reset every
session.

De-duplication (what V1 lacked): each person is one ``track_id``. A track that
dies and is re-born nearby within the campaign's ``encounter_cap_seconds`` is
resurrected with its old id, so a double pass inside the cap window is still
ONE reach.

Layers, separately testable without hardware:

  ``extract_clusters``  pure-ish (numpy) — points -> human-sized 2D clusters
  ``BackgroundModel``   per-cell occupancy EMA — walls/furniture fade out
  ``AudienceTracker``   pure python — centroids -> tracks -> moments
  ``AudienceEngine``    thread-safe facade the adapters/streamer talk to

Frame convention matches ``lidar.py``: +x forward, +y left (metres); bearing
degrees clockwise from front (0 = ahead, +90 = right).
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("kovio.perception.audience")

# --- metric definitions (spec §Phase B) --------------------------------------
PASSERBY_RADIUS_M = 3.0
PASSERBY_MIN_AGE_S = 1.0
DWELL_RADIUS_M = 1.5
DWELL_EXIT_M = 1.8            # hysteresis so range jitter doesn't reset dwell
DWELL_TIERS = (("paused", 3.0), ("engaged", 6.0), ("deep", 12.0))
CLOSE_APPROACH_M = 1.0
CLOSE_EXIT_M = 1.25           # hysteresis on episode end
CLOSE_MIN_DURATION_S = 0.5
# The D435i's horizontal FOV is ~87°; a lidar cluster within this half-angle of
# straight ahead is "in the camera cone" for fusion checks.
CAMERA_HALF_FOV_DEG = 50.0
# A track must have travelled this far to produce moments — a pillar or parked
# cart never moves, so static clutter can't become a "passerby" even before the
# background model has learned it.
MIN_TRAVEL_M = 0.4
DEFAULT_ENCOUNTER_CAP_S = 300.0


@dataclass(frozen=True)
class Cluster:
    cx: float
    cy: float
    n_points: int
    extent_m: float

    @property
    def range_m(self) -> float:
        return math.hypot(self.cx, self.cy)

    @property
    def bearing_deg(self) -> float:
        # +y is left -> negate for clockwise bearing.
        return math.degrees(math.atan2(-self.cy, self.cx))


@dataclass(frozen=True)
class Moment:
    kind: str                  # 'passerby' | 'dwell' | 'close_approach'
    track_id: int
    t: float                   # unix seconds
    payload: dict
    moment_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def wire(self) -> dict:
        out = {
            "moment_id": self.moment_id,
            "kind": self.kind,
            "track_id": self.track_id,
            "t": round(self.t, 3),
        }
        out.update(self.payload)
        return out


# =============================================================================
# Cluster extraction — grid occupancy + connected components. numpy does the
# per-point heavy lifting (20k pts × 10 Hz); the component merge runs over a
# few hundred occupied cells in pure python.
# =============================================================================

def extract_clusters(
    points,
    *,
    radius_m: float = 4.5,
    # The MID-360 sits on the robot and sees the G1's own shoulders/arms as
    # near returns (observed ~0.27 m); anything this close can't be a separate
    # person's body centre, so it's self-geometry — excluded.
    min_range_m: float = 0.45,
    z_min: float = 0.15,
    z_max: float = 1.9,
    cell_m: float = 0.15,
    min_points: int = 5,
    min_extent_m: float = 0.08,
    max_extent_m: float = 1.2,
    background=None,
) -> list[Cluster]:
    """Points (Nx3 array-like) -> human-sized 2D clusters.

    ``background`` is an optional :class:`BackgroundModel`; when given it is
    updated with this frame's occupied cells and cells it currently calls
    static are dropped before clustering.
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        if background is not None:
            background.update(set())
        return []
    pts = pts.reshape(-1, pts.shape[-1])[:, :3]

    z = pts[:, 2]
    r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    keep = (
        (z >= z_min)
        & (z <= z_max)
        & (r2 <= radius_m * radius_m)
        & (r2 >= min_range_m * min_range_m)
    )
    xy = pts[keep, :2]
    if xy.shape[0] == 0:
        if background is not None:
            background.update(set())
        return []

    ij = np.floor(xy / cell_m).astype(np.int64)
    # Cells span ±radius/cell (≈±30); offset well into the positive range and
    # pack (i, j) into one int so np.unique can group them.
    OFF = 4096
    packed = (ij[:, 0] + OFF) * (2 * OFF) + (ij[:, 1] + OFF)
    codes, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    sums_x = np.bincount(inverse, weights=xy[:, 0])
    sums_y = np.bincount(inverse, weights=xy[:, 1])

    cell_ij: dict[tuple[int, int], int] = {}
    for k, code in enumerate(codes.tolist()):
        i = code // (2 * OFF) - OFF
        j = code % (2 * OFF) - OFF
        cell_ij[(int(i), int(j))] = k

    occupied = set(cell_ij.keys())
    if background is not None:
        static = background.update(occupied)
        occupied -= static

    # Connected components over occupied cells (8-neighbourhood).
    seen: set[tuple[int, int]] = set()
    clusters: list[Cluster] = []
    for start in occupied:
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            c = stack.pop()
            comp.append(c)
            ci, cj = c
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (ci + di, cj + dj)
                    if nb in occupied and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        n = sum(int(counts[cell_ij[c]]) for c in comp)
        if n < min_points:
            continue
        sx = sum(float(sums_x[cell_ij[c]]) for c in comp)
        sy = sum(float(sums_y[cell_ij[c]]) for c in comp)
        is_ = [c[0] for c in comp]
        js = [c[1] for c in comp]
        extent = max(max(is_) - min(is_) + 1, max(js) - min(js) + 1) * cell_m
        if extent < min_extent_m or extent > max_extent_m:
            continue
        clusters.append(Cluster(sx / n, sy / n, n, round(extent, 3)))
    clusters.sort(key=lambda c: c.range_m)
    return clusters


class BackgroundModel:
    """Per-cell occupancy EMA: cells occupied nearly all the time are static
    scenery (walls, furniture, the charging dock) and get subtracted before
    clustering. Time constant ~60 s at 10 Hz, so even a ``deep`` dweller (12 s)
    is nowhere near being absorbed into the background."""

    def __init__(self, alpha: float = 1.0 / 600.0, threshold: float = 0.7) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self._ema: dict[tuple[int, int], float] = {}

    def update(self, occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """Fold in one frame; return the cells currently considered static."""
        a = self.alpha
        ema = self._ema
        for c in occupied:
            ema[c] = ema.get(c, 0.0) * (1 - a) + a
        # Decay cells not seen this frame; drop the long-empty to bound memory.
        dead = []
        for c, v in ema.items():
            if c not in occupied:
                v *= 1 - a
                if v < 0.01:
                    dead.append(c)
                else:
                    ema[c] = v
        for c in dead:
            del ema[c]
        thr = self.threshold
        return {c for c in occupied if ema.get(c, 0.0) >= thr}

    def reset(self) -> None:
        self._ema.clear()


# =============================================================================
# Tracking + moments — pure python over cluster centroids.
# =============================================================================

@dataclass
class Track:
    track_id: int
    born_t: float
    x: float
    y: float
    last_t: float
    min_range_m: float
    travel_m: float = 0.0
    counted_passerby: bool = False
    # Continuous time within the dwell radius (with exit hysteresis).
    dwell_since: float | None = None
    dwell_s: float = 0.0
    dwell_tier_idx: int = -1   # index into DWELL_TIERS already emitted
    misses: int = 0

    @property
    def range_m(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def bearing_deg(self) -> float:
        return math.degrees(math.atan2(-self.y, self.x))

    @property
    def eligible(self) -> bool:
        return self.travel_m >= MIN_TRAVEL_M


@dataclass
class _DeadTrack:
    track_id: int
    x: float
    y: float
    died_t: float
    counted_passerby: bool
    dwell_s: float
    dwell_tier_idx: int


class AudienceTracker:
    """Nearest-centroid tracker + the passerby/dwell state machines.

    ``update(clusters, now)`` returns the moments this frame produced. Close
    approach lives in :class:`AudienceEngine` because it needs the depth
    camera; the tracker only supplies candidate tracks.
    """

    def __init__(
        self,
        *,
        match_gate_m: float = 0.9,
        miss_timeout_s: float = 0.8,
        reuse_gate_m: float = 2.5,
        encounter_cap_s: float = DEFAULT_ENCOUNTER_CAP_S,
        camera_confirm=None,   # callable(track) -> bool | None
    ) -> None:
        self.match_gate_m = match_gate_m
        self.miss_timeout_s = miss_timeout_s
        self.reuse_gate_m = reuse_gate_m
        self.encounter_cap_s = encounter_cap_s
        self._camera_confirm = camera_confirm
        self.tracks: list[Track] = []
        self._dead: list[_DeadTrack] = []
        self._next_id = 1

    # -- public ------------------------------------------------------------ #

    def reset(self) -> None:
        """New session: fresh ids, no cross-session identity (privacy)."""
        self.tracks = []
        self._dead = []
        self._next_id = 1

    def update(self, clusters: list[Cluster], now: float) -> list[Moment]:
        moments: list[Moment] = []

        # 1) Greedy nearest matching, gated.
        unmatched = list(range(len(clusters)))
        matched: dict[int, int] = {}  # track index -> cluster index
        for ti, tr in enumerate(self.tracks):
            best, best_d = None, self.match_gate_m
            for ci in unmatched:
                d = math.hypot(clusters[ci].cx - tr.x, clusters[ci].cy - tr.y)
                if d <= best_d:
                    best, best_d = ci, d
            if best is not None:
                matched[ti] = best
                unmatched.remove(best)

        # 2) Update matched tracks.
        for ti, ci in matched.items():
            tr = self.tracks[ti]
            c = clusters[ci]
            tr.travel_m += math.hypot(c.cx - tr.x, c.cy - tr.y)
            tr.x, tr.y = c.cx, c.cy
            tr.last_t = now
            tr.misses = 0
            tr.min_range_m = min(tr.min_range_m, c.range_m)

        # 3) Kill stale tracks (moved out of field / occlusion beyond timeout).
        survivors: list[Track] = []
        for ti, tr in enumerate(self.tracks):
            if ti in matched:
                survivors.append(tr)
            elif now - tr.last_t <= self.miss_timeout_s:
                tr.misses += 1
                survivors.append(tr)
            else:
                # Close an open dwell episode before the track dies.
                self._fold_dwell(tr, now)
                self._dead.append(
                    _DeadTrack(
                        tr.track_id, tr.x, tr.y, tr.last_t,
                        tr.counted_passerby, tr.dwell_s, tr.dwell_tier_idx,
                    )
                )
        self.tracks = survivors
        self._dead = [
            d for d in self._dead if now - d.died_t <= self.encounter_cap_s
        ]

        # 4) Births — resurrect from the reuse pool inside the encounter cap so
        # the same person re-entering is NOT a second reach.
        for ci in unmatched:
            c = clusters[ci]
            revived = None
            best_d = self.reuse_gate_m
            for d in self._dead:
                dist = math.hypot(c.cx - d.x, c.cy - d.y)
                if dist <= best_d:
                    revived, best_d = d, dist
            if revived is not None:
                self._dead.remove(revived)
                self.tracks.append(
                    Track(
                        track_id=revived.track_id, born_t=now,
                        x=c.cx, y=c.cy, last_t=now, min_range_m=c.range_m,
                        # Preserve what was already counted/emitted for this id.
                        counted_passerby=revived.counted_passerby,
                        dwell_s=revived.dwell_s,
                        dwell_tier_idx=revived.dwell_tier_idx,
                        # A resurrected person already proved they move.
                        travel_m=MIN_TRAVEL_M,
                    )
                )
            else:
                self.tracks.append(
                    Track(
                        track_id=self._next_id, born_t=now,
                        x=c.cx, y=c.cy, last_t=now, min_range_m=c.range_m,
                    )
                )
                self._next_id += 1

        # 5) Metric state machines.
        for tr in self.tracks:
            moments.extend(self._passerby(tr, now))
            moments.extend(self._dwell(tr, now))
        return moments

    # -- state machines ------------------------------------------------------ #

    def _passerby(self, tr: Track, now: float) -> list[Moment]:
        if (
            not tr.counted_passerby
            and tr.eligible
            and now - tr.born_t >= PASSERBY_MIN_AGE_S
            and tr.min_range_m <= PASSERBY_RADIUS_M
        ):
            tr.counted_passerby = True
            return [
                Moment(
                    "passerby", tr.track_id, now,
                    {
                        "first_seen": round(tr.born_t, 3),
                        "closest_m": round(tr.min_range_m, 2),
                        "lidar_confirmed": True,
                    },
                )
            ]
        return []

    def _dwell(self, tr: Track, now: float) -> list[Moment]:
        rng = tr.range_m
        if tr.dwell_since is None:
            if rng <= DWELL_RADIUS_M and tr.eligible:
                tr.dwell_since = now
            return []
        if rng > DWELL_EXIT_M:
            self._fold_dwell(tr, now)
            return []
        dwell = tr.dwell_s + (now - tr.dwell_since)
        out: list[Moment] = []
        for idx, (tier, threshold) in enumerate(DWELL_TIERS):
            if dwell >= threshold and tr.dwell_tier_idx < idx:
                tr.dwell_tier_idx = idx
                confirmed = None
                if self._camera_confirm is not None:
                    confirmed = self._camera_confirm(tr)
                out.append(
                    Moment(
                        "dwell", tr.track_id, now,
                        {
                            "dwell_s": round(dwell, 1),
                            "min_m": round(tr.min_range_m, 2),
                            "tier": tier,
                            "camera_confirmed": bool(confirmed),
                            "lidar_confirmed": True,
                        },
                    )
                )
        return out

    @staticmethod
    def _fold_dwell(tr: Track, now: float) -> None:
        if tr.dwell_since is not None:
            tr.dwell_s += now - tr.dwell_since
            tr.dwell_since = None
            # Leaving the zone re-arms nothing: tiers already emitted stay
            # emitted (dwell_tier_idx survives) so re-entry can only ADD time
            # toward the next tier, never re-fire an old one.


# =============================================================================
# Engine — thread-safe facade: lidar thread feeds points, the perception loop
# feeds camera context, the session streamer drains moments.
# =============================================================================

class AudienceEngine:
    """Owns the background model, tracker and close-approach fusion.

    ``ingest_points`` is called from the lidar poll thread at cloud rate;
    ``set_camera`` from the perception loop (~1 Hz); ``drain``/health from the
    session streamer. Moments accumulate only while ``arm()``-ed (an admin
    session is open) so idle operation stores nothing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._background = BackgroundModel()
        self._tracker = AudienceTracker(camera_confirm=self._camera_confirms)
        self._armed = False
        self._pending: list[Moment] = []
        self._clusters: list[Cluster] = []
        # Camera context: (wall time, nearest depth m or None, person ranges).
        self._cam_t = 0.0
        self._cam_near: float | None = None
        self._cam_people: tuple[float, ...] = ()
        # Close-approach episode state.
        self._close_track: Track | None = None
        self._close_started = 0.0
        self._close_min = float("inf")
        # Lidar liveness (EMA of inter-cloud gaps).
        self._last_cloud_t = 0.0
        self._hz = 0.0

    # -- lidar side ---------------------------------------------------------- #

    def ingest_points(self, points, now: float | None = None) -> list[Cluster]:
        """Feed one decoded cloud; returns this frame's clusters (for the
        CrowdReading the scene payload still carries)."""
        now = time.time() if now is None else now
        try:
            clusters = extract_clusters(points, background=self._background)
        except Exception:
            log.exception("audience: cluster extraction failed")
            return []
        with self._lock:
            if self._last_cloud_t:
                gap = max(now - self._last_cloud_t, 1e-3)
                inst = 1.0 / gap
                self._hz = inst if self._hz == 0 else 0.9 * self._hz + 0.1 * inst
            self._last_cloud_t = now
            self._clusters = clusters
            moments = self._tracker.update(clusters, now)
            moments.extend(self._close_approach(now))
            if self._armed and moments:
                # Stored as wire dicts so a failed upload can requeue verbatim;
                # moment_id keeps retries idempotent server-side.
                self._pending.extend(m.wire() for m in moments)
                del self._pending[:-500]  # bound RAM if the uplink is down
        return clusters

    # -- camera side ---------------------------------------------------------- #

    def set_camera(
        self, near_m: float | None, person_ranges=(), now: float | None = None
    ) -> None:
        """Perception loop reports the nearest valid depth in the camera cone
        and the ranges of camera-confirmed people. ``near_m=None`` = no depth
        signal this tick (camera dark)."""
        with self._lock:
            self._cam_t = time.time() if now is None else now
            self._cam_near = near_m
            self._cam_people = tuple(person_ranges)

    # -- session streamer side -------------------------------------------------- #

    def arm(self) -> None:
        """New session: reset identity + start queueing moments."""
        with self._lock:
            self._tracker.reset()
            self._pending.clear()
            self._close_track = None
            self._armed = True

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
            self._pending.clear()
            self._close_track = None

    def set_encounter_cap(self, seconds: float | None) -> None:
        with self._lock:
            self._tracker.encounter_cap_s = (
                float(seconds) if seconds else DEFAULT_ENCOUNTER_CAP_S
            )

    def drain(self, limit: int = 200) -> list[dict]:
        with self._lock:
            out, self._pending = self._pending[:limit], self._pending[limit:]
        return out

    def requeue(self, wire_moments: list[dict]) -> None:
        """Put drained moments back (upload failed); they retry next tick."""
        with self._lock:
            if self._armed:
                self._pending[:0] = wire_moments
                del self._pending[:-500]

    def health(self, now: float | None = None) -> dict:
        """Sensor-health snapshot for the dashboard's degraded states."""
        now = time.time() if now is None else now
        with self._lock:
            lidar_ok = bool(self._last_cloud_t) and now - self._last_cloud_t < 3.0
            depth_ok = self._cam_near is not None and now - self._cam_t < 5.0
            return {
                "lidar_ok": lidar_ok,
                "lidar_hz": round(self._hz, 1) if lidar_ok else 0.0,
                "depth_ok": depth_ok,
                "tracks": len(self._tracker.tracks),
            }

    def latest_people(self) -> tuple[tuple[float, float], ...]:
        """(range_m, bearing_deg) per current cluster, nearest first — feeds
        the scene payload / radar exactly like CrowdReading.people did."""
        with self._lock:
            return tuple(
                (round(c.range_m, 2), round(c.bearing_deg, 1)) for c in self._clusters
            )

    # -- fusion internals (called under lock) ---------------------------------- #

    def _camera_fresh(self, now: float) -> bool:
        return now - self._cam_t < 5.0

    def _camera_confirms(self, tr: Track) -> bool:
        """Does the depth camera corroborate this near lidar track? Only
        answerable for tracks inside the camera cone with fresh depth."""
        now = time.time()
        if not self._camera_fresh(now) or abs(tr.bearing_deg) > CAMERA_HALF_FOV_DEG:
            return False
        rng = tr.range_m
        for pr in self._cam_people:
            if pr is not None and abs(pr - rng) <= 0.6:
                return True
        # No detected person, but a solid near-field depth return at the same
        # range still confirms a body-sized obstacle (YOLO often can't box a
        # torso 60 cm from a chest-height lens).
        return self._cam_near is not None and abs(self._cam_near - rng) <= 0.4

    def _close_approach(self, now: float) -> list[Moment]:
        """Depth-primary close approach, guarded by lidar coincidence."""
        near = self._cam_near if self._camera_fresh(now) else None
        active = self._close_track is not None

        if not active:
            if near is None or near >= CLOSE_APPROACH_M:
                return []
            # Need a lidar cluster in the camera cone, near enough to be the
            # same body — otherwise it's a hand/bag at the lens; ignore.
            cand = None
            for tr in self._tracker.tracks:
                if abs(tr.bearing_deg) <= CAMERA_HALF_FOV_DEG and tr.range_m <= 1.8:
                    if cand is None or tr.range_m < cand.range_m:
                        cand = tr
            if cand is None:
                return []
            self._close_track = cand
            self._close_started = now
            self._close_min = near
            return []

        # Episode running: track the minimum, end on hysteresis or camera loss.
        if near is not None and near < self._close_min:
            self._close_min = near
        ended = near is None or near > CLOSE_EXIT_M
        if not ended:
            return []
        tr, started, min_m = self._close_track, self._close_started, self._close_min
        self._close_track = None
        duration = now - started
        if duration < CLOSE_MIN_DURATION_S:
            return []
        return [
            Moment(
                "close_approach", tr.track_id, now,
                {
                    "min_m": round(min_m, 2),
                    "duration_s": round(duration, 1),
                    "camera_confirmed": True,   # depth-primary by definition
                    "lidar_confirmed": True,    # required to open the episode
                },
            )
        ]
