"""Gesture classification from pose keypoints — pure geometry, no deps.

Takes COCO-17 skeletons (whatever pose model produced them) and decides whether
a person is waving, raising for a high-five, offering a handshake, or throwing a
fist-bump. All reasoning is scale-invariant (normalised by torso length) and
side-aware (left/right arm), so it works at any distance and for either hand.

Honesty about the sensor: a single RGB frame robustly supports the *vertical*
gestures (wave, high-five, raised hand). Handshake and fist-bump are forward
motions toward the robot — ambiguous in 2D — so they additionally consume a
``forward`` hint the depth camera supplies (wrist measurably closer than the
torso). Without depth those two degrade to low confidence rather than firing
false positives. Identity never enters: input is geometry, keyed by ephemeral
track id only to give the wave detector temporal memory.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# COCO-17 keypoint indices.
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# Each side: (shoulder, elbow, wrist)
_ARMS = {
    "left": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right": (R_SHOULDER, R_ELBOW, R_WRIST),
}


@dataclass(frozen=True)
class GestureHit:
    kind: str          # InteractionKind.* string
    confidence: float
    side: str | None = None


def _pt(kp, i):
    """(x, y, conf) for keypoint i, tolerant of short/None inputs."""
    if kp is None or i >= len(kp) or kp[i] is None:
        return (0.0, 0.0, 0.0)
    x, y, *rest = kp[i]
    c = rest[0] if rest else 1.0
    return (float(x), float(y), float(c))


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class GestureClassifier:
    """Stateful per-track gesture recogniser.

    State is only the short wrist-motion history needed to tell a *wave* (it
    oscillates) from a static *high-five* (it doesn't). Everything else is
    decided from the current frame.

    Args:
        kp_conf_min: ignore a joint below this keypoint confidence.
        wave_window_s: look-back window for wave oscillation.
        wave_min_reversals: direction changes needed in-window to call a wave.
        raise_margin: wrist must clear the shoulder by this fraction of torso
            to count as "raised"; clear the nose to count as "overhead".
    """

    def __init__(
        self,
        kp_conf_min: float = 0.3,
        wave_window_s: float = 1.6,
        wave_min_reversals: int = 2,
        raise_margin: float = 0.15,
    ) -> None:
        self.kp_conf_min = kp_conf_min
        self.wave_window_s = wave_window_s
        self.wave_min_reversals = wave_min_reversals
        self.raise_margin = raise_margin
        # track_id -> side -> deque[(t, normalized_wrist_x)]
        self._hist: dict[int, dict[str, deque]] = {}

    def forget(self, track_id: int) -> None:
        self._hist.pop(track_id, None)

    def classify(
        self,
        track_id: int,
        keypoints,
        now: float,
        forward_left: bool = False,
        forward_right: bool = False,
    ) -> list[GestureHit]:
        """Return the gestures this person shows this frame (possibly empty)."""
        sh_l, sh_r = _pt(keypoints, L_SHOULDER), _pt(keypoints, R_SHOULDER)
        hip_l, hip_r = _pt(keypoints, L_HIP), _pt(keypoints, R_HIP)
        nose = _pt(keypoints, NOSE)
        if sh_l[2] < self.kp_conf_min or sh_r[2] < self.kp_conf_min:
            return []  # no reliable shoulders -> no scale, bail

        sh_c = _mid(sh_l, sh_r)
        # Torso length for scale; fall back to shoulder width if hips are weak.
        if hip_l[2] >= self.kp_conf_min and hip_r[2] >= self.kp_conf_min:
            torso = _dist(sh_c, _mid(hip_l, hip_r))
        else:
            torso = _dist(sh_l, sh_r) * 1.5
        if torso <= 1e-6:
            return []
        shoulder_y = sh_c[1]
        nose_y = nose[1] if nose[2] >= self.kp_conf_min else shoulder_y - 0.4 * torso

        forward = {"left": forward_left, "right": forward_right}
        hits: list[GestureHit] = []

        for side, (sh_i, el_i, wr_i) in _ARMS.items():
            shoulder = _pt(keypoints, sh_i)
            wrist = _pt(keypoints, wr_i)
            elbow = _pt(keypoints, el_i)
            if shoulder[2] < self.kp_conf_min or wrist[2] < self.kp_conf_min:
                continue

            # NOTE: image y grows downward, so "above" means a *smaller* y.
            raised = wrist[1] < shoulder_y - self.raise_margin * torso
            overhead = wrist[1] < nose_y
            # Mid-torso height band, used by forward gestures.
            mid_height = shoulder_y < wrist[1] < shoulder_y + 1.2 * torso

            # Record wrist horizontal position (normalised) for wave detection.
            nx = (wrist[0] - sh_c[0]) / torso
            self._push(track_id, side, now, nx)

            conf = min(shoulder[2], wrist[2])

            if overhead:
                hits.append(GestureHit("high_five", round(0.5 + 0.5 * conf, 3), side))
            elif raised and self._is_waving(track_id, side, now):
                hits.append(GestureHit("wave", round(0.5 + 0.5 * conf, 3), side))

            # Forward gestures need the depth hint; band separates the two.
            if forward[side] and mid_height and not raised:
                chest_band = wrist[1] < shoulder_y + 0.5 * torso
                kind = "fist_bump" if chest_band else "handshake"
                ec = elbow[2] if elbow[2] > 0 else conf
                hits.append(GestureHit(kind, round(0.4 + 0.4 * min(conf, ec), 3), side))

        return self._dedupe(hits)

    # --- wave temporal logic ---

    def _push(self, track_id: int, side: str, now: float, nx: float) -> None:
        per = self._hist.setdefault(track_id, {})
        dq = per.setdefault(side, deque())
        dq.append((now, nx))
        cutoff = now - self.wave_window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _is_waving(self, track_id: int, side: str, now: float) -> bool:
        dq = self._hist.get(track_id, {}).get(side)
        if not dq or len(dq) < 3:
            return False
        # Count direction reversals with non-trivial amplitude.
        reversals = 0
        prev_dir = 0
        last_ext = dq[0][1]
        for _t, x in list(dq)[1:]:
            if abs(x - last_ext) < 0.12:  # ignore jitter (<12% of torso)
                continue
            d = 1 if x > last_ext else -1
            if prev_dir and d != prev_dir:
                reversals += 1
            prev_dir = d
            last_ext = x
        return reversals >= self.wave_min_reversals

    @staticmethod
    def _dedupe(hits: list[GestureHit]) -> list[GestureHit]:
        """Keep the highest-confidence hit per kind."""
        best: dict[str, GestureHit] = {}
        for h in hits:
            cur = best.get(h.kind)
            if cur is None or h.confidence > cur.confidence:
                best[h.kind] = h
        return list(best.values())
