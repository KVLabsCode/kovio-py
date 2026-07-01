"""Core data types for the Kovio SDK.

Treat the AdEvent schema as APPEND-ONLY. Adding fields is free; renaming
or removing them is a coordinated multi-quarter migration.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskState(Enum):
    """What the robot is currently doing. Drives the task gate."""

    IDLE = "idle"
    NAVIGATING = "navigating"
    DELIVERING = "delivering"
    CUSTOMER_HANDOFF = "customer_handoff"
    LOW_BATTERY = "low_battery"
    MANUAL_CONTROL = "manual_control"
    ERROR = "error"

    @property
    def is_busy(self) -> bool:
        return self != TaskState.IDLE


class InteractionKind:
    """Discrete interaction events derived on-device. Strings, never identity.

    Physical gestures come from pose estimation; PHONE_OUT from object
    detection; GAZE_DWELL from sustained head-on attention. APPEND-ONLY —
    new kinds are free to add; downstream treats unknown kinds as generic.
    """

    HANDSHAKE = "handshake"
    WAVE = "wave"
    HIGH_FIVE = "high_five"
    FIST_BUMP = "fist_bump"
    PHONE_OUT = "phone_out"
    GAZE_DWELL = "gaze_dwell"      # a tracked person crossed the sustained-gaze threshold

    ALL = (HANDSHAKE, WAVE, HIGH_FIVE, FIST_BUMP, PHONE_OUT, GAZE_DWELL)
    PHYSICAL = (HANDSHAKE, WAVE, HIGH_FIVE, FIST_BUMP)


@dataclass(frozen=True)
class Interaction:
    """One discrete interaction this tick. Counts and kind, never identity."""

    kind: str
    confidence: float = 1.0
    track_id: int | None = None
    distance_m: float | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SceneState:
    """A perceptual snapshot — counts and attention, never identity.

    The first three fields are the original v0 contract (depth-camera people +
    attention). Everything below is additive: lidar crowd/proximity, gaze/dwell
    from the tracker, and a per-tick stream of discrete ``interactions``. All
    new fields default so any adapter that only fills the originals still works.
    """

    person_count: int
    attended_count: int          # gaze-oriented toward the screen
    mean_distance_m: float | None

    # --- lidar: wide-FOV crowd & proximity (None when no lidar) ---
    people_nearby: int | None = None        # bodies within the lidar radius
    crowd_density: float | None = None       # people per m^2 in the radius
    nearest_distance_m: float | None = None  # closest body (lidar; sub-camera-FOV)
    approach_bearing_deg: float | None = None  # bearing of nearest body, 0=front, +right
    # per-person polar blips (range_m, bearing_deg) for the live radar
    lidar_people: tuple[tuple[float, float], ...] | None = None
    # unique bodies that ENTERED the lidar field this tick (cumulative "passed by")
    lidar_passed: int | None = None

    # --- depth/RGB: attention & dwell (None when not computed) ---
    looked_count: int | None = None          # people whose gaze is on the screen this tick
    mean_dwell_s: float | None = None         # mean dwell of currently-tracked people

    # --- discrete interaction events observed this tick ---
    interactions: tuple[Interaction, ...] = ()

    timestamp: float = field(default_factory=time.time)

    def scalar_payload(self) -> dict:
        """The scene's scalar metrics as a JSON-ready dict (omits Nones).

        This is what the agent ships in a ``scene_observed`` event payload.
        Interactions travel separately as ``interaction_observed`` events.
        """
        fields = {
            "person_count": self.person_count,
            "attended_count": self.attended_count,
            "mean_distance_m": self.mean_distance_m,
            "people_nearby": self.people_nearby,
            "crowd_density": self.crowd_density,
            "nearest_distance_m": self.nearest_distance_m,
            "approach_bearing_deg": self.approach_bearing_deg,
            "looked_count": self.looked_count,
            "mean_dwell_s": self.mean_dwell_s,
            # JSON-ready blips: [[range_m, bearing_deg], ...] for the live radar
            "lidar_people": (
                [[r, b] for r, b in self.lidar_people]
                if self.lidar_people is not None
                else None
            ),
            "lidar_passed": self.lidar_passed,
        }
        return {k: v for k, v in fields.items() if v is not None}


@dataclass(frozen=True)
class GateDecision:
    """Whether the task gate allows an ad right now, and why."""

    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> "GateDecision":
        return cls(True, None)

    @classmethod
    def suppress(cls, reason: str) -> "GateDecision":
        return cls(False, reason)


@dataclass(frozen=True)
class AdEvent:
    """A single event from the SDK. APPEND-ONLY schema."""

    event_type: str
    payload: dict
    robot_id: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
