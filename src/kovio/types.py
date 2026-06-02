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


@dataclass(frozen=True)
class SceneState:
    """A perceptual snapshot — counts and attention, never identity."""

    person_count: int
    attended_count: int          # gaze-oriented toward the screen
    mean_distance_m: float | None
    timestamp: float = field(default_factory=time.time)


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
