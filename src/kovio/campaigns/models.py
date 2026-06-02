"""Campaign data models.

A Campaign is an ad: a creative + targeting rules + scheduling metadata.
A TargetingRule is a single predicate that must hold for the campaign to be
eligible. A DecisionContext is everything the selector knows about *right now*
when it picks what to play.

Rules are simple by design: a field name, an operator, and a value. AND
semantics across rules. New operators are cheap to add; ML scoring will live
in a separate Selector implementation when we get there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..types import SceneState, TaskState


@dataclass(frozen=True)
class TargetingRule:
    """One predicate. AND-combined with other rules on the same campaign."""

    field: str
    op: str
    value: Any

    def evaluate(self, ctx: "DecisionContext") -> bool:
        actual = ctx.get(self.field)
        if actual is None:
            return False
        op, v = self.op, self.value
        if op == "gte":
            return actual >= v
        if op == "lte":
            return actual <= v
        if op == "eq":
            return actual == v
        if op == "ne":
            return actual != v
        if op == "in":
            return actual in v
        if op == "between":
            lo, hi = v
            return lo <= actual <= hi
        if op == "has_tag":
            return v in actual if hasattr(actual, "__contains__") else False
        return False


@dataclass(frozen=True)
class Campaign:
    """An ad: creative + targeting + scheduling."""

    campaign_id: str
    name: str
    advertiser: str
    creative_path: str
    targeting: list[TargetingRule] = field(default_factory=list)
    priority: int = 0
    encounter_cap_seconds: int = 300   # don't replay within 5 min by default
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Campaign":
        rules = [TargetingRule(**r) for r in d.get("targeting", [])]
        return cls(
            campaign_id=d["campaign_id"],
            name=d["name"],
            advertiser=d.get("advertiser", ""),
            creative_path=d["creative_path"],
            targeting=rules,
            priority=d.get("priority", 0),
            encounter_cap_seconds=d.get("encounter_cap_seconds", 300),
            enabled=d.get("enabled", True),
        )

    def matches(self, ctx: "DecisionContext") -> bool:
        return self.enabled and all(r.evaluate(ctx) for r in self.targeting)


@dataclass
class DecisionContext:
    """Snapshot of the world the selector reasons over."""

    robot_id: str
    scene: SceneState
    task_state: TaskState
    timestamp: float
    tags: list[str] = field(default_factory=list)

    @property
    def hour_of_day(self) -> int:
        return datetime.fromtimestamp(self.timestamp).hour

    @property
    def day_of_week(self) -> int:
        # Monday = 0, Sunday = 6
        return datetime.fromtimestamp(self.timestamp).weekday()

    def get(self, name: str) -> Any:
        """Field lookup used by TargetingRule.evaluate."""
        if name == "person_count":
            return self.scene.person_count
        if name == "attended_count":
            return self.scene.attended_count
        if name == "mean_distance_m":
            return self.scene.mean_distance_m
        if name == "hour_of_day":
            return self.hour_of_day
        if name == "day_of_week":
            return self.day_of_week
        if name == "task_state":
            return self.task_state.value
        if name == "tags":
            return self.tags
        if name == "robot_id":
            return self.robot_id
        return None
