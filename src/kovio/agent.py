"""KovioAgent — the main object an OEM instantiates.

Owns the event loop, the task gate, the campaign selector (optional), and
the event log. Pluggable via ScreenAdapter and PerceptionAdapter; extensible
via .task_gate and .on() hooks; campaign-driven via .selector.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from .adapters.perception import PerceptionAdapter
from .adapters.screen import ScreenAdapter
from .types import AdEvent, GateDecision, SceneState, TaskState

log = logging.getLogger("kovio.agent")

# Task states the SDK ALWAYS suppresses ads in. Cannot be overridden.
_HARD_SUPPRESS: frozenset[TaskState] = frozenset(
    {
        TaskState.LOW_BATTERY,
        TaskState.MANUAL_CONTROL,
        TaskState.ERROR,
        TaskState.CUSTOMER_HANDOFF,
    }
)


def _default_creative_url() -> str:
    p = files("kovio").joinpath("default_creative.html")
    return f"file://{p}"


class KovioAgent:
    """Drop on your robot and call .start()."""

    def __init__(
        self,
        robot_id: str,
        screen: ScreenAdapter | None = None,
        perception: PerceptionAdapter | None = None,
        selector: "CampaignSelector | None" = None,  # noqa: F821 (forward ref)
        creative_url: str | None = None,
        db_path: str | Path = "kovio.db",
        api_key: str | None = None,
    ) -> None:
        self.robot_id = robot_id
        self.screen = screen or ScreenAdapter.logger()
        self.perception = perception or PerceptionAdapter.stub()
        self.selector = selector
        self.creative_url = creative_url or _default_creative_url()
        self.api_key = api_key

        self._task_state = TaskState.IDLE
        self._last_scene: SceneState | None = None
        self._displayed = False
        self._current_campaign_id: str | None = None
        self._tags: list[str] = []
        self._lock = threading.Lock()

        self._db = _EventDB(Path(db_path))

        self._custom_gate: Callable[[TaskState, SceneState | None], GateDecision] | None = None
        self._on_scene_callbacks: list[Callable[[SceneState], None]] = []

    # ---------- construction ----------

    @classmethod
    def autodetect(cls, robot_id: str, **kwargs) -> "KovioAgent":
        """Construct an agent using platform-detected adapters.

        Perception and screen adapters are chosen from the running platform
        (overridable via KOVIO_PERCEPTION / KOVIO_SCREEN env vars, or by
        passing explicit `perception=` / `screen=` kwargs).
        """
        from .adapters.perception import make_perception_adapter
        from .adapters.screen import make_screen_adapter

        perception = kwargs.pop("perception", None) or make_perception_adapter()
        screen = kwargs.pop("screen", None) or make_screen_adapter()
        return cls(robot_id=robot_id, perception=perception, screen=screen, **kwargs)

    # ---------- public API ----------

    def start(self) -> None:
        log.info("starting agent (robot_id=%s, selector=%s)",
                 self.robot_id, type(self.selector).__name__ if self.selector else "none")
        self._emit("agent_started", {"version": "0.0.3"})
        self.perception.start(self._handle_scene)

    def stop(self) -> None:
        log.info("stopping agent")
        self.perception.stop()
        self.screen.clear()
        self._emit("agent_stopped", {})

    def update_task_state(self, state: TaskState) -> None:
        with self._lock:
            if state != self._task_state:
                log.info("task_state: %s -> %s", self._task_state.value, state.value)
                self._task_state = state
                self._reevaluate()

    def push_scene_state(self, scene: SceneState) -> None:
        """Tier-3: bring your own perception."""
        self._handle_scene(scene)

    def set_tags(self, tags: list[str]) -> None:
        """OEM-pushed contextual tags (e.g., 'near_coffee_shop'). Used in targeting."""
        with self._lock:
            self._tags = list(tags)

    # ---------- hooks (Tier 2) ----------

    def task_gate(self, fn):
        self._custom_gate = fn
        return fn

    def on(self, event: str):
        if event != "scene_state":
            raise ValueError(f"unknown event: {event!r}")

        def decorator(fn):
            self._on_scene_callbacks.append(fn)
            return fn

        return decorator

    # ---------- internals ----------

    def _handle_scene(self, scene: SceneState) -> None:
        with self._lock:
            self._last_scene = scene
            for cb in self._on_scene_callbacks:
                try:
                    cb(scene)
                except Exception:
                    log.exception("scene callback raised")
            self._emit(
                "scene_observed",
                {
                    "person_count": scene.person_count,
                    "attended_count": scene.attended_count,
                    "mean_distance_m": scene.mean_distance_m,
                },
            )
            self._reevaluate()

    def _reevaluate(self) -> None:
        decision = self._evaluate_gate()
        if decision.allowed:
            self._maybe_play()
        elif self._displayed:
            self._suppress(decision.reason or "unspecified")

    def _evaluate_gate(self) -> GateDecision:
        if self._task_state in _HARD_SUPPRESS:
            return GateDecision.suppress(f"hard_rule:{self._task_state.value}")
        if self._custom_gate:
            try:
                return self._custom_gate(self._task_state, self._last_scene)
            except Exception:
                log.exception("custom gate raised; defaulting to suppress")
                return GateDecision.suppress("custom_gate_error")
        if self._task_state == TaskState.IDLE:
            return GateDecision.allow()
        return GateDecision.suppress(f"default:{self._task_state.value}")

    def _build_context(self):
        """Construct a DecisionContext from current state. Imported lazily."""
        from .campaigns.models import DecisionContext  # local import: subpackage optional
        return DecisionContext(
            robot_id=self.robot_id,
            scene=self._last_scene or SceneState(0, 0, None),
            task_state=self._task_state,
            timestamp=time.time(),
            tags=list(self._tags),
        )

    def _maybe_play(self) -> None:
        """Pick a creative via selector (if any) and play, or fall back to creative_url."""
        if self.selector is not None:
            ctx = self._build_context()
            campaign = self.selector.select(ctx)
            if campaign is None:
                # Nothing eligible — suppress (or stay suppressed).
                if self._displayed:
                    self._suppress("no_eligible_campaign")
                return
            # If we're already showing this campaign, no-op.
            if self._displayed and campaign.campaign_id == self._current_campaign_id:
                return
            self.screen.display(campaign.creative_path)
            self._displayed = True
            self._current_campaign_id = campaign.campaign_id
            self._emit("ad_played", {
                "campaign_id": campaign.campaign_id,
                "advertiser": campaign.advertiser,
                "creative_path": campaign.creative_path,
            })
            self.selector.record_play(campaign.campaign_id)
        else:
            # Fixed-URL fallback (Tier-1 without a selector).
            if self._displayed:
                return
            self.screen.display(self.creative_url)
            self._displayed = True
            self._emit("ad_played", {"creative_url": self.creative_url})

    def _suppress(self, reason: str) -> None:
        self.screen.clear()
        self._displayed = False
        self._current_campaign_id = None
        self._emit("ad_suppressed", {"reason": reason})

    def _emit(self, event_type: str, payload: dict) -> None:
        evt = AdEvent(event_type=event_type, payload=payload, robot_id=self.robot_id)
        self._db.write(evt)
        log.info("event: %s %s", evt.event_type, evt.payload)


class _EventDB:
    """Append-only local SQLite log of events. WAL so the dashboard can read."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id     TEXT PRIMARY KEY,
                timestamp    REAL NOT NULL,
                event_type   TEXT NOT NULL,
                robot_id     TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def write(self, evt: AdEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (event_id, timestamp, event_type, robot_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    evt.event_id, evt.timestamp, evt.event_type,
                    evt.robot_id, json.dumps(evt.payload),
                ),
            )
            self._conn.commit()
