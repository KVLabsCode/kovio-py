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
        store: "CampaignStore | None" = None,  # noqa: F821 (forward ref)
        sink: "CloudEventSink | None" = None,  # noqa: F821 (forward ref)
        session_streamer: "SessionStreamer | None" = None,  # noqa: F821 (forward ref)
    ) -> None:
        self.robot_id = robot_id
        self.screen = screen or ScreenAdapter.logger()
        self.perception = perception or PerceptionAdapter.stub()
        # A campaign store (local or cloud) becomes a RuleBasedSelector unless an
        # explicit selector was supplied. With neither, we fall back to the fixed
        # creative_url — the local default-creative path stays intact.
        if selector is None and store is not None:
            from .campaigns.selector import RuleBasedSelector
            selector = RuleBasedSelector(store)
        self.selector = selector
        self.creative_url = creative_url or _default_creative_url()
        self.api_key = api_key
        self._sink = sink
        self._session_streamer = session_streamer

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
    def autodetect(cls, robot_id: str | None = None, **kwargs) -> "KovioAgent":
        """Construct an agent using platform-detected adapters and env-based cloud config.

        Perception and screen adapters are chosen from the running platform
        (overridable via KOVIO_PERCEPTION / KOVIO_SCREEN env vars, or by
        passing explicit `perception=` / `screen=` kwargs).

        When KOVIO_API_URL and KOVIO_API_KEY are both set, the agent is wired to
        a CloudCampaignStore (real campaigns) and a CloudEventSink (event upload).
        With them unset it falls back to the local default-creative path — no
        cloud calls, identical to local-only behavior.
        """
        from .adapters.perception import make_perception_adapter
        from .adapters.screen import make_screen_adapter
        from .config import load_cloud_config

        config = load_cloud_config()
        effective_robot_id = robot_id or config.robot_id
        db_path = kwargs.get("db_path", "kovio.db")

        # Build the cloud-backed store and sink ONLY if config is complete and the
        # caller didn't pass their own. Construction is best-effort: a CloudCampaignStore
        # that can't reach the cloud logs a warning and serves its local cache.
        store = kwargs.pop("store", None)
        sink = kwargs.pop("sink", None)

        if config.is_configured and store is None:
            from .cloud import CloudCampaignStore
            store = CloudCampaignStore(
                api_url=config.api_url,
                api_key=config.api_key,
                db_path=db_path,
                ttl_seconds=config.campaign_ttl_seconds,
                timeout=config.api_timeout_seconds,
            )

        if config.is_configured and sink is None:
            from .cloud import CloudEventSink
            sink = CloudEventSink(
                api_url=config.api_url,
                api_key=config.api_key,
                db_path=db_path,
                robot_id=effective_robot_id,
                timeout=config.api_timeout_seconds,
            )

        perception = kwargs.pop("perception", None) or make_perception_adapter()
        screen = kwargs.pop("screen", None) or make_screen_adapter()

        # Text-to-speech adapter (kovio_tts binary when KOVIO_TTS_BIN is set,
        # else a no-op logger). Best-effort: a misconfigured speaker must never
        # stop the agent from coming up.
        speaker = kwargs.pop("speaker", None)
        if speaker is None:
            from .adapters.audio import make_audio_adapter
            try:
                speaker = make_audio_adapter()
            except Exception:
                log.exception("audio adapter construction failed; speech disabled")
                speaker = None

        # Admin live view + dashboard TTS both ride the /session/v1/current
        # poll, so we run the streamer whenever cloud-configured. Idle cost is
        # one 5s poll; frames leave the robot only while a session is open, and
        # only when the adapter can hand over frames (rich/RealSense) — a
        # camera-only robot still polls so it can receive speak commands.
        session_streamer = kwargs.pop("session_streamer", None)
        frame_source = getattr(perception, "latest_frame_bgr", None)
        if config.is_configured and session_streamer is None:
            from .session_stream import SessionStreamer
            session_streamer = SessionStreamer(
                api_url=config.api_url,
                api_key=config.api_key,
                robot_id=effective_robot_id,
                frame_source=frame_source,
                timeout=config.api_timeout_seconds,
                # V2 audience moments ride the same session poll (created by the
                # rich adapter in start(); None on camera-only adapters is fine —
                # the streamer then only relays frames).
                audience_engine=getattr(perception, "audience_engine", None),
                speaker=speaker,
            )

        return cls(
            robot_id=effective_robot_id,
            perception=perception,
            screen=screen,
            store=store,
            sink=sink,
            session_streamer=session_streamer,
            **kwargs,
        )

    # ---------- public API ----------

    def start(self) -> None:
        from . import __version__  # lazy: avoids a circular import at module load
        log.info("starting agent (robot_id=%s, selector=%s)",
                 self.robot_id, type(self.selector).__name__ if self.selector else "none")
        self._emit("agent_started", {"version": __version__})
        if self._sink is not None:
            self._sink.start()
        if self._session_streamer is not None:
            self._session_streamer.start()
        self.perception.start(self._handle_scene)

    def stop(self) -> None:
        log.info("stopping agent")
        self.perception.stop()
        if self._session_streamer is not None:
            self._session_streamer.stop()
        if self._sink is not None:
            self._sink.stop()
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
            # Scalar snapshot (counts, attention, dwell, crowd) — the scene_observed
            # event the cloud correlates with each impression. scalar_payload()
            # omits None fields so a basic adapter still emits the original three.
            self._emit("scene_observed", scene.scalar_payload())
            # Discrete interactions travel as their own events so the cloud can
            # count them into the engagement funnel without parsing scene blobs.
            for ix in scene.interactions:
                self._emit(
                    "interaction_observed",
                    {
                        "kind": ix.kind,
                        "confidence": ix.confidence,
                        "track_id": ix.track_id,
                        "distance_m": ix.distance_m,
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
