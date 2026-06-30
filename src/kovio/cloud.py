"""Cloud sync for the kovio SDK.

  - CloudCampaignStore: drop-in replacement for the local CampaignStore that
    fetches campaigns from the Kovio cloud API on a TTL, with a local SQLite
    cache as fallback for offline operation.

  - CloudEventSink: a background uploader that drains the agent's local event
    log and posts batches to the cloud /sdk/v1/events/batch endpoint.

Both pieces use the same API key, set via KOVIO_API_KEY env var or passed at
construction time. The API key carries the fleet scope server-side; the SDK
doesn't need to know its own fleet id.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from urllib import error, request

from .campaigns.models import Campaign

log = logging.getLogger("kovio.cloud")

_DEFAULT_TIMEOUT = 8.0  # seconds


# ---------- helpers ----------

def _post_json(url: str, body: dict, headers: dict, timeout: float = _DEFAULT_TIMEOUT) -> tuple[int, dict | None]:
    data = json.dumps(body).encode()
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
            return resp.status, payload
    except error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        log.warning("cloud.post.http_error", extra={"url": url, "status": e.code, "body": body_text[:200]})
        return e.code, None
    except (error.URLError, TimeoutError, OSError) as e:
        log.warning("cloud.post.network_error", extra={"url": url, "error": str(e)})
        return 0, None


def _get_json(url: str, headers: dict, timeout: float = _DEFAULT_TIMEOUT) -> tuple[int, dict | None]:
    req = request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
            return resp.status, payload
    except error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        log.warning("cloud.get.http_error", extra={"url": url, "status": e.code, "body": body_text[:200]})
        return e.code, None
    except (error.URLError, TimeoutError, OSError) as e:
        log.warning("cloud.get.network_error", extra={"url": url, "error": str(e)})
        return 0, None


# ---------- CloudCampaignStore ----------

class CloudCampaignStore:
    """Pulls campaigns from the Kovio cloud API; caches locally for offline use.

    Drop-in replacement for kovio.campaigns.CampaignStore. Exposes the same
    `active_campaigns()` and `get(campaign_id)` interface. The selector doesn't
    know or care whether the store is local or cloud-backed.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        db_path: str | Path = "kovio.db",
        ttl_seconds: int = 300,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.timeout = timeout

        self._campaigns: list[Campaign] = []
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()
        self._init_db()
        self.reload()

    # ---- public interface (compat with CampaignStore) ----

    def active_campaigns(self) -> list[Campaign]:
        # Lazy refresh on read if past TTL — non-blocking; serve stale on failure.
        if time.time() - self._fetched_at > self.ttl_seconds:
            self._try_refresh()
        with self._lock:
            return [c for c in self._campaigns if c.enabled]

    def get(self, campaign_id: str) -> Campaign | None:
        with self._lock:
            for c in self._campaigns:
                if c.campaign_id == campaign_id:
                    return c
            return None

    def reload(self) -> None:
        """Force a re-fetch from cloud. Falls back to local cache on failure."""
        self._try_refresh(force=True)

    # ---- internals ----

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cloud_campaigns_cache "
            "(campaign_id TEXT PRIMARY KEY, raw_json TEXT NOT NULL, fetched_at REAL NOT NULL)"
        )
        conn.commit()
        conn.close()

    def _try_refresh(self, force: bool = False) -> None:
        status, payload = _get_json(
            f"{self.api_url}/sdk/v1/campaigns",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if status == 200 and payload:
            raw_campaigns = payload.get("campaigns", [])
            campaigns = [Campaign.from_dict(c) for c in raw_campaigns]
            with self._lock:
                self._campaigns = campaigns
                self._fetched_at = time.time()
            self._persist_to_cache(raw_campaigns)
            log.info("cloud.campaigns.refreshed", extra={"count": len(campaigns)})
            return

        # Failure path: try local cache.
        if force or not self._campaigns:
            cached = self._load_from_cache()
            if cached:
                with self._lock:
                    self._campaigns = cached
                log.warning("cloud.campaigns.using_cache", extra={"count": len(cached)})

    def _persist_to_cache(self, raw_campaigns: list[dict]) -> None:
        conn = sqlite3.connect(str(self.db_path))
        now = time.time()
        for c in raw_campaigns:
            conn.execute(
                "INSERT OR REPLACE INTO cloud_campaigns_cache VALUES (?, ?, ?)",
                (c["campaign_id"], json.dumps(c), now),
            )
        conn.commit()
        conn.close()

    def _load_from_cache(self) -> list[Campaign]:
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT raw_json FROM cloud_campaigns_cache").fetchall()
        conn.close()
        return [Campaign.from_dict(json.loads(r[0])) for r in rows]


# ---------- CloudEventSink ----------

class CloudEventSink:
    """Background uploader for events. Drains the agent's local event log,
    batches events, POSTs to the cloud, marks uploaded.

    Designed to be safe across crashes: events stay in local SQLite until
    confirmed-accepted by the cloud, then marked. Idempotent on event_id
    (the server de-dupes on conflict).
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        db_path: str | Path = "kovio.db",
        robot_id: str = "",
        flush_interval_seconds: float = 30.0,
        batch_size: int = 100,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.db_path = Path(db_path)
        self.robot_id = robot_id
        self.flush_interval = flush_interval_seconds
        self.batch_size = batch_size
        self.timeout = timeout

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ensure_uploaded_column()

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="kovio-cloud-sink")
        self._thread.start()
        log.info("cloud.sink.started", extra={"api_url": self.api_url})

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def flush_once(self) -> tuple[int, int]:
        """Force one flush pass. Returns (sent, total_pending). Useful in tests."""
        return self._drain_once()

    # ---- internals ----

    def _ensure_uploaded_column(self) -> None:
        """Add cloud_uploaded column to events table if needed. Tolerant of the
        table not existing yet — the agent's _EventDB creates it lazily."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
            if cols and "cloud_uploaded" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN cloud_uploaded INTEGER NOT NULL DEFAULT 0")
                conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            # Table doesn't exist yet — will be created by the agent. Retry on first drain.
            pass

    def _heartbeat(self) -> None:
        """Register/refresh this robot in the cloud. The server auto-registers a
        robot on its first heartbeat (keyed by the fleet API key + robot_id), so
        a freshly-flashed robot needs no manual provisioning: boot the agent and
        it appears in its fleet. Sent before each drain so the robot row exists
        by the time its events arrive (otherwise they'd land unattributed)."""
        if not self.robot_id:
            return
        status, payload = _post_json(
            f"{self.api_url}/sdk/v1/heartbeat",
            {"robot_id": self.robot_id, "status": "online"},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if status == 200 and payload:
            if payload.get("registered"):
                log.info("cloud.robot.registered", extra={"robot_id": self.robot_id})
        else:
            log.warning("cloud.heartbeat.failed", extra={"status": status})

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._heartbeat()  # auto-register + liveness, before draining events
                self._drain_once()
            except Exception:
                log.exception("cloud.sink.drain_failed")
            self._stop.wait(self.flush_interval)

    def _drain_once(self) -> tuple[int, int]:
        # Make sure the table exists and has the cloud_uploaded column.
        self._ensure_uploaded_column()

        # Pull up to batch_size pending events from the local log.
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT event_id, timestamp, event_type, robot_id, payload_json "
                "FROM events WHERE cloud_uploaded = 0 ORDER BY timestamp ASC LIMIT ?",
                (self.batch_size,),
            ).fetchall()
        except sqlite3.OperationalError:
            # events table not created yet (agent hasn't started writing)
            conn.close()
            return 0, 0
        finally:
            conn.close()

        if not rows:
            return 0, 0

        events = [
            {
                "event_id": str(r[0]),
                "timestamp": float(r[1]),
                "event_type": str(r[2]),
                "robot_id": str(r[3]) or self.robot_id,
                "payload": json.loads(r[4]),
            }
            for r in rows
        ]

        status, payload = _post_json(
            f"{self.api_url}/sdk/v1/events/batch",
            {"events": events},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

        if status == 200 and payload:
            event_ids = [str(r[0]) for r in rows]
            conn = sqlite3.connect(str(self.db_path))
            conn.executemany(
                "UPDATE events SET cloud_uploaded = 1 WHERE event_id = ?",
                [(eid,) for eid in event_ids],
            )
            conn.commit()
            conn.close()
            log.info(
                "cloud.sink.drained",
                extra={"accepted": payload.get("accepted"), "duplicates": payload.get("duplicates")},
            )
            return len(event_ids), len(event_ids)

        log.warning("cloud.sink.drain_failed", extra={"status": status, "pending": len(events)})
        return 0, len(events)


__all__ = ["CloudCampaignStore", "CloudEventSink"]
