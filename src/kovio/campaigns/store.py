"""CampaignStore — single source of truth for campaigns on the robot.

Reads from a JSON file (human-editable), mirrors to SQLite so the dashboard
can query without parsing JSON each time. Thread-safe reads. Reload by
calling .reload() — works at runtime; no restart required.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .models import Campaign


class CampaignStore:
    """JSON source of truth; SQLite mirror for query and dashboard access."""

    def __init__(self, json_path: str | Path, db_path: str | Path = "kovio.db"):
        self.json_path = Path(json_path)
        self.db_path = Path(db_path)
        self._campaigns: list[Campaign] = []
        self._lock = threading.Lock()
        self._init_db()
        self.reload()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id           TEXT PRIMARY KEY,
                name                  TEXT NOT NULL,
                advertiser            TEXT,
                creative_path         TEXT NOT NULL,
                targeting_json        TEXT NOT NULL,
                priority              INTEGER DEFAULT 0,
                encounter_cap_seconds INTEGER DEFAULT 300,
                enabled               INTEGER DEFAULT 1,
                updated_at            REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def reload(self) -> None:
        """Re-read the JSON file and refresh both the in-memory list and SQLite."""
        with self._lock:
            if not self.json_path.exists():
                self._campaigns = []
                return
            raw = json.loads(self.json_path.read_text())
            campaigns = [Campaign.from_dict(d) for d in raw]
            self._campaigns = campaigns

            conn = sqlite3.connect(str(self.db_path))
            for c in campaigns:
                targeting = json.dumps([asdict(r) for r in c.targeting])
                conn.execute(
                    "INSERT OR REPLACE INTO campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        c.campaign_id, c.name, c.advertiser, c.creative_path,
                        targeting, c.priority, c.encounter_cap_seconds,
                        int(c.enabled), time.time(),
                    ),
                )
            conn.commit()
            conn.close()

    def active_campaigns(self) -> list[Campaign]:
        with self._lock:
            return [c for c in self._campaigns if c.enabled]

    def get(self, campaign_id: str) -> Campaign | None:
        with self._lock:
            for c in self._campaigns:
                if c.campaign_id == campaign_id:
                    return c
            return None
