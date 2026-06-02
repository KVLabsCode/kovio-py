"""Kovio local dashboard.

A small FastAPI app reading from the SDK's SQLite event log. Run with:

    python -m kovio.dashboard.server --db kovio.db --port 8000

Then open http://<pi-host>:8000 in a browser.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as e:
    raise SystemExit(
        "Dashboard requires fastapi + uvicorn. Install with:\n"
        "  pip install 'kovio[dashboard]'\n"
        f"(missing: {e})"
    )

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
DEFAULT_DB = Path("kovio.db")


def create_app(db_path: str | Path = DEFAULT_DB) -> FastAPI:
    db_path = Path(db_path)
    app = FastAPI(title="Kovio Dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _conn() -> sqlite3.Connection:
        if not db_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"db not found: {db_path} — has the agent run yet?",
            )
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        return c

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/state")
    def get_state() -> dict:
        c = _conn()
        try:
            last_scene = c.execute(
                "SELECT * FROM events WHERE event_type='scene_observed' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            last_ad = c.execute(
                "SELECT * FROM events WHERE event_type IN ('ad_played','ad_suppressed') "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            campaigns = c.execute(
                "SELECT campaign_id, name, advertiser, priority FROM campaigns "
                "WHERE enabled=1 ORDER BY priority DESC"
            ).fetchall()
        except sqlite3.OperationalError:
            campaigns = []
            last_scene = last_ad = None
        c.close()

        return {
            "last_scene": dict(last_scene) if last_scene else None,
            "last_ad": dict(last_ad) if last_ad else None,
            "campaigns": [dict(r) for r in campaigns],
            "server_time": time.time(),
        }

    @app.get("/api/today")
    def get_today() -> dict:
        c = _conn()
        start = time.time() - 86400
        impressions = c.execute(
            """
            SELECT json_extract(payload_json, '$.campaign_id') AS campaign_id,
                   COUNT(*) AS count
            FROM events
            WHERE event_type='ad_played' AND timestamp >= ?
            GROUP BY campaign_id
            ORDER BY count DESC
            """,
            (start,),
        ).fetchall()
        people = c.execute(
            """
            SELECT COALESCE(SUM(CAST(json_extract(payload_json, '$.person_count') AS INTEGER)), 0) AS total
            FROM events WHERE event_type='scene_observed' AND timestamp >= ?
            """,
            (start,),
        ).fetchone()
        c.close()
        return {
            "impressions_by_campaign": [dict(r) for r in impressions],
            "people_observed_total": int(people["total"]) if people else 0,
        }

    @app.get("/api/events")
    def get_events(limit: int = 50) -> list[dict]:
        c = _conn()
        rows = c.execute(
            "SELECT timestamp, event_type, robot_id, payload_json "
            "FROM events ORDER BY timestamp DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]

    return app


def main() -> None:
    """Entry point: `python -m kovio.dashboard.server`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Kovio dashboard")
    parser.add_argument("--db", default="kovio.db", help="path to kovio.db")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    args = parser.parse_args()

    app = create_app(args.db)
    print(f"\n📊 Kovio dashboard → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
