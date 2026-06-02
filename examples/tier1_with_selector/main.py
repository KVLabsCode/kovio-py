"""Tier-1 with a real Astra camera AND context-driven ad selection.

What this demonstrates:
  - Real depth perception (OrbbecPerceptionAdapter)
  - Campaign store loaded from campaigns.json
  - RuleBasedSelector picking the right ad based on time of day and people
  - Live dashboard available at http://<pi>:8000 (run server.py separately)

Run in two terminals:

  # Terminal 1 — the agent
  python examples/tier1_with_selector/main.py

  # Terminal 2 — the dashboard
  python -m kovio.dashboard.server --db kovio.db
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from kovio import KovioAgent, ScreenAdapter, TaskState
from kovio.campaigns import CampaignStore, RuleBasedSelector

# Swap PerceptionAdapter.stub() for the real Astra adapter when ready.
USE_REAL_CAMERA = False

if USE_REAL_CAMERA:
    from kovio.adapters.orbbec_perception import OrbbecPerceptionAdapter
    perception = OrbbecPerceptionAdapter(min_depth_m=0.6, max_depth_m=4.0, rate_hz=2.0)
else:
    from kovio import PerceptionAdapter
    perception = PerceptionAdapter.stub(rate_hz=1.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-26s %(message)s",
    datefmt="%H:%M:%S",
)

# Find the project root so creative paths resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_JSON = PROJECT_ROOT / "campaigns.json"
DB_PATH = PROJECT_ROOT / "kovio.db"

# Load campaigns + build selector
store = CampaignStore(json_path=CAMPAIGNS_JSON, db_path=DB_PATH)
selector = RuleBasedSelector(store)
print(f"Loaded {len(store.active_campaigns())} active campaigns from {CAMPAIGNS_JSON}")

# Choose screen
have_chrome = bool(shutil.which("chromium-browser") or shutil.which("chromium"))
screen = ScreenAdapter.pi_touchscreen() if have_chrome else ScreenAdapter.logger()

agent = KovioAgent(
    robot_id="tank-001",
    screen=screen,
    perception=perception,
    selector=selector,
    db_path=DB_PATH,
)

agent.start()

try:
    while True:
        agent.update_task_state(TaskState.IDLE)
        time.sleep(60)
except KeyboardInterrupt:
    print("\nshutting down...")
finally:
    agent.stop()
