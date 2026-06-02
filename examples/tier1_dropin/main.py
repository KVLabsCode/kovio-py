"""Tier-1 drop-in example: get ads playing in ~10 lines.

Run on a Pi with Chromium installed:

    python examples/tier1_dropin/main.py

On a dev machine without a display, swap ScreenAdapter.pi_touchscreen()
for ScreenAdapter.logger() and you'll see events log to stdout.

Ctrl-C to stop. Events accumulate in kovio.db (SQLite); inspect with:

    sqlite3 kovio.db "select * from events order by timestamp desc limit 20;"
"""
from __future__ import annotations

import logging
import shutil
import time

from kovio import (
    PerceptionAdapter,
    KovioAgent,
    ScreenAdapter,
    TaskState,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-22s %(message)s",
    datefmt="%H:%M:%S",
)

# Auto-detect: use the Pi touchscreen adapter if Chromium is on PATH,
# otherwise log events to stdout so this also runs on a dev machine.
HAVE_CHROMIUM = bool(
    shutil.which("chromium-browser")
    or shutil.which("chromium")
    or shutil.which("google-chrome")
)
screen = ScreenAdapter.pi_touchscreen() if HAVE_CHROMIUM else ScreenAdapter.logger()

agent = KovioAgent(
    robot_id="tank-001",
    screen=screen,
    perception=PerceptionAdapter.stub(rate_hz=1.0),
)

agent.start()

try:
    # Simulate a typical delivery loop.
    while True:
        print("\n--- robot is IDLE (ads play) ---")
        agent.update_task_state(TaskState.IDLE)
        time.sleep(10)

        print("\n--- robot is NAVIGATING (ads suppress) ---")
        agent.update_task_state(TaskState.NAVIGATING)
        time.sleep(5)

        print("\n--- robot is DELIVERING (ads suppress) ---")
        agent.update_task_state(TaskState.DELIVERING)
        time.sleep(3)

        print("\n--- handing off to customer (HARD suppress) ---")
        agent.update_task_state(TaskState.CUSTOMER_HANDOFF)
        time.sleep(3)
except KeyboardInterrupt:
    print("\nshutting down...")
finally:
    agent.stop()
