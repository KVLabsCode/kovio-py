"""Tier-1 drop-in example with a REAL Astra depth camera.

Prerequisites:
  - OpenNI 2 installed via scripts/setup_astra_pi.sh
  - pip install "kovio[astra]"  (or: pip install openni numpy)
  - An Astra-class depth camera plugged in via USB
  - OPENNI2_REDIST in your environment

What you'll see:
  - Real SceneState events with person_count derived from depth
  - Ads play when the simulated task state is IDLE and people are nearby
  - Walk in front of the camera and watch the counts move

Ctrl-C to stop.
"""
from __future__ import annotations

import logging
import shutil
import time

from kovio import KovioAgent, ScreenAdapter, TaskState
from kovio.adapters.orbbec_perception import OrbbecPerceptionAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-26s %(message)s",
    datefmt="%H:%M:%S",
)

# Use the Pi touchscreen if Chromium is available; otherwise log.
HAVE_CHROMIUM = bool(
    shutil.which("chromium-browser")
    or shutil.which("chromium")
    or shutil.which("google-chrome")
)
screen = ScreenAdapter.pi_touchscreen() if HAVE_CHROMIUM else ScreenAdapter.logger()

agent = KovioAgent(
    robot_id="tank-001",
    screen=screen,
    perception=OrbbecPerceptionAdapter(
        min_depth_m=0.6,    # Astra's minimum reliable range
        max_depth_m=4.0,    # Tune for your venue
        rate_hz=2.0,        # 2 scene updates per second
    ),
)

agent.start()

try:
    # Keep the task state IDLE so ads play whenever people are detected.
    # In a real Tank, your autonomy stack drives this.
    while True:
        agent.update_task_state(TaskState.IDLE)
        time.sleep(60)
except KeyboardInterrupt:
    print("\nshutting down...")
finally:
    agent.stop()
