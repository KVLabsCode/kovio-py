# Always-on Kovio agent (robot deployment)

Run the Kovio SDK as a systemd service so on-device perception + ad playout
starts on every boot and restarts on failure. Target: Unitree Go2 / Jetson Orin
with an Intel RealSense depth camera and the Go2 Livox lidar.

## Install

From a checkout of `kovio-py` on the robot:

```bash
sudo deploy/install_robot.sh --install-python   # omit the flag if you already have python3.10+
```

This builds a venv at `/opt/kovio/venv`, installs `kovio[jetson]`, pre-caches the
YOLO detection + pose models, writes `/etc/kovio/kovio.env`, and enables
`kovio-agent.service`.

Then set your credentials and restart:

```bash
sudoedit /etc/kovio/kovio.env     # KOVIO_API_KEY, KOVIO_ROBOT_ID
sudo systemctl restart kovio-agent
```

## Verify

```bash
systemctl status kovio-agent
journalctl -u kovio-agent -f      # watch perception + event uploads live
/opt/kovio/venv/bin/kovio doctor  # platform + adapter sanity check
```

## What it captures

With `KOVIO_PERCEPTION=rich` the agent emits, entirely on-device (no frame ever
leaves the robot):

| Metric | Source | Event |
|---|---|---|
| people in view, attention, distance | RealSense RGB-D + YOLO | `scene_observed` |
| looked-at-robot, dwell seconds | tracker + gaze proxy | `scene_observed` |
| people nearby, density, nearest, approach bearing | Livox lidar | `scene_observed` |
| phone-out | YOLO `cell phone` | `interaction_observed` |
| handshake / wave / high-five / fist-bump | YOLO-pose + gestures | `interaction_observed` |

Every capability degrades gracefully: no pose model → no gestures; no lidar →
no crowd fields; the original `person_count` / `attended_count` / distance are
always present.

## Tuning the feature gates

`kovio serve` runs `RichPerceptionAdapter` with every branch on. To disable one
(e.g. skip gestures on a lower-power unit) or change thresholds, run a thin
launcher instead of `kovio serve` in the service's `ExecStart`:

```python
# /opt/kovio/launch.py
from kovio import KovioAgent
from kovio.adapters.screen import ScreenAdapter
from kovio.adapters.rich_perception import RichPerceptionAdapter

agent = KovioAgent(
    screen=ScreenAdapter.from_env(),
    perception=RichPerceptionAdapter(
        enable_gestures=True,
        enable_phone=True,
        enable_gaze=True,
        enable_lidar=True,
        gaze_dwell_seconds=1.5,
        lidar_radius_m=4.0,
    ),
)
agent.start()
import signal; signal.pause()
```

```ini
# /etc/systemd/system/kovio-agent.service  (override ExecStart)
ExecStart=/opt/kovio/venv/bin/python /opt/kovio/launch.py
```

## Privacy

Perception runs on-device. Only derived counts and aggregates travel to the
cloud — never images, never identity. Track ids are ephemeral integers that
reset on restart.
