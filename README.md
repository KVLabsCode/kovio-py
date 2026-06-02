# Kovio

The open SDK for monetizing autonomous robots.

Kovio runs on your robot. Each interaction it has with a person — a glance at the screen, a paused dwell at the shelf-edge, a delivery handoff — becomes a measurable, revenue-generating moment. Perception happens on-device; camera frames never leave the robot. Only the derived counts and aggregates travel.

```python
from kovio import KovioAgent, ScreenAdapter, PerceptionAdapter
from kovio.campaigns import CampaignStore, RuleBasedSelector

store = CampaignStore("campaigns.json")
agent = KovioAgent(
    robot_id="my-robot-001",
    screen=ScreenAdapter.pi_touchscreen(),
    perception=PerceptionAdapter.orbbec_astra(),
    selector=RuleBasedSelector(store),
)
agent.start()
```

This is enough to play context-aware ads on a Raspberry Pi 5 with an Orbbec Astra depth camera and a 7" touchscreen. Status: closed beta. Working with the first design-partner OEMs and brands. [kovio.dev](https://kovio.dev)

## Install

The same `kovio` package runs on three platforms. Pick the section for your hardware.

### Development laptop (any OS — macOS, Linux, Windows)

```bash
pip install -e '.[dev]'
kovio doctor          # confirms platform detection + adapter availability
kovio demo            # runs with synthetic scenes — no camera needed
```

`kovio demo` serves the robot screen at **http://localhost:8001** — open it in
your browser. It breathes a "kovio" wordmark while idle, then plays the default
creative (with a save-QR) when the mock perception reports someone attending.
Tap the creative to record an engagement in `kovio.db`.

### Raspberry Pi 5 + Orbbec Astra

```bash
# One-time system setup
bash scripts/setup_astra_pi.sh
# SDK install
pip install -e '.[pi]'
kovio doctor
kovio serve
```

### NVIDIA Jetson Orin Nano Super + Intel RealSense D455

```bash
# One-time RealSense system setup
sudo apt install librealsense2-utils librealsense2-dev
# SDK install
pip install -e '.[jetson]'
kovio doctor
kovio serve
```

### Override platform detection

```bash
# Force any adapter regardless of platform:
KOVIO_PERCEPTION=mock kovio serve         # Synthetic scenes
KOVIO_PERCEPTION=orbbec kovio serve       # Force Astra
KOVIO_PERCEPTION=realsense kovio serve    # Force D455
KOVIO_SCREEN=logger kovio serve           # Log creative URLs instead of opening Chromium
```

PyPI release is gated on the v0.1 release. For now, install from this repo at a tag.

## Three things you should know before going further

**On-device by construction.** All perception runs on the robot. Camera frames are never transmitted, logged, or persisted. The SDK accepts only derived `SceneState` events (counts and aggregates). The cloud never sees images. This is enforced in the type system, not by convention.

**Three integration tiers.** Most OEMs land in one of these:

- **Tier 1** — use a bundled perception adapter (Orbbec Astra today; RealSense and OAK-D Lite next). Zero perception code from the OEM.
- **Tier 2** — Tier 1 + custom hooks: override the task gate, plug in extra event sinks, customize creative selection.
- **Tier 3** — bring your own perception. Subclass `PerceptionAdapter` against the `SceneState` schema; the SDK never touches your camera. Usually ~100–300 lines of OEM-side code. This is what mature robotics teams choose.

**Python in-process today; ROS 2 native next.** Today the SDK runs as a Python library co-located with your perception. ROS 2 (Humble + Jazzy) lifecycle node lands in v0.5. Local HTTP / gRPC transport for cross-language stacks in v0.6+.

## What's in the box

| Module | What it does |
|---|---|
| `kovio.agent` | The main `KovioAgent` event loop: task gate, perception subscriber, event log, dashboard wiring |
| `kovio.adapters.screen` | Bundled screen adapters (Chromium kiosk, logger-only for headless test) |
| `kovio.adapters.perception` | Bundled perception adapters: stub/mock for tests; Orbbec Astra (depth-only person counting) on Pi; Intel RealSense D455 (YOLOv8n + depth) on Jetson |
| `kovio.platform` | Platform detection + adapter defaults (Pi / Jetson / laptop) |
| `kovio.cli` | `kovio doctor`, `kovio demo`, `kovio serve` — platform-aware entry points |
| `kovio.campaigns` | `Campaign`, `TargetingRule`, `DecisionContext`, `CampaignStore`, `RuleBasedSelector` |
| `kovio.dashboard` | Optional FastAPI dashboard server + single-page HTML view of live state, campaigns, impressions |
| `kovio.types` | `SceneState`, `TaskState`, `AdEvent`, `GateDecision` — the public schemas |

Everything else — billing, demand integration, cloud campaign sync — lives in the closed-source Kovio platform.

## Architecture

```
[ camera ] → [ perception adapter ] → SceneState
                                         │
[ task state from autonomy ] ───────────▶│
                                         ▼
                              [ KovioAgent + selector ]
                                         │
                                         ▼
                              [ screen ] + [ event log ]
                                                │
                                                ▼
                                  [ local dashboard ] [ cloud sync, v0.5+ ]
```

Every input the agent reasons over is a `SceneState` event. Every output is a creative play + an `AdEvent` written to a local SQLite event log. The cloud platform consumes the event log; raw frames never appear in it.

The complete architecture, integration patterns, perception contract, and conformance test suite are in [the design doc](https://kovio.dev/docs).

## Quick start

There's a runnable example at `examples/tier1_with_selector/`:

```bash
git clone https://github.com/kovio-labs/kovio.git
cd kovio
pip install -e ".[dashboard]"

# Edit examples/tier1_with_selector/main.py — flip USE_REAL_CAMERA=True if you have an Astra wired up
python examples/tier1_with_selector/main.py

# In another terminal, the dashboard:
python -m kovio.dashboard.server --db kovio.db
# Open http://localhost:8000
```

See [kovio.dev/docs](https://kovio.dev/docs) for the full integration guide.

## Privacy posture

The SDK is engineered around one constraint: **raw camera data never leaves the robot.** Not to our cloud, not to an OEM's cloud, not to disk in any persistent form. The `SceneState` schema accepts only:

- `person_count` (integer)
- `attended_count` (integer, ≤ `person_count`)
- `mean_distance_m` (float, in meters)
- `timestamp` (epoch seconds)

The conformance test suite includes an invariant that fails if any `bytes`-typed field appears in any event payload. This makes the privacy posture *provable* rather than promised.

Full statement: [kovio.dev/privacy](https://kovio.dev/privacy).

## Status

This is v0.0.4 — closed beta. The SDK is in active development with the first design-partner OEMs and brands. The API surface is stable enough to integrate against; breaking changes will be flagged in `CHANGELOG.md` with at least one minor-version notice.

**Working today**

- Tier 1 / Tier 3 Python integration
- Campaign loading from JSON
- Rule-based selector with priority + encounter capping
- Local FastAPI dashboard
- Orbbec Astra perception adapter (depth-only person counting)

**Coming next**

- ROS 2 lifecycle node (Humble + Jazzy)
- `RecordingPerceptionAdapter` + `ReplayPerceptionAdapter` for cross-robot testing
- Conformance test suite (pytest plugin) for OEM adapter validation
- Cloud-synced `CampaignStore` and event sink
- Hailo + RGB head-pose pipeline for verified attention (`attended_count > 0`)

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

- General: hello@kovio.dev
- OEM integrations: oem@kovio.dev
- Security: security@kovio.dev
- Repo: [github.com/kovio-labs/kovio](https://github.com/kovio-labs/kovio)
