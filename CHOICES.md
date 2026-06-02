# Implementation choices — 0.0.6 multi-platform support

Non-obvious decisions made while implementing multi-platform support. Defaults
from the prompt followed unless noted below.

## Deviations / additions beyond the prompt spec

- **`pi` extra includes `numpy`.** The prompt's `pi` extra listed only `openni`,
  but `OrbbecPerceptionAdapter` imports `numpy` for depth-mask math. Without it,
  `pip install '.[pi]'` would leave the Pi adapter broken at runtime. Added
  `numpy>=1.24.0`.

- **`astra` extra kept as a deprecated alias for `pi`.** 0.0.5 shipped a
  `kovio[astra]` extra. Removing it outright would break existing Pi install
  scripts, so `astra = ["kovio[pi]"]` preserves backward compatibility.

- **`dev` extra keeps `build` / `twine` / `ruff`.** The prompt's `dev` extra was
  just `dashboard` + `pytest`. Kept the 0.0.5 tooling deps so `python -m build`
  and linting work straight from `pip install -e '.[dev]'`.

- **Added `make_screen_adapter()` factory.** The prompt fully specified
  `make_perception_adapter()` but only sketched "screen adapter via similar
  factory." Implemented a symmetric `make_screen_adapter()` honoring
  `KOVIO_SCREEN` + `default_screen()`, used by `KovioAgent.autodetect()`.

- **`ChromiumKioskAdapter` now raises `SystemExit` on construction when no
  browser is found** (per Phase 4.2), replacing the previous silent log-only
  fallback. This is only reachable when a browser is explicitly requested
  (`KOVIO_SCREEN=chromium` or direct construction); `autodetect()`/`default_screen()`
  only pick `chromium` when `find_chromium()` already succeeded, so the normal
  laptop path still lands on the logger adapter. The error message points users
  at `KOVIO_SCREEN=logger`.

- **`MockPerceptionAdapter` vs existing `StubPerceptionAdapter`.** Kept both. The
  stub emits *random* scenes (still the `PerceptionAdapter.stub()` factory and the
  agent's default). The new mock loops a *deterministic 12-scene script* for
  reproducible demos — this is what `kovio demo` / `KOVIO_PERCEPTION=mock` use.

- **`cmd_serve` stops the agent on exit.** Added a `finally: agent.stop()` and a
  nested KeyboardInterrupt guard on the Windows `time.sleep` fallback so Ctrl-C
  shuts the perception thread + screen down cleanly.

## Followed as specified

- Platform detection via `/proc/device-tree/model`, falling back to
  `platform.system()` / `platform.machine()`.
- YOLOv8n ONNX downloaded on first use to `~/.cache/kovio/models/` — NOT shipped
  in the wheel.
- Depth used only for `mean_distance_m`; adapter does not hard-require depth.
- `find_chromium()` is the single source of truth for the browser binary.
- Core package keeps ZERO mandatory dependencies; all hardware libs are extras.
- `SceneState` unchanged — `attended_count` / contextual targeting deferred to the
  next milestone per the prompt.

## Known gaps (carried into the summary)

- YOLOv8n runs via onnxruntime; Jetson TensorRT optimization is a follow-up.
- RealSense `attended_count` heuristic = "person closer than 2m"; real gaze
  estimation is a follow-up.
- No automated tests for adapter switching yet.
- The two example scripts under `examples/` still probe PATH for Chromium
  directly; they guard construction with `HAVE_CHROMIUM` so they remain correct,
  but were left untouched to keep this change adapter-focused.

# Browser demo screen (post-0.0.6)

Decisions made adding the browser-viewable `kovio demo` screen.

- **`BrowserScreenAdapter` uses stdlib `http.server`, not FastAPI.** Screen
  adapters live in the zero-dep core, so the demo server can't take the
  dashboard's FastAPI/uvicorn dependency. A small `ThreadingTCPServer` keeps the
  adapter installable everywhere. (The QR is the one optional dep — `qrcode`, in
  `[dev]` — and degrades to a placeholder when absent.)
- **`kovio demo` registers an attention gate.** Without a selector the agent
  plays continuously whenever the task state is IDLE, so the screen would never
  idle. A `task_gate` that allows only when `scene.attended_count > 0` produces
  the intended idle → creative → idle storyline as the mock script cycles.
- **The adapter writes engagements straight to the events table.** Rather than
  threading a callback back into the agent, the adapter appends an `engagement`
  `AdEvent` to the same `kovio.db` (creating the table if needed). This keeps the
  cloud sink's drain path unchanged — engagements upload like any other event.
- **`make_screen_adapter()` does NOT construct the browser adapter.** That
  name-based factory has no DB/robot-id context, and the demo screen needs both.
  `kovio demo` builds it explicitly and passes `screen=` to `autodetect()`.
- **Local `file://` creatives are proxied via `/creative`** so the page can frame
  them same-origin; `http(s)` creatives are framed directly.

# Cloud connectivity (0.0.8)

Decisions made wiring `CloudCampaignStore` / `CloudEventSink` into `autodetect()`.

- **`CloudCampaignStore` / `CloudEventSink` construction.** `autodetect()` builds
  these from `CloudConfig`. (In 0.0.8 they took only their original params and
  `KOVIO_API_TIMEOUT` reached just the `doctor` probe; 0.0.9 added a `timeout`
  param to both so it flows into every cloud call — see below.)
- **A cloud `store` becomes a `RuleBasedSelector` inside `KovioAgent.__init__`.**
  The agent plays via a *selector*, not a store, so `autodetect()` hands the store
  to `__init__`, which wraps it in the default `RuleBasedSelector` unless an explicit
  `selector=` was given. With no store and no selector, the fixed `creative_url`
  (default creative) path is untouched — that's the backward-compatible local mode.
- **Cloud mode shows real campaigns or nothing — never the default creative.** When
  a selector is present but returns no eligible campaign, the agent suppresses (idle)
  rather than falling back to `creative_url`. So with the cloud configured you see
  your campaigns (or a blank screen if none are eligible / cache is empty), which is
  the intended "no more default_creative.html" behavior. The default creative only
  appears in local (no-selector) mode.
- **CLI `--robot-id` default changed from `"tank-001"` to `None`.** Otherwise the CLI
  always passed an explicit id and `KOVIO_ROBOT_ID` / hostname could never apply.
  Resolution order is now: `--robot-id` flag > `KOVIO_ROBOT_ID` > hostname. The only
  observable change in the no-env path is the default robot id label (hostname instead
  of `tank-001`) — a label in the event stream, not a behavior change.
- **`agent.py` reads `__version__` via a lazy `from . import __version__`** inside
  `start()`, not a top-level import — the package `__init__` imports `.agent`, so a
  module-level import would be circular. By the time `start()` runs the package is
  fully initialized.
- **`load_cloud_config()` is called twice on the `kovio demo` path** (once in
  `cmd_demo` to resolve the shared robot id for the screen + agent, once inside
  `autodetect`). It's idempotent (`.env` only fills unset vars; shell env wins), so
  the only cost is a duplicated info log line. Accepted rather than threading a
  pre-built config through `autodetect`'s signature.
- **`/api/current` reads the latest `ad_played` from SQLite** for `campaign_id` /
  `advertiser` rather than plumbing campaign metadata through `ScreenAdapter.display()`
  (which takes only a URL). Keeps the adapter interface unchanged; campaign fields are
  null for the default creative (its play event carries no campaign), which is the
  signal distinguishing a cloud campaign from the local fallback.

# Cloud campaign fixes (0.0.9)

Found by connecting `kovio demo` to the live `kovio-api.fly.dev` with a real SDK key.

- **Symbolic targeting operators.** The production API returns targeting rules like
  `{"op": ">=", "field": "person_count", "value": 1}`, but `TargetingRule.evaluate`
  only recognized named ops (`gte`, `lte`, …) and silently returned `False` for
  anything else — so *every* cloud campaign failed to match and nothing ever played.
  Added an `_OP_ALIASES` map (`>=`→`gte`, `<=`→`lte`, `>`→`gt`, `<`→`lt`, `==`/`=`→`eq`,
  `!=`→`ne`) and `gt`/`lt` branches. Local `campaigns.json` (named ops) is unaffected.
- **`timeout` on `CloudCampaignStore` / `CloudEventSink`.** The fly server takes
  ~6.5s to accept a 20-event batch (cold), so the hardcoded 8s timeout failed the
  sink's 100-event backlog drains — uploads silently stalled. Both classes now take a
  `timeout` (default unchanged at `_DEFAULT_TIMEOUT`) threaded into `_get_json`/
  `_post_json`, fed from `KOVIO_API_TIMEOUT`. The `.env` for the demo sets `30`.
  `batch_size` left at 100 — a clean robot's batches are small; the timeout is the
  real lever for a slow link.
- **Relative-path creatives render.** A cloud campaign's `creative_path` is a bare
  relative path (`creatives/kovio_brand.html`) the robot is expected to have on disk,
  not a URL — and the API does not serve it. `BrowserScreenAdapter` now routes any
  non-`http(s)` creative through `/creative`, resolving file:// **and** relative/
  absolute local paths against the working directory. Shipped `creatives/kovio_brand.html`
  (Kovio's own brand ad) so the bundled cloud campaign renders out of the box.
- **Cloud mode is "real campaigns or idle."** Confirmed live: with the cloud
  configured the screen idles until a campaign both passes the demo attention gate and
  matches its targeting, then plays the real creative — it never shows the local
  default. (That's local-only mode.)
