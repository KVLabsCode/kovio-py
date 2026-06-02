# Changelog

## 0.0.8 — Cloud connectivity

- New `kovio.config.CloudConfig` reads cloud URL + API key from env vars
- Optional `.env` file support (no python-dotenv dependency — built-in parser)
- `KovioAgent.autodetect()` now constructs `CloudCampaignStore` + `CloudEventSink`
  automatically when `KOVIO_API_URL` and `KOVIO_API_KEY` are present
- `kovio doctor` shows cloud configuration + reachability (`/healthz` probe)
- `BrowserScreenAdapter` adds `/api/current` (state + live campaign id/advertiser)
- Backward compatible: SDK falls back to default_creative.html if env vars are unset
- Fixed: `agent_started` event payload now reads version from `kovio.__version__`
  instead of a hardcoded string (was `0.0.3`)

## 0.0.7 — Browser demo screen

- `kovio demo` now serves a browser-viewable robot screen at
  http://localhost:8001 instead of only logging creative URLs. Idle shows a
  breathing italic "kovio" wordmark; when the mock perception reports attention
  the default creative goes full-screen with a save-QR and an engage hint.
  Tapping records an `engagement` event in `kovio.db` (cloud-syncs like the rest).
- Added `BrowserScreenAdapter` (+ `ScreenAdapter.browser()` factory) — a
  dependency-free stdlib HTTP screen used by the demo.
- `kovio demo` gates creatives on `attended_count > 0` so the screen idles until
  someone is actually looking.
- `kovio[dev]` extra now pulls `qrcode` for a scannable save-QR (degrades to a
  placeholder if absent).

## 0.0.6 — Multi-platform support

- Added `kovio/platform.py` for hardware detection (Pi 5 / Jetson Orin / laptop)
- Added `MockPerceptionAdapter` (cross-platform dev — deterministic scripted scenes)
- Added `RealSensePerceptionAdapter` (D455 on Jetson + anywhere with pyrealsense2)
- Added `make_perception_adapter()` / `make_screen_adapter()` factories
- Added `KovioAgent.autodetect(...)` — platform-detected adapters with env overrides
- Pip extras: `kovio[pi]`, `kovio[jetson]`, `kovio[dev]`, `kovio[all]` (`[astra]` now aliases `[pi]`)
- New CLI: `kovio doctor`, `kovio demo`, `kovio serve` (registered as `kovio` console script)
- `ChromiumKioskAdapter` now searches multiple browser binary names via `find_chromium()`
- README documents three platform install paths
- `KOVIO_PERCEPTION` / `KOVIO_SCREEN` env vars override platform defaults

## 0.0.5

- Cloud-synced `CampaignStore` and event sink
