# Changelog

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
