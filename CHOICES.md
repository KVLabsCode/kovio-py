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
