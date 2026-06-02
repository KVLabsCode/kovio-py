"""Kovio CLI — entry point for running the SDK on any platform."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from .agent import KovioAgent
from .platform import detect_platform, find_chromium

log = logging.getLogger("kovio")


def cmd_doctor(args) -> int:
    """Print platform info — useful for debugging install issues."""
    import importlib.util

    plat = detect_platform()
    chromium = find_chromium() or "(not found)"
    print(f"Platform:            {plat.value}")
    print(f"Chromium binary:     {chromium}")

    # Each adapter module imports cleanly (heavy deps are imported lazily inside
    # start()), so we probe the underlying runtime dependencies directly to
    # report whether the adapter can actually run on this machine.
    adapters = [
        ("Mock perception", []),
        ("Orbbec perception", ["openni", "numpy"]),
        ("RealSense perception", ["pyrealsense2", "onnxruntime", "cv2", "numpy"]),
    ]
    for name, deps in adapters:
        missing = [d for d in deps if importlib.util.find_spec(d) is None]
        if missing:
            print(f"{name:20s} unavailable (install: pip install '{_extra_hint(name)}'; "
                  f"missing {', '.join(missing)})")
        else:
            print(f"{name:20s} OK")
    return 0


def _extra_hint(adapter_name: str) -> str:
    """Map an adapter to the pip extra that installs its dependencies."""
    if "Orbbec" in adapter_name:
        return "kovio[pi]"
    if "RealSense" in adapter_name:
        return "kovio[jetson]"
    return "kovio[dev]"


DEMO_PORT = 8001


def cmd_demo(args) -> int:
    """Run the SDK with mock perception and a browser-viewable screen.

    Unlike `serve`, the demo doesn't spawn a kiosk browser — it serves the
    robot screen at http://localhost:8001 for you to open. An attention gate
    keeps the screen idle until the (scripted) mock perception reports someone
    looking, at which point the default creative goes up with a save QR. Tap it
    to record an engagement.
    """
    import os

    from .adapters.screen import BrowserScreenAdapter
    from .types import GateDecision

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    os.environ["KOVIO_PERCEPTION"] = "mock"

    db_path = "kovio.db"
    screen = BrowserScreenAdapter(db_path=db_path, robot_id=args.robot_id, port=DEMO_PORT)
    agent = KovioAgent.autodetect(robot_id=args.robot_id, screen=screen, db_path=db_path)

    @agent.task_gate
    def _attention_gate(task_state, scene) -> GateDecision:
        # Demo storyline: only surface a creative when someone is actually
        # attending the screen. Idle wordmark the rest of the time.
        if scene and scene.attended_count > 0:
            return GateDecision.allow()
        return GateDecision.suppress("no_attention")

    _print_demo_banner(DEMO_PORT)
    return _run_agent(agent)


def _print_demo_banner(port: int) -> None:
    lines = [
        "kovio demo — robot screen is live",
        "",
        f"Open  http://localhost:{port}  in your browser.",
        "It breathes a wordmark while idle, then plays the",
        "creative when the mock perception reports attention.",
        "Tap the creative to record an engagement.",
        "",
        "Ctrl-C to stop.",
    ]
    width = max(len(s) for s in lines) + 4
    top = "  ┌" + "─" * width + "┐"
    bottom = "  └" + "─" * width + "┘"
    body = "\n".join(f"  │  {s:<{width - 4}}  │" for s in lines)
    print(f"\n{top}\n{body}\n{bottom}\n")


def cmd_serve(args) -> int:
    """Run the SDK with platform-detected adapters."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    agent = KovioAgent.autodetect(robot_id=args.robot_id)
    return _run_agent(agent)


def _run_agent(agent: KovioAgent) -> int:
    """Start the agent and block until Ctrl-C, then shut it down cleanly."""
    agent.start()
    log.info("Agent running. Press Ctrl-C to stop.")
    try:
        if hasattr(signal, "pause"):
            signal.pause()  # Unix: blocks until a signal (Ctrl-C) arrives
        else:
            # Windows has no signal.pause() — busy-wait instead.
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="kovio")
    parser.add_argument("--robot-id", default="tank-001")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Print platform & adapter status")
    sub.add_parser("demo", help="Run with mock perception (laptop dev)")
    sub.add_parser("serve", help="Run with platform-detected adapters")

    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
