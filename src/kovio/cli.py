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


def cmd_demo(args) -> int:
    """Run the SDK with mock perception — works on any laptop."""
    import os
    os.environ["KOVIO_PERCEPTION"] = "mock"
    return cmd_serve(args)


def cmd_serve(args) -> int:
    """Run the SDK with platform-detected adapters."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    agent = KovioAgent.autodetect(robot_id=args.robot_id)
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
