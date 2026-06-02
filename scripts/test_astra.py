"""Astra smoke test — pure OpenNI 2, no Kovio imports.

Run after scripts/setup_astra_pi.sh has completed successfully and you've
unplugged + replugged the camera. Captures 30 depth frames and prints
basic stats to confirm the camera is alive.

  python3 scripts/test_astra.py
"""
from __future__ import annotations

import os
import sys
import time

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install --user numpy")
    sys.exit(1)

try:
    from openni import openni2
except ImportError:
    print("ERROR: openni not installed. Run: pip install --user openni")
    sys.exit(1)


# Try common OpenNI 2 install locations.
SEARCH_PATHS = [
    os.environ.get("OPENNI2_REDIST", ""),
    os.path.expanduser("~/orbbec-build/OpenNI2/Bin/Arm64-Release"),
    "/usr/local/lib/OpenNI2/Redist",
    "/usr/lib/OpenNI2/Redist",
]


def find_openni2_redist() -> str | None:
    for p in SEARCH_PATHS:
        if not p:
            continue
        lib = os.path.join(p, "libOpenNI2.so")
        if os.path.exists(lib):
            return p
    return None


def show_usb() -> None:
    print("\nUSB devices currently visible:")
    os.system("lsusb 2>/dev/null | sed 's/^/  /'")
    print("\nLook for a line with vendor 2bc5 (Orbbec). If absent:")
    print("  - check the cable (Astra needs USB data+power)")
    print("  - try a different USB port")
    print("  - try a powered USB hub")


def main() -> int:
    path = find_openni2_redist()
    if path:
        print(f"==> Using OpenNI 2 from: {path}")
        openni2.initialize(path)
    else:
        print("==> No explicit OpenNI 2 path found; trying default search")
        openni2.initialize()

    try:
        device = openni2.Device.open_any()
    except Exception as e:
        print(f"❌ No Astra camera found: {e}")
        show_usb()
        return 1

    info = device.get_device_info()
    name = info.name.decode() if isinstance(info.name, bytes) else info.name
    print(f"✅ Camera opened: {name}")

    depth = device.create_depth_stream()
    depth.start()
    print("\n==> Capturing 30 depth frames at ~10 Hz")
    print("    (depth values are in millimeters)\n")

    for i in range(30):
        frame = depth.read_frame()
        buf = frame.get_buffer_as_uint16()
        img = np.frombuffer(buf, dtype=np.uint16).reshape(frame.height, frame.width)
        valid = img[(img > 0) & (img < 8000)]

        if valid.size:
            print(
                f"  frame {i:02d}: {frame.width}x{frame.height}  "
                f"valid={valid.size:6d}  "
                f"min={valid.min():4d}mm  "
                f"mean={valid.mean():6.0f}mm  "
                f"max={valid.max():4d}mm"
            )
        else:
            print(f"  frame {i:02d}: no valid depth (camera blocked or too close?)")
        time.sleep(0.1)

    depth.stop()
    openni2.unload()
    print("\n✅ Astra is working. You can now run the SDK example:")
    print("   python3 examples/tier1_with_astra/main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
