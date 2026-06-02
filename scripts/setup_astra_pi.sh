#!/usr/bin/env bash
#
# Kovio — Astra/OpenNI 2 setup for Raspberry Pi 5 (aarch64).
#
# Supports the classic Astra family:
#   Astra · Astra Pro · Astra S · Astra Mini · Astra Pro Plus
#
# For newer cameras (Astra+, Astra 2, Femto, Gemini), use pyOrbbecSDK
# instead — separate path, not covered by this script.
#
# What this does:
#   1. Installs build dependencies via apt
#   2. Clones Orbbec's OpenNI 2 fork and builds it for Arm64
#   3. Installs udev rules so non-root users can talk to the camera
#   4. Installs the Python wrapper (`openni`)
#   5. Adds OPENNI2_REDIST to ~/.bashrc
#
# Run: ./scripts/setup_astra_pi.sh
# Expect: 10-15 minutes on a Pi 5.

set -euo pipefail

echo
echo "==> Kovio — Astra setup for Pi 5"
echo

# ---- 0. sanity ----
if ! command -v lsusb >/dev/null; then
    sudo apt update && sudo apt install -y usbutils
fi

echo "==> Looking for Orbbec device on USB..."
if lsusb | grep -qi "2bc5\|orbbec"; then
    echo "  ✅ Detected:"
    lsusb | grep -i "2bc5\|orbbec" | sed 's/^/    /'
else
    echo "  ⚠️  No Orbbec device detected on USB."
    echo "     Plug the camera in (USB data+power cable) and re-run."
    echo "     Continuing setup anyway — you can plug it in afterwards."
fi
echo

# ---- 1. apt deps ----
echo "==> Installing build dependencies"
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    libusb-1.0-0-dev libudev-dev \
    freeglut3-dev \
    openjdk-17-jdk-headless \
    python3-dev python3-pip python3-venv

# ---- 2. clone + build OpenNI 2 ----
WORK_DIR="${HOME}/orbbec-build"
mkdir -p "${WORK_DIR}"

if [[ ! -d "${WORK_DIR}/OpenNI2" ]]; then
    echo "==> Cloning Orbbec's OpenNI 2 fork"
    git clone --depth 1 https://github.com/orbbec/OpenNI2.git "${WORK_DIR}/OpenNI2"
fi

cd "${WORK_DIR}/OpenNI2"
echo "==> Building OpenNI 2 for Arm64 (5-10 min on Pi 5)"
make PLATFORM=Arm64 -j"$(nproc)" release

REDIST="${WORK_DIR}/OpenNI2/Bin/Arm64-Release"
if [[ ! -f "${REDIST}/libOpenNI2.so" ]]; then
    echo "  ❌ Build finished but libOpenNI2.so not found at ${REDIST}"
    echo "     Check the build log above."
    exit 1
fi
echo "  ✅ Built. Redist at: ${REDIST}"

# ---- 3. udev rules ----
echo "==> Installing udev rules"
sudo tee /etc/udev/rules.d/55-orbbec.rules > /dev/null <<'EOF'
# Orbbec / Astra family
SUBSYSTEM=="usb", ATTRS{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"
# Older Orbbec vendor ID (some Astra Pro units)
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d27", MODE="0666", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger

# ---- 4. Python wrapper ----
echo "==> Installing Python bindings (openni + numpy)"
pip install --user openni numpy

# ---- 5. env var ----
ENV_LINE="export OPENNI2_REDIST=\"${REDIST}\""
if ! grep -qF "OPENNI2_REDIST" "${HOME}/.bashrc" 2>/dev/null; then
    echo "${ENV_LINE}" >> "${HOME}/.bashrc"
    echo "==> Added OPENNI2_REDIST to ~/.bashrc"
fi

echo
echo "============================================================"
echo "  ✅ Setup complete."
echo "============================================================"
echo
echo "Reload your shell or run this once:"
echo "   ${ENV_LINE}"
echo
echo "Then UNPLUG and REPLUG the Astra (udev rules need a fresh attach), and:"
echo "   python3 scripts/test_astra.py"
echo
echo "If that prints depth frames, you're ready to run the SDK example:"
echo "   python3 examples/tier1_with_astra/main.py"
echo
