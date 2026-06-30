#!/usr/bin/env bash
#
# install_robot.sh — install the Kovio agent as an always-on systemd service.
#
# Tested target: Unitree Go2 / NVIDIA Jetson Orin (L4T focal, aarch64) with an
# Intel RealSense depth camera and the Go2 Livox lidar. Idempotent: safe to
# re-run to upgrade. Run from a checkout of kovio-py:
#
#     sudo deploy/install_robot.sh [--install-python] [--no-models]
#
# Steps: ensure Python >=3.10, build a venv at /opt/kovio/venv, install the SDK
# with the [jetson] extra, pre-cache the YOLO models, drop config in /etc/kovio,
# then enable + start kovio-agent.service so perception runs on every boot.
set -euo pipefail

VENV=/opt/kovio/venv
ENV_DIR=/etc/kovio
STATE_DIR=/var/lib/kovio
SERVICE=kovio-agent.service
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-unitree}"

INSTALL_PYTHON=0
FETCH_MODELS=1
for arg in "$@"; do
  case "$arg" in
    --install-python) INSTALL_PYTHON=1 ;;
    --no-models)      FETCH_MODELS=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Please run with sudo (it installs a system service)." >&2
  exit 1
fi

log() { printf '\033[1;36m[kovio]\033[0m %s\n' "$*"; }

# --- 1. Python >=3.10 -------------------------------------------------------
# The SDK targets 3.10+, but L4T focal ships 3.8. Find a good interpreter, or
# install one from deadsnakes when asked.
find_python() {
  for p in python3.12 python3.11 python3.10; do
    if command -v "$p" >/dev/null 2>&1; then echo "$p"; return 0; fi
  done
  return 1
}

PY="$(find_python || true)"
if [[ -z "${PY}" ]]; then
  if [[ $INSTALL_PYTHON -eq 1 ]]; then
    log "Installing Python 3.10 from deadsnakes ..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.10 python3.10-venv python3.10-dev
    PY=python3.10
  else
    echo "No python3.10+ found. Re-run with --install-python, or install it yourself." >&2
    exit 1
  fi
fi
log "Using interpreter: $($PY --version)"

# --- 2. System libraries ----------------------------------------------------
log "Installing system dependencies ..."
apt-get update
apt-get install -y \
  "${PY}-venv" \
  libusb-1.0-0 \
  libgl1 libglib2.0-0    # OpenCV runtime libs (headless still needs libGL/glib)

# --- 3. Virtualenv + SDK ----------------------------------------------------
log "Creating venv at ${VENV} ..."
mkdir -p "$(dirname "$VENV")"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel

log "Installing kovio[jetson] from ${REPO_ROOT} ..."
# pyrealsense2 wheels are not always on PyPI for aarch64. If this fails, install
# librealsense + its python bindings system-wide (the robot already ships
# librealsense2.so via ROS) and re-run with the camera deps satisfied.
if ! "$VENV/bin/pip" install "${REPO_ROOT}[jetson]"; then
  log "WARN: [jetson] extra failed (likely pyrealsense2 on aarch64)."
  log "      Installing core + vision deps without pyrealsense2; provide it separately."
  "$VENV/bin/pip" install "${REPO_ROOT}[dashboard]" \
    numpy onnxruntime opencv-python-headless
fi

# --- 4. Pre-cache detection models -----------------------------------------
if [[ $FETCH_MODELS -eq 1 ]]; then
  log "Pre-downloading YOLO detection + pose models ..."
  sudo -u "$SERVICE_USER" "$VENV/bin/python" - <<'PY'
from kovio.adapters import detectors as d
for m in ("yolov8n", "yolov8n-pose"):
    print("  cached", d.ensure_model(m))
PY
fi

# --- 5. Config + state ------------------------------------------------------
mkdir -p "$ENV_DIR" "$STATE_DIR"
chown "$SERVICE_USER" "$STATE_DIR"
if [[ ! -f "$ENV_DIR/kovio.env" ]]; then
  install -m 600 "$REPO_ROOT/deploy/kovio.env.example" "$ENV_DIR/kovio.env"
  log "Wrote ${ENV_DIR}/kovio.env — EDIT IT to set KOVIO_API_KEY and KOVIO_ROBOT_ID."
else
  log "Keeping existing ${ENV_DIR}/kovio.env"
fi

# --- 6. systemd service -----------------------------------------------------
log "Installing ${SERVICE} ..."
# Point ExecStart at this venv (the unit file assumes /opt/kovio/venv).
install -m 644 "$REPO_ROOT/deploy/$SERVICE" "/etc/systemd/system/$SERVICE"
systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

log "Done. The Kovio agent is enabled and will start on every boot."
log "Status:  systemctl status $SERVICE"
log "Logs:    journalctl -u $SERVICE -f"
log "NOTE:    set your API key in ${ENV_DIR}/kovio.env, then: systemctl restart $SERVICE"
