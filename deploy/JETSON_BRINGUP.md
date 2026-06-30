# Jetson Orin (aarch64) bringup — real-world notes

The stock `install_robot.sh --install-python` path does NOT work on a Unitree
Go2/G1 Jetson Orin (L4T focal, aarch64): the **deadsnakes PPA has no arm64
builds**, and **`pyrealsense2` / `cyclonedds` have no aarch64 PyPI wheels**. This
is the procedure that actually works, using a userspace conda env (no system
Python changes). Verified on an Orin NX, JetPack R35.3.1, RealSense D435i.

## 1. Python 3.10 + camera + vision stack (conda, no sudo)
```bash
# Miniforge gives aarch64 Python 3.10 binaries (deadsnakes can't).
curl -fL -o ~/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash ~/miniforge.sh -b -p ~/miniforge3
# conda-forge HAS prebuilt aarch64 pyrealsense2 -> no librealsense source build.
~/miniforge3/bin/conda create -y -n kovio -c conda-forge \
  python=3.10 pyrealsense2 onnxruntime opencv numpy
~/miniforge3/envs/kovio/bin/pip install .   # the SDK (NOT [jetson]: pyrealsense2 is from conda)
```

**Runtime gotcha (always needed):** cv2/onnxruntime fail to import with
`libgomp.so.1: cannot allocate memory in static TLS block`. Fix by preloading it:
```bash
export LD_PRELOAD=$HOME/miniforge3/envs/kovio/lib/libgomp.so.1
```

## 2. YOLO models — the SDK's URLs are dead
`detectors.py` `_MODELS` point at `ultralytics/assets` `*.onnx` which **404**
(that release ships `.pt`, not `.onnx`). Export them once:
```bash
~/miniforge3/envs/kovio/bin/pip install ultralytics onnx
# torch on aarch64 hits the same static-TLS error -> raise the surplus AND
# import torch BEFORE ultralytics (which pulls cv2 and eats the TLS budget):
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4000000 \
LD_PRELOAD=$HOME/miniforge3/envs/kovio/lib/libgomp.so.1 \
~/miniforge3/envs/kovio/bin/python - <<'PY'
import torch  # FIRST
from ultralytics import YOLO
for n in ("yolov8n","yolov8n-pose"):
    YOLO(f"{n}.pt").export(format="onnx", imgsz=640)   # .pt DOES exist at ultralytics v8.2.0
PY
mkdir -p ~/.cache/kovio/models && cp yolov8n*.onnx ~/.cache/kovio/models/
```
TODO (productize): host these onnx as a release asset and fix `_MODELS`.

## 3. Lidar (Livox via Unitree DDS) — optional, for crowd metrics
```bash
conda install -y -n kovio -c conda-forge cyclonedds         # the C library
export CYCLONEDDS_HOME=$HOME/miniforge3/envs/kovio
~/miniforge3/envs/kovio/bin/pip install cyclonedds           # python binding (11.x)
# unitree pins an OLD cyclonedds; install without its deps (works vs 11.x):
~/miniforge3/envs/kovio/bin/pip install --no-deps \
  git+https://github.com/unitreerobotics/unitree_sdk2_python.git
```
The adapter reads DDS topic `rt/utlidar/cloud` on `eth0` (the robot's internal
`192.168.123.x` net). **The robot only publishes this topic when its main
control service is active** — an idle/damped robot publishes nothing. Probe:
```bash
# 0 messages => robot not publishing (bring the robot up / check topic name).
```

## 4. Always-on service
Use `deploy/kovio-agent-jetson.service` (conda env + the env fixes above). It
reads creds from `/etc/kovio/kovio.env` (KOVIO_API_URL, KOVIO_API_KEY,
KOVIO_ROBOT_ID, KOVIO_PERCEPTION=rich, KOVIO_SCREEN=logger — `logger`, NOT
`browser`, on a headless unit). Install:
```bash
sudo install -m600 kovio.env /etc/kovio/kovio.env
sudo install -m644 deploy/kovio-agent-jetson.service /etc/systemd/system/kovio-agent.service
sudo systemctl daemon-reload && sudo systemctl enable --now kovio-agent
journalctl -u kovio-agent -f
```
Only ONE process may hold the RealSense at a time.
