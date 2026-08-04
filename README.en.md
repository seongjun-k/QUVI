# QUVI (QUality VIsion)

[![한국어](https://img.shields.io/badge/lang-한국어-blue)](README.md)
[![English](https://img.shields.io/badge/lang-English-red)](README.en.md)
[![日本語](https://img.shields.io/badge/lang-日本語-white)](README.ja.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/MCU-ESP32--S3-E7352C)
![Docker](https://img.shields.io/badge/Docker-quvi--dev-2496ED?logo=docker&logoColor=white)
[![Web Demo](https://img.shields.io/badge/Web_Demo-Live-brightgreen)](https://seongjun-k.github.io/QUVI/)

**An unmanned inspection and sorting system for 3D-printer output, powered by an AI vision robot arm**

> "Seeing is Quality"

▶ **[Try the Web Demo](https://seongjun-k.github.io/QUVI/)** — experience the pass/fail inspection cycles, replayed from real hardware recordings, with no hardware required. Works on mobile too.

---

## Problem Statement

As the price of 3D printers has fallen, more small operators have built businesses around print-on-demand, prototyping, and low-volume production. But in FDM printing, failure isn't an accident you can eliminate — it's a constant that persists as a probability. Desktop FDM failure rates in shared print environments have been measured at 41.1% (Song & Telenko, *Material and energy loss due to human and machine error in commercial FDM printers*, Journal of Cleaner Production, 2017), and even skilled operators and print farms only push it down to 2–8%, never to zero.

The real loss isn't the failure itself — it's **the time nobody notices the failure**. For an operator running a printer overnight, a failure at 3 a.m. means the bed stays occupied until morning, and those few hours wipe out an entire day's cycle. The loss shows up not in inspection labor cost, but in printer utilization.

Industrial vision inspection equipment costs tens of thousands of dollars and requires re-preparing drawings and labeling data every time the product changes — a poor fit for an environment where every order produces something different. QUVI automates the entire **grasping (imitation-learning robot arm) → inspection (machine vision) → sorting (automatic stacking)** pipeline with roughly 600,000 KRW worth of parts. Once a finished bed comes in, a robot arm grasps the part, captures it from 4 angles on a turntable inside the inspection chamber, judges pass/fail, and stacks it at the PASS or FAIL station — with no human hands involved at any point.

The entire process is autonomously controlled by a finite state machine (FSM) orchestrator. Grasping is performed via LeRobot ACT imitation learning, and inspection combines surface-feature rule-based judgment with PatchCore anomaly detection in a hybrid decision scheme.

---

## Key Numbers

| Metric | Value |
| :--- | :--- |
| Total parts cost | About 600,000 KRW — under 1/50 the cost of tens-of-thousands-of-dollars industrial vision inspection equipment |
| Cycle time | About 1 minute 10 seconds from grasp start to sorted stacking (measured from real hardware recordings; 67s for a good part, 64s for a defective part) |
| Real-hardware inspection cycles | 133 completed cumulative (from inspection logs, 2026-06-26 to 07-30) |
| Grasp success rate | 11 of 13 real-hardware attempts succeeded, 84.6% (measured 2026-07-30) |
| Rule-ML verdict agreement rate | 100% over the most recent 13 cases (since 2026-07-15) — 88.3% across the full 103 cases including early validation |

---

## Architecture

### System Components

| Component | Technical Spec | Role & Features |
| :--- | :--- | :--- |
| **Main Controller** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | Orchestrates all nodes and controls the finite state machine (FSM) |
| **Sub-Controller** | ESP32-S3 + TB6600 (micro-ROS) | Drives the linear rail (stepper motor), turntable angle, and lighting LEDs |
| **Robot Arm** | ROBOTIS OMX manipulator (follower) | Dynamixel (XL430/XL330)-based, supports leader-follower teleoperation |
| **Cameras** | 2× USB UVC cameras | Side camera (Zone 1: grasping area), inspection camera (Zone 2: quality inspection chamber) |
| **AI & Algorithms** | LeRobot ACT + OpenCV + PatchCore | Imitation-learning grasp control, surface-feature rule-based judgment, ML anomaly detection (hybrid decision) |

### Inspection Method

The inspection camera captures images from 4 turntable angles (0°/90°/180°/270°), analyzes them along two tracks, and **combines both results** into the final verdict.

1. **Surface-feature rule-based judgment** — determines PASS/FAIL using the worst case across all angles
   * Solidity (contour area vs. convex hull — detects warping)
   * Area ratio (vs. reference image — detects under-/over-extrusion). To offset the object-to-camera distance change caused by turntable eccentricity, comparison uses **area/width² distance-invariant normalization**
   * Hole count and hole area ratio (detects layer separation) — FAIL starting from a single hole
   * Texture variance (Laplacian — detects stringing)

   Judgment thresholds are managed in `src/quvi_inspect/config/inspect_params.yaml` and kept in sync with the HMI display thresholds (`dashboard.js` THRESHOLDS).
2. **PatchCore anomaly detection** — computes an anomaly score using a WideResNet50-backbone, per-angle memory bank. Since it trains on known-good images alone, no defect samples or labeling are required.

**Hybrid decision rule** — a part passes only when both the rule-based judgment and the ML model pass it. If the rules pass a part but the ML model explicitly fails it, the final verdict flips to FAIL. The reverse never happens: the ML model cannot turn a rule-based FAIL into a PASS. In other words, ML only tightens the verdict and never increases false accepts. When no ML model is loaded, the system automatically falls back to rule-based judgment alone.

Each verdict also publishes `anomaly_score_worst` (worst anomaly score across the 4 angles) and `ml_passed` (-1 = unused / 0 = FAIL / 1 = PASS), shown on the HMI dashboard as the ML anomaly score.

Reference images are generated by capturing known-good parts using the HMI's reference-image capture mode. The area-ratio rule still relies on them, so reference images remain necessary for now. That said, in low-volume production where the same part is printed repeatedly, that reference image effectively is the correct answer, so it isn't a real constraint — removing the reference-image dependency entirely remains future work.

### ROS 2 Package and Node Structure

| Package | Executable Node | Main Role |
| :--- | :--- | :--- |
| **`quvi_robot_control`** | `main_orchestrator_node` | Controls the full autonomous sequence FSM (grasp → seat in chamber → inspect → sort → return home) |
| | `robot_control_node` | Robot arm Dynamixel control, LeRobot ACT grasp inference, rail/turntable command relay, E-STOP handling |
| **`quvi_inspect`** | `inspect_node` | 4-angle surface-feature analysis + PatchCore anomaly detection hybrid pass/fail judgment, inspection log storage, reference-image / ML dataset capture modes |
| **`quvi_hmi`** | `hmi_node` | **Flask + SocketIO-based real-time web dashboard** (status monitoring, MJPEG streaming, manual control) |
| **`quvi_msgs`** | - | Custom messages (`SystemStatus`, `InspectionResult`, `GraspGoal`, `MotorStatus`, etc.) |
| **`quvi_bringup`** | - | System launch files (`full_system.launch.py`, `vision_pipeline.launch.py`) |

Topic names are centrally managed in `quvi_robot_control/topics.py`.

### Web HMI Key Features (Dashboard)

* **Real-time system status monitoring**
  * Robot joint angle visualization (`/robot/joint_states` live gauges)
  * Linear rail track motion (station map: INSPECT / PASS / FAIL / BED)
  * Turntable compass dial, FSM stage flow visualization
  * Inspection history & statistics (PASS/FAIL counts)
* **Real-time MJPEG video streaming** — side camera, inspection camera, inspection debug view (4-angle tile + judgment overlay)
* **Manual control**
  * Start/stop the autonomous sequence, **emergency stop (E-STOP)**, and reset
  * Leader-follower teleoperation toggle, manual rail/turntable/LED control
  * ACT model scan & selection (hot-swap), device mapping (cameras, serial ports) configuration and restart
  * Reference-image capture mode, ML known-good dataset capture mode

### Project Folder Structure

```
QUVI/
├── docker/                  # Docker development environment (Dockerfile, compose)
├── firmware/                # ESP32-S3 rail/turntable firmware (PlatformIO, micro-ROS)
├── lerobot/                 # LeRobot submodule (OMX-supporting branch)
├── src/                     # ROS 2 source
│   ├── quvi_msgs/           # Custom message definitions
│   ├── quvi_bringup/        # Launch files
│   ├── quvi_robot_control/  # Robot arm, FSM orchestrator, shared utils/topics
│   ├── quvi_inspect/        # Pass/fail judgment + PatchCore anomaly detection package
│   └── quvi_hmi/            # Flask + SocketIO web dashboard
├── data/                    # Reference images, inspection logs, ML datasets/models, device config
├── scripts/                 # ACT recording/training, anomaly detection training, calibration/diagnostic scripts
├── tests/                   # pytest logic tests
└── docs/                    # Technical design documents
```

---

## Tech Stack

* **Operating System**: Ubuntu 24.04 LTS
* **Middleware**: ROS 2 Jazzy + micro-ROS (ESP32-S3)
* **Vision & AI**: OpenCV, PyTorch (numpy pinned <2), Hugging Face LeRobot (ACT), PatchCore (WideResNet50)
* **Web HMI**: Flask, Flask-SocketIO (threading mode), Vanilla JS, HTML5/CSS3 (Industrial Dark Theme)
* **Embedded**: ESP32-S3, TB6600, Dynamixel SDK (Protocol 2.0), PlatformIO

---

## How to Run (Docker Environment)

> **If you do not have the hardware**, use the [web demo](https://seongjun-k.github.io/QUVI/) instead of the steps below. It runs in the browser with no installation and replays a full inspection cycle recorded on the real machine.
> The steps below assume a physical setup (robot arm, rail, cameras) on an Ubuntu 24.04 host with Docker.

### 1. Clone the repository and initialize submodules
```bash
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI
git submodule update --init --recursive
```

### 2. Install device udev rules (once, when hardware is connected)
The ESP32-S3 and the two cameras must be bound to fixed symlinks (`/dev/ttyESP32`, `/dev/sidecam`, `/dev/fixed_cam`) to match the default device paths in the launch file.
```bash
sudo cp scripts/99-esp32.rules scripts/99-uvc-cameras.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
> The camera symlinks are keyed to the physical USB hub port, so they will not appear if the hub is plugged into a different port. In that case, select the raw node directly in the HMI device settings.

### 3. Set up the Docker environment
```bash
cd docker
docker compose build
docker compose up -d
```

### 4. Build and run (from the host)
```bash
./build.sh   # Start the container + colcon build --symlink-install
./run.sh     # Run full_system.launch.py
```
* Open `http://localhost:5000` in a web browser → HMI dashboard.
* For manual execution: inside the container, run `ros2 launch quvi_bringup full_system.launch.py`

### 5. Testing
```bash
docker exec quvi-dev bash -c "cd /workspace && python3 -m pytest tests/ -q"
```

---

## LeRobot ACT Imitation Learning Guide

Robot arm grasping (Zone 1) is performed via visuomotor control based on LeRobot ACT (Action Chunking with Transformers) imitation learning.

> **What is imitation learning?** Instead of programming motions with coordinates and rules, a human demonstrates the grasp several times with a leader arm, and a neural network learns from the paired camera images and joint positions recorded during those demonstrations. The trained model then generates joint motions on its own from camera input alone, so it can grasp parts even when their position varies slightly — no re-teaching required. ACT is an imitation-learning policy that predicts motions in multi-step chunks rather than one step at a time, which makes the movement smoother and more stable.

### 1. Teleoperation data collection
Record leader-follower demonstration data using the helper script (runs on either host or container).
```bash
./scripts/act_record.sh <HF_USER> <num_episodes> <episode_duration_sec>
```

### 2. ACT model training
```bash
./scripts/act_train.sh <HF_USER>
```
Falls back to CPU with a warning if CUDA is unavailable.

### 3. Inference and deployment
Runtime model swapping is available via model scan/selection on the HMI dashboard. **The last selected model is saved to `data/act_last_model.json` and automatically loaded/activated on restart.** If no saved selection exists, the default path from the `act_model_path` parameter is used.

---

## PatchCore Anomaly Detection Training Pipeline

```bash
# 0. Collect known-good images via HMI dataset capture mode or existing PASS inspection logs
python3 scripts/build_anomaly_dataset.py     # PASS logs → organize per-angle raw/ + generate review sheets
# (A human reviews review_sheet_{angle}.png and removes any defective images mixed into raw/)

# 1. Train per-angle memory bank + compute thresholds
python3 scripts/train_anomaly_bank.py        # → data/models/bank_{angle}.pt, thresholds.json

# 2. Report rule-vs-ML agreement rate (decision-confidence check)
python3 scripts/shadow_report.py
```

Toggled via the launch argument `anomaly_enabled` (default true). If the model file is missing or fails to load, it is automatically disabled and only rule-based judgment is used.

---

## ESP32-S3 Firmware Build & Flash

The ESP32-S3 firmware responsible for the linear rail, turntable, and LEDs is a PlatformIO project (`firmware/quvi_esp32_firmware/`). The ESP32-S3 connects via a CH340 bridge (`/dev/ttyESP32` udev symlink, `scripts/99-esp32.rules`) and uploads with automatic reset, no boot-button press needed.

```bash
# Run on the host. Stop the micro-ROS agent first if it is holding the port.
cd firmware/quvi_esp32_firmware
pio run                                        # Compile
pio run -t upload --upload-port /dev/ttyESP32  # Flash
```

Hardware calibration constants — homing (3-stage: coarse search → backoff → precision search), rail coordinate system, soft limits, etc. — are managed in `Config.h`.

---

## Future Plans

The next goal isn't inspection automation by itself, but **an unmanned loop where print → retrieve → inspect → reprint all runs without a person**. The only step a human still does is loading the bed, and closing that loop takes three additions.

| To add | How |
| :--- | :--- |
| Bed loading/unloading | Same position, same repeated trajectory — handled not by the robot arm but by a dedicated mechanism at the level already used for the rail |
| Reprint instruction | Feed the defect count from inspection straight to the printer's public API (OctoPrint, Moonraker) |
| Defective-part retrieval | Grind down rejected parts and feed them back into filament |

Planned hardware extensions include a top-view camera (grasp accuracy), a magnetic tool changer (automatic gripper swapping), and proximity-sensing stops (a safety layer on top of the E-STOP).

---

## Team

- **Seoul Robotics High School Capstone Team "All-Rounder"** — a club built around competitions. QUVI started as a capstone project and has continued through to entry in the National Meister High School Star Project.

---

## AI Usage Disclosure

In the interest of transparency, we disclose the AI, open-source software, and external assistance used in building this project.

- **Development-assistance AI**: Anthropic **Claude Code** (Claude Fable 5/Opus/Sonnet/Haiku models), OpenAI **Codex**, Google **Antigravity** — assisted with code writing, review, and debugging. Final design decisions and hardware validation were performed by the team directly
- **AI models embedded in the product**: **ACT** grasp policy (trained on our own teleoperation data collected via LeRobot), **PatchCore** anomaly detection (WideResNet50 backbone, memory bank built from our own known-good images)
- **Open source** (with license types): ROS 2 Jazzy (Apache-2.0), micro-ROS (Apache-2.0), LeRobot (Apache-2.0), OpenCV (Apache-2.0), PyTorch (BSD-3-Clause), Flask (BSD-3-Clause), Flask-SocketIO (MIT), NumPy (BSD-3-Clause), DYNAMIXEL SDK (Apache-2.0), PlatformIO (Apache-2.0) — see [References](#references) below for detailed sources
- **Web demo analytics**: Google Analytics 4 — counts visits and completed inspection cycles on the public web demo. No personally identifiable information is collected
- **External consulting**: None (no consultation from outside institutions or companies beyond our supervising teacher)

---

## References

- **LeRobot** — Hugging Face's robot imitation-learning framework. This project uses the OMX-supporting fork [ROBOTIS-GIT/lerobot](https://github.com/ROBOTIS-GIT/lerobot) as a submodule (upstream: [huggingface/lerobot](https://github.com/huggingface/lerobot))
- **ACT** — Zhao et al., ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705) (RSS 2023) — grasp imitation-learning policy
- **ROBOTIS OpenMANIPULATOR-X (OMX)** — [ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) — robot arm hardware and [DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- **PatchCore** — Roth et al., ["Towards Total Recall in Industrial Anomaly Detection"](https://arxiv.org/abs/2106.08265) (CVPR 2022) — anomaly detection algorithm
- **micro-ROS** — [micro-ROS-Agent](https://github.com/micro-ROS/micro-ROS-Agent), [micro_ros_platformio](https://github.com/micro-ROS/micro_ros_platformio) — ESP32-S3 ↔ ROS 2 communication

---

## License

This project is licensed under the [MIT License](LICENSE).
