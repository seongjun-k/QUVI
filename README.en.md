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

## Mission & Metric

**Mission** — For makers and small workshops or print shops running printers for long, unattended jobs, we eliminate the waste of materials and time caused by catching a defect too late — by having a robot photograph the print the moment it finishes and letting AI judge whether it failed.

**North-star metric** — The time from the moment a print finishes to the moment judgment and sorting are complete. Following earlier review feedback, we redefined the north star as a measurable interval and measured it on real hardware. What used to sit unattended for tens of minutes to several hours until someone came to check is now down to a **current median of 64 seconds** (23 real-hardware runs, -10% from 71 seconds in September). The next goal is to shorten the longest inspection step (24 seconds) and bring the total to 60 seconds or less.

Major failures (filament runout, collisions) are stopped by the printer itself, while defects in prints that run to completion (layer shifting, warping, stringing) are caught by QUVI. QUVI starts an inspection only when the Klipper state is `complete`.

---

## Finals Period Improvements (2026-10-21 ~ 11-10)

These values were re-counted from real-hardware logs (4,356 ROS log files, 203 grasp trajectories, 268 inspection records) and measurement scripts. The before/after reference points are 11-08 19:10 for grasping and training (when we switched to the new VLA model) and 11-09 18:55 for inspection (when automatic product selection was applied). Detailed numbers and the list of raw data are in [`docs/metrics_20261110.md`](docs/metrics_20261110.md).

| Metric | Before | After | Change |
| :--- | ---: | ---: | ---: |
| **North star: print complete → classification complete time (median)** | 71 s (66 runs) | **64 s** (23 runs) | **-7 s (-10%)** |
| Grasp motion completion rate | 53% (37/70) | **77%** (34/44) | **+24 pp** |
| Error / emergency-stop rate | 26% (18/70) | **5%** (2/44) | **-21 pp** |
| Empty-hand carry rate | 20% (1/5) | **0%** (0/16) | **-20 pp** |
| VLA training prediction error (mean absolute joint error) | 0.1457 | **0.0409** | **-72%** |
| Inspection verdict completion rate | 30% (43/143) | **100%** (7/7) | **+70 pp** |
| Inspection verdict-unavailable count | 100 | **0** | **-100** |
| False-positive rate on good parts (1.3x brightness, box) | 46% | **6%** (2/32) | **-40 pp** |
| Setup time for a new product | not measured | **about 15 min** | first measurement |
| Real-user tests | 0 | **1** (Geekble maker Sudragon, 11-10) | +1 |

**Real-user test — Geekble maker Sudragon (2026-11-10)**

We had him use the QUVI prototype directly and interviewed him. He said he would use it if it were commercialized. (Quotes translated from Korean.)

* "Most existing solutions took the approach of swapping the bed in an open chamber, so I'm very impressed that you thought about a solution that works for both enclosed chambers and open-frame printers. I'm especially looking forward to it for cases where the same part has to be printed many times."
* "I hope the solution actually gets commercialized. I think it would be very convenient."
* "From a maker's point of view, I liked that problems in prints can be resolved quickly."

| Feedback | How we reflect it |
| :--- | :--- |
| Works for both enclosed and open-frame printers | Confirmed that bed swapping cannot be used on enclosed chambers, so we present it as a differentiator and comparison basis |
| Looking forward to repeated printing of the same part | Position repeat-production sites as the core use case, citing the 64-second continuous cycle and about 15 minutes to prepare a new product |
| Resolving print problems quickly | Keep measuring with the north-star metric (print complete → classification complete, 64 s) |

**What we changed**

* **Grasp model retraining** — Collected 115 new demonstrations and refined them to 112 (9,296 frames), then trained SmolVLA for 15,000 steps (`smolvla_1108_box2c_new15k`, loss 0.542 → 0.079). Matched the home pose to the training start pose and matched the VLA execution time to the demonstration length of 5.5 seconds
* **In-house collection and training pipeline** — Consolidated into a web collection tool (record, replay, review) → LeRobot v3.0 conversion → training script, with no dependency on the external Cyclo stack. Collection and inference use the same camera exposure and white-balance profile
* **Stability** — Physical emergency-stop button (stops while pressed, resumes on release), automatic reconnection when the robot arm USB drops, automatic restart of the inspection camera node, USB bandwidth time-sharing for the three cameras and automatic recovery on hub reconnection
* **Inspection** — Registered multi-product inspection assets (box, cube) and automatic product selection at startup, bringing verdict-unavailable cases (70 with no product selected, 30 with insufficient reference images) to 0. A new product is prepared by photographing 160 good parts (13 minutes) and training (1-2 minutes), with no defect samples or labeling required
* **Hardware tuning** — Turntable placement pose raised 8 mm and moved 10 mm forward, gripper grasp force reduced by 50%

**Tried but not adopted** — Inspection brightness normalization (11-02) cut the box 1.3x-brightness false-positive rate from 46% to 4%, but we rejected it because misses increased: scratch detection fell from 92% to 50%, corner-defect detection from 54% to 8%, and the cube clipping false-positive rate rose from 0% to 54% (synthetic defects, 24 images per product). Changing the playback rate from 30 to 15 Hz (September experiment) was rejected because jitter doubled. We confirmed the cause of the jitter is oscillation in the policy output itself, not the playback rate, and the next step is action smoothing.

**Measurement limits**

* The grasp motion completion rate is the share of runs that executed the grasp sequence to the end. The gripper empty-hand judgment is still in shadow mode and is not reflected; whether a part was actually carried was checked separately through inspection results (all 16 cycles with a judgment record had a print present).
* The training prediction error is an open-loop evaluation value on the current model's training data.
* The post-fix sample for the inspection verdict completion rate is 7. The September value of the north star is from an earlier tally, and in demos the HMI start button substitutes for the print-complete signal.
* The inspection false-positive rate is affected by lighting brightness, so the inspection chamber lighting must be kept constant.

---

## Problem Statement

As the price of 3D printers has fallen, more small operators have built businesses around print-on-demand, prototyping, and low-volume production. But in FDM printing, failure isn't an accident you can eliminate — it's a constant that persists as a probability. Desktop FDM failure rates in shared print environments have been measured at 41.1% (Song & Telenko, *Material and energy loss due to human and machine error in commercial FDM printers*, Journal of Cleaner Production, 2017), and even skilled operators and print farms only push it down to 2–8%, never to zero.

The real loss isn't the failure itself — it's **the time after a print finishes when nobody checks it**. A defect in a print that ran to completion can only be found by a person looking at it, and for an operator running a printer overnight, a print that finishes at dawn means the bed stays occupied until morning, and those few hours wipe out an entire day's cycle. The loss shows up not in inspection labor cost, but in printer utilization.

Industrial vision inspection equipment costs tens of thousands of dollars and requires re-preparing drawings and labeling data every time the product changes — a poor fit for an environment where every order produces something different. QUVI automates the entire **grasping (imitation-learning robot arm) → inspection (machine vision) → sorting (automatic stacking)** pipeline with roughly 600,000 KRW worth of parts. Once a finished bed comes in, a robot arm grasps the part, captures it from 4 angles on a turntable inside the inspection chamber, judges pass/fail, and stacks it at the PASS or FAIL station — with no human hands involved at any point.

The entire process is autonomously controlled by a finite state machine (FSM) orchestrator. Grasping is performed via SmolVLA (vision-language-action model) imitation learning, and inspection combines surface-feature rule-based judgment with PatchCore anomaly detection in a hybrid decision scheme. The sequence can be started by the print-complete signal from a Klipper printer.

---

## Key Numbers (First Tally, as of 2026-07-30)

For the improvement numbers from the finals period, see the [Finals Period Improvements](#finals-period-improvements-2026-10-21--11-10) table above. The grasp success rate below is for the earlier ACT policy and a different counting basis, so it cannot be compared directly with the grasp motion completion rate in the table above.

| Metric | Value |
| :--- | :--- |
| Total parts cost | About 600,000 KRW — under 1/50 the cost of tens-of-thousands-of-dollars industrial vision inspection equipment |
| Cycle time | About 1 minute 10 seconds from grasp start to sorted stacking (measured from real hardware recordings; 67s for a good part, 64s for a defective part) |
| Real-hardware inspection cycles | 133 completed cumulative (from inspection logs, 2026-06-26 to 07-30 20:43 KST) |
| Grasp success rate | 11 of 13 real-hardware attempts succeeded, 84.6% (measured 2026-07-30, excluding 4 tests manually stopped by a person; 11/17 = 64.7% if included) |
| Rule-ML verdict agreement rate | 100% over the most recent 13 cases (since 2026-07-15) — 88.3% across the full 103 cases including early validation (excluding 30 pre-07-08 records with no ML verdict) |

The full raw data, including records after this range, gives 137 cycles, rule-ML 95/107 = 88.8%, and grasp 15/17 = 88.2%. Raw data is in [`docs/metrics/`](docs/metrics/); recompute with `python3 scripts/recount_metrics.py`. Web demo metrics follow the [W6 GA4 report](docs/metrics/QUVI_W6_실측지표.pdf) (2026-08-31 to 09-27).

---

## Architecture

### System Components

| Component | Technical Spec | Role & Features |
| :--- | :--- | :--- |
| **Main Controller** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | Orchestrates all nodes and controls the finite state machine (FSM) |
| **Sub-Controller** | ESP32-S3 + TB6600 (micro-ROS) | Drives the linear rail (stepper motor), turntable angle, and lighting LEDs, physical emergency-stop button |
| **Robot Arm** | ROBOTIS OMX manipulator (follower) | Dynamixel (XL430/XL330)-based, supports leader-follower teleoperation |
| **Cameras** | 3× USB UVC cameras | Side camera and top-view camera (Zone 1: grasping area, VLA input), inspection camera (Zone 2: quality inspection chamber). Same USB 2.0 bus, so bandwidth is time-shared |
| **Printer Integration** | Klipper + Moonraker (KP3S) | Print-complete detection → automatic start (optional), bed ejection via the print-end macro |
| **AI & Algorithms** | LeRobot SmolVLA + OpenCV + PatchCore | Imitation-learning grasp control, surface-feature rule-based judgment, ML anomaly detection (hybrid decision) |

### Inspection Method

The inspection camera captures images from 4 turntable angles (0°/90°/180°/270°), analyzes them along two tracks, and **combines both results** into the final verdict. Inspection criteria (reference images and ML models) are registered per product, and the product is selected automatically at startup (currently 2 products: box and cube).

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
| | `robot_control_node` | Robot arm Dynamixel control, SmolVLA grasp inference (separate sidecar process), rail/turntable command relay, E-STOP handling |
| | `printer_monitor_node` | Monitors Klipper/Moonraker print status and publishes the print-complete signal (moves nothing) |
| **`quvi_inspect`** | `inspect_node` | 4-angle surface-feature analysis + PatchCore anomaly detection hybrid pass/fail judgment, per-product inspection asset management, inspection log storage, reference-image / ML dataset capture modes |
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
* **Real-time MJPEG video streaming** — side camera, top-view camera, inspection camera, inspection debug view (4-angle tile + judgment overlay)
* **Manual control**
  * Start/stop the autonomous sequence, **emergency stop (E-STOP)**, and reset (the physical button stops while pressed and resumes automatically on release)
  * Leader-follower teleoperation toggle, manual rail/turntable/LED control
  * VLA model scan & selection (hot-swap), device mapping (cameras, serial ports) configuration and restart
  * Inspection product selection and registration, reference-image capture mode, ML known-good dataset capture mode

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
├── scripts/                 # VLA training/evaluation, anomaly detection training, metrics tally, calibration/diagnostic scripts
│   ├── collect/             # VLA demonstration collection web app (record, replay, review; port 5100)
│   └── vla_sidecar/         # SmolVLA inference sidecar (separate venv with lerobot 0.6.1)
├── tests/                   # pytest logic tests
└── docs/                    # Technical design documents
```

---

## Tech Stack

* **Operating System**: Ubuntu 24.04 LTS
* **Middleware**: ROS 2 Jazzy + micro-ROS (ESP32-S3)
* **Vision & AI**: OpenCV, PyTorch (numpy pinned <2), Hugging Face LeRobot (SmolVLA), PatchCore (WideResNet50)
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
The ESP32-S3 and the three cameras must be bound to fixed symlinks (`/dev/ttyESP32`, `/dev/sidecam`, `/dev/topcam`, `/dev/fixed_cam`) to match the default device paths in the launch file.
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

## SmolVLA Imitation Learning Guide

Robot arm grasping (Zone 1) is performed via visuomotor control based on LeRobot SmolVLA imitation learning. It takes two images (side camera and top-view camera) and the joint values as input.

> **What is imitation learning?** Instead of programming motions with coordinates and rules, a human demonstrates the grasp several times with a leader arm, and a neural network learns from the paired camera images and joint positions recorded during those demonstrations. The trained model then generates joint motions on its own from camera input alone, so it can grasp parts even when their position varies slightly — no re-teaching required. SmolVLA is a compact VLA policy that adds an action-output head on top of a pretrained vision-language model, and it can be fine-tuned with a small amount of demonstration data.

The main Python in quvi-dev is a lerobot 0.3.4 / numpy<2 environment, so SmolVLA (lerobot 0.6.1) runs inference in a sidecar process (`scripts/vla_sidecar/`) in a separate venv.

### 1. Demonstration data collection (web collection tool)
After turning on teleoperation in the HMI, record and replay episodes in the collection web app (`http://localhost:5100`) and review which ones to use. The recording start pose is matched to the average pose of the training data.
```bash
# On the host: create the conversion-tool venv (once)
docker exec quvi-dev bash /workspace/scripts/collect/setup_tools_venv.sh
# Inside the quvi-dev container: run the collection app
source /opt/ros/jazzy/setup.bash
PYTHONPATH=/workspace/src/quvi_robot_control:$PYTHONPATH python3 /workspace/scripts/collect/app.py
```

### 2. LeRobot v3.0 dataset conversion
Resample the raw collection (`data/episodes_raw/<session>`) to 15 fps, convert it to a LeRobot v3.0 dataset, and check its quality. Run it from the collection app or directly.
```bash
# Run inside the quvi-dev container
/workspace/data/collect_tools/venv/bin/python /workspace/scripts/build_lerobot_dataset.py \
    /workspace/data/episodes_raw/<session> --out /workspace/data/datasets/<name>
```

### 3. SmolVLA training
```bash
# Run on the host (it uses nearly all of the 8GB VRAM, so shut QUVI down first)
./scripts/smolvla_train.sh <dataset name> <output name> [steps] [batch]
# To continue training from a trained model: BASE_MODEL=<path under data/models> ./scripts/smolvla_train.sh ...
```
Image augmentation is on by default, and `scripts/eval_vla_open_loop.py` evaluates the prediction error against recorded data and the jitter (reversal rate).

### 4. Inference and deployment
Runtime model swapping is available via model scan/selection on the HMI dashboard. **The last selected model is saved to `data/act_last_model.json` and automatically loaded/activated on restart.** If no saved selection exists, the default path from the `act_model_path` parameter is used. Scripts for the earlier ACT policy (`act_record.sh`, `act_train.sh`) also remain.

---

## PatchCore Anomaly Detection Training Pipeline

```bash
# 0. Collect known-good images via HMI dataset capture mode or existing PASS inspection logs
python3 scripts/build_anomaly_dataset.py     # PASS logs → organize per-angle raw/ + generate review sheets
# (A human reviews review_sheet_{angle}.png and removes any defective images mixed into raw/)

# 1. Train per-angle memory bank + compute thresholds
python3 scripts/train_anomaly_bank.py --product <product>   # → data/inspection_products/<product>/models/bank_{angle}.pt, thresholds.json

# 2. Report rule-vs-ML agreement rate (decision-confidence check)
python3 scripts/shadow_report.py
```

Toggled via the launch argument `anomaly_enabled`. If the model file is missing or fails to load, it is automatically disabled and only rule-based judgment is used.

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

The next goal isn't inspection automation by itself, but **an unmanned loop where print → retrieve → inspect → reprint all runs without a person**. Printer integration is already in the code, and because it makes a real printer move, it is turned off by default.

| Step | Status |
| :--- | :--- |
| Print-complete detection → automatic start | Implemented (`printer_monitor_node` + Moonraker, `auto_start_on_print_done` default false) |
| Reprint instruction | Implemented — requests a reprint of the last gcode after a FAIL verdict (`auto_reprint_on_fail` default false) |
| Bed loading/unloading | Planned — same position, same repeated trajectory, so handled not by the robot arm but by a dedicated mechanism at the rail level |
| Defective-part retrieval | Planned — grind down rejected parts and feed them back into filament |

The next grasping tasks are action smoothing to reduce policy-output jitter and deploying the gripper empty-hand judgment on real hardware (currently in shadow mode). Planned hardware extensions include a magnetic tool changer (automatic gripper swapping) and proximity-sensing stops (a safety layer on top of the E-STOP).

---

## Team

- **Seoul Robotics High School Capstone Team "All-Rounder"** — a club built around competitions. QUVI started as a capstone project and has continued through to entry in the National Meister High School Star Project.

---

## AI Usage Disclosure

In the interest of transparency, we disclose the AI, open-source software, and external assistance used in building this project.

- **Development-assistance AI**: Anthropic **Claude Code** (Claude Fable 5/Opus/Sonnet/Haiku models), OpenAI **Codex**, Google **Antigravity** — assisted with code writing, review, and debugging. Final design decisions and hardware validation were performed by the team directly
- **AI models embedded in the product**: **SmolVLA** grasp policy (the pretrained model `lerobot/smolvla_base` fine-tuned on our own teleoperation demonstration data; the earlier version used an **ACT** policy we trained ourselves), **PatchCore** anomaly detection (WideResNet50 backbone, memory bank built from our own known-good images)
- **Open source** (with license types): ROS 2 Jazzy (Apache-2.0), micro-ROS (Apache-2.0), LeRobot (Apache-2.0), OpenCV (Apache-2.0), PyTorch (BSD-3-Clause), Flask (BSD-3-Clause), Flask-SocketIO (MIT), NumPy (BSD-3-Clause), DYNAMIXEL SDK (Apache-2.0), PlatformIO (Apache-2.0), ROBOTIS cyclo_intelligence (Apache-2.0, partially vendored) — see [References](#references) below for detailed sources
- **Narration voice (AI TTS)**: the Week 8 three-minute demo video uses **edge-tts** (Microsoft Edge neural voice, `ko-KR-HyunsuMultilingualNeural`); the pitch video and the Week 4 three-minute demo video (first demo day) use AI voices synthesized with **Typecast** (typecast.ai)
- **Web demo analytics**: Google Analytics 4 — counts visits and completed inspection cycles on the public web demo. No personally identifiable information is collected
- **External consulting**: None (no consultation from outside institutions or companies beyond our supervising teacher)

---

## References

- **LeRobot** — Hugging Face's robot imitation-learning framework. This project uses the OMX-supporting fork [ROBOTIS-GIT/lerobot](https://github.com/ROBOTIS-GIT/lerobot) as a submodule (upstream: [huggingface/lerobot](https://github.com/huggingface/lerobot))
- **SmolVLA** — Shukor et al., ["SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"](https://arxiv.org/abs/2506.01844) (2025) — current grasp policy, base model [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **ACT** — Zhao et al., ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705) (RSS 2023) — earlier grasp imitation-learning policy
- **ROBOTIS OpenMANIPULATOR-X (OMX)** — [ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) — robot arm hardware and [DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- **PatchCore** — Roth et al., ["Towards Total Recall in Industrial Anomaly Detection"](https://arxiv.org/abs/2106.08265) (CVPR 2022) — anomaly detection algorithm
- **ROBOTIS cyclo_intelligence** — [ROBOTIS-GIT/cyclo_intelligence](https://github.com/ROBOTIS-GIT/cyclo_intelligence) (Apache-2.0). Four inference modules (constants, image_preprocessing, loading, prediction) from `cyclo_brain/policy/lerobot/lerobot_engine/` were vendored unmodified (as of 2026-09-08) in `scripts/vla_sidecar/lerobot_engine/`, with ROBOTIS copyright and Apache-2.0 headers retained. Only `image_preprocessing.py` was later changed (2 lines: duplicate constant replaced with an import from `constants.py`, 2026-09-17, commit 5465f84)
- **micro-ROS** — [micro-ROS-Agent](https://github.com/micro-ROS/micro-ROS-Agent), [micro_ros_platformio](https://github.com/micro-ROS/micro_ros_platformio) — ESP32-S3 ↔ ROS 2 communication

---

## License

This project is licensed under the [MIT License](LICENSE). The ROBOTIS files in `scripts/vla_sidecar/lerobot_engine/` remain under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
