# QUVI Linear Rail & Turntable Controller Firmware

This directory contains the production-grade, highly-optimized firmware for the **Wemos LOLIN S3 V1.0.0** (ESP32-S3) microcontroller. It provides high-performance, multi-threaded control for two stepper motors driven by **TB6600 industrial motor drivers**:
1. **Linear Rail:** Coordinates motion along the 3D printer bed (BED=0), inspection chamber (INSPECT=1000), PASS bin (PASS=1700), and FAIL bin (FAIL=2400). Calibrated on boot via a left-side limit switch (motor side).
2. **Turntable:** Controls sample rotation (0°, 90°, 180°, 270°) with a **3:1 gear ratio** and an absolute **shortest-path angular routing algorithm**.

---

## 1. Hardware Pin Mapping & Wiring Guide

Connect the Wemos LOLIN S3 board to the two TB6600 drivers, the homing limit switch, and the Emergency Stop button as follows:

| LOLIN S3 PIN | Signal Name | Target Hardware Device | TB6600 Driver Pin | Description |
| :--- | :--- | :--- | :--- | :--- |
| **GND** | GND | Common Ground | **GND / PUL- / DIR- / ENA-** | Tie all negative signals together |
| **GPIO 1** | `RAIL_PUL` | Linear Rail TB6600 | **PUL+ (PUL)** | Pulse step signal (Active HIGH) |
| **GPIO 2** | `RAIL_DIR` | Linear Rail TB6600 | **DIR+ (DIR)** | Direction signal (HIGH=Right, LOW=Left/Homing) |
| **GPIO 3** | `RAIL_ENA` | Linear Rail TB6600 | **ENA+ (ENA)** | Enable signal (Active LOW) |
| **GPIO 4** | `RAIL_LIMIT`| Rail Limit Switch (Left) | **COM / NO** | Active LOW (Grounds GPIO 4 on contact) |
| **GPIO 5** | `TURN_PUL` | Turntable TB6600 | **PUL+ (PUL)** | Pulse step signal (Active HIGH) |
| **GPIO 6** | `TURN_DIR` | Turntable TB6600 | **DIR+ (DIR)** | Direction signal |
| **GPIO 7** | `TURN_ENA` | Turntable TB6600 | **ENA+ (ENA)** | Enable signal (Active LOW) |
| **GPIO 8** | `TURN_LIMIT`| Turntable Index Switch | **Optional** | Active LOW index switch for turntable homing |
| **GPIO 9** | `ESTOP` | Emergency Stop | **NC (Normally Closed)** | active LOW hardware interrupt (Estop on ground loss) |
| **GPIO 38** | Onboard LED | Built-in WS2812 RGB | Onboard | Provides colored visual feedback |

> [!WARNING]
> **Common Ground:** Ensure the **GND** of the LOLIN S3 board is securely tied to the **GND/V-** of the TB6600 motor power supply (e.g. 24V) to avoid floating voltages and signal corruption.
>
> **Optocoupler Isolation:** Since TB6600 inputs are opto-isolated, tie PUL-, DIR-, and ENA- together to the ESP32 GND, and feed PUL+, DIR+, and ENA+ from the GPIOs. The LOLIN S3 GPIOs operate at 3.3V, which is fully sufficient to trigger TB6600 inputs.

---

## 2. TB6600 DIP Switch Configurations

Set the physical DIP switches on the side of both TB6600 drivers according to these tables.

### A. Microstepping (16x microsteps / 3200 steps per rev)
For both drivers, we use **1/16 microstepping** to ensure ultra-quiet, high-precision motion:

| Microstep | Steps/Rev | S1 | S2 | S3 |
| :--- | :--- | :--- | :--- | :--- |
| **1/16** | **3200** | **ON** | **OFF** | **ON** |

### B. Current Limit (e.g. 1.5A or 2.0A)
Configure the current limits to match your stepper motors' rated current. Adjust this table based on your motor datasheets:

| Current (A) | Peak (A) | S4 | S5 | S6 |
| :--- | :--- | :--- | :--- | :--- |
| **1.5A** | 1.7A | ON | OFF | OFF |
| **2.0A** | 2.2A | OFF | ON | ON |

---

## 3. Compilation & Flashing Guide

The project is structured to compile natively in both the **Arduino IDE** and **PlatformIO**.

### Step 1: Install Library Dependencies
Open your Arduino IDE Library Manager (`Ctrl+Shift+I`) and install:
1. **Adafruit NeoPixel** (by Adafruit) - Used to drive the onboard WS2812B status LED.
2. **micro_ros_arduino** (Required only if `USE_MICRO_ROS` is enabled)
   - Download the precompiled library zip for your ROS 2 version (Jazzy) from the [micro-ROS Arduino GitHub Releases](https://github.com/micro-ROS/micro_ros_arduino/releases).
   - Install it via **Sketch** -> **Include Library** -> **Add .ZIP Library**.

### Step 2: Choose Communication Mode
Open [Config.h](./Config.h):
- **For ROS 2 Native Integration (Recommended):** Leave `#define USE_MICRO_ROS` active.
- **For Plug-and-Play Testing / UART Bridge:** Comment out `//#define USE_MICRO_ROS`. This will fallback to a fast ASCII Serial CLI monitor.

### Step 3: Flash the Board
1. Select **Board:** "Lolin S3" (under ESP32 Arduino Board Manager).
2. Connect your LOLIN S3 to your PC via the USB-C port.
3. Select the correct COM/tty port.
4. Click **Upload** (`Ctrl+U`).

---

## 4. Onboard Visual Status Color Codes

The onboard WS2812B RGB LED provides the following real-time system states:
- 🟣 **Blinking Purple:** Initializing micro-ROS transports / serial ports.
- 🟡 **Blinking Yellow:** Auto-homing calibration active (linear rail moving left).
- 🔵 **Solid Blue:** System idle, calibration successful. Ready for commands.
- 🟢 **Solid Green:** Motors active & executing smooth acceleration profiles.
- 🔴 **Solid Red:** Emergency Stop triggered! Both motors are hard-disabled.

---

## 5. Testing & Verification

### A. Testing in Standard Serial Mode (CLI)
When `USE_MICRO_ROS` is disabled, open the Arduino Serial Monitor (baudrate `115200` with `LF` / `Newline` endings) and enter:
- **`H`** - Re-triggers the coarse/fine homing calibration.
- **`R 1000`** - Drives the linear rail to step 1000.
- **`T 90`** - Smoothly rotates the turntable to 90 degrees.
- **`S`** - Prints current positions and limit switch electrical state.
- **`E`** - Simulates an instant software Emergency Stop.

### B. Testing in micro-ROS Mode
When micro-ROS mode is active, the board acts as a native ROS 2 publisher/subscriber over the micro-DDS protocol.

1. **Launch the micro-ROS Agent on the Ubuntu Host (Docker Container):**
   ```bash
   ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 921600
   ```
   *(Replace `/dev/ttyACM0` with the actual device port of your LOLIN S3. The
   baudrate `921600` must match `MICRO_ROS_BAUDRATE` in `Config.h`. A helper
   script is provided: `scripts/run_microros_agent.sh`.)*

2. **Verify topics are visible in ROS 2:**
   ```bash
   ros2 topic list
   ```
   You should see:
   - `/motor/rail`
   - `/motor/turntable`

3. **Publish movements manually from host:**
   ```bash
   # Move Linear Rail to INSPECT position
   ros2 topic pub /motor/rail std_msgs/msg/Int32 "{data: 1000}" --once

   # Rotate Turntable to 270 degrees
   ros2 topic pub /motor/turntable std_msgs/msg/Int32 "{data: 270}" --once
   ```
