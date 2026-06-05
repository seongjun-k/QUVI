#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// =============================================================================
// COMMUNICATION MODE SELECTION
// =============================================================================
// Comment out the line below to switch the firmware to "Standard Serial Mode" (UART).
// When defined, the ESP32-S3 will run micro-ROS and expect a micro-ROS agent on the host.
#define USE_MICRO_ROS

#ifdef USE_MICRO_ROS
  // micro-ROS agent 와 반드시 일치해야 하는 전송 보드레이트.
  // 호스트: ros2 run micro_ros_agent micro_ros_agent serial --dev <port> -b 921600
  // (네이티브 USB CDC 사용 시 물리적으로는 nominal 값이지만, agent 인자/문서와의
  //  일관성을 위해 프로젝트 표준값 921600 으로 통일한다.)
  #define MICRO_ROS_BAUDRATE 921600
#else
  // 표준 시리얼(CLI) 모드 전용 — Arduino Serial Monitor 수동 테스트용.
  #define SERIAL_BAUDRATE 115200
#endif

// =============================================================================
// HARDWARE PIN MAPPING (Wemos LOLIN S3 V1.0.0)
// =============================================================================
// Status Visuals
#define ONBOARD_LED_PIN       38  // Built-in WS2812B RGB LED on LOLIN S3

// Safety & System
#define ESTOP_PIN             9   // Emergency Stop Switch (GPIO 9, Active LOW with Internal Pull-Up)

// LINEAR RAIL (TB6600 Stepper Driver)
#define RAIL_PUL_PIN          1   // Pulse/Step signal
#define RAIL_DIR_PIN          2   // Direction signal
#define RAIL_ENA_PIN          3   // Enable signal (Active LOW)
#define RAIL_LIMIT_PIN        4   // Limit switch at motor side (left) (Active LOW with Internal Pull-Up)

// TURNTABLE (TB6600 Stepper Driver)
#define TURN_PUL_PIN          5   // Pulse/Step signal
#define TURN_DIR_PIN          6   // Direction signal
#define TURN_ENA_PIN          7   // Enable signal (Active LOW)
#define TURN_LIMIT_PIN        8   // Optional Turntable Index/Limit Switch (Active LOW with Internal Pull-Up)

// =============================================================================
// MOTOR & GEOMETRIC CONSTANTS
// =============================================================================
// Generic Stepper Motor Parameters
#define STEPPER_STEPS_PER_REV 200   // Standard 1.8 degree stepper motor

// 1. Linear Rail Configuration
#define RAIL_MICROSTEPPING    16    // Configured on TB6600 DIP switches (e.g. 16 means 3200 steps/rev)
#define RAIL_STEPS_PER_MM     40.0  // Steps per millimeter calibration factor (adjust based on lead screw / belt pitch)

// 2. Turntable Configuration
#define TURN_MICROSTEPPING    16    // Configured on TB6600 DIP switches (e.g. 16 means 3200 steps/rev)
#define TURN_GEAR_RATIO       3.0   // 3:1 mechanical reduction ratio

// Derived steps per full revolution of the turntable table:
// (200 steps/rev * 16 microsteps * 3.0 ratio = 9600 steps per 360 degrees)
#define TURN_STEPS_PER_REV    (STEPPER_STEPS_PER_REV * TURN_MICROSTEPPING * TURN_GEAR_RATIO)
#define TURN_STEPS_PER_DEGREE (TURN_STEPS_PER_REV / 360.0)

// =============================================================================
// MOTION STYLES & CALIBRATION (TRAPEZOIDAL CONTROL)
// =============================================================================
// Speed & Acceleration Limits
#define RAIL_MAX_SPEED        3000.0 // Maximum speed in steps/sec
#define RAIL_ACCELERATION     5000.0 // Acceleration rate in steps/sec^2
#define TURN_MAX_SPEED        1500.0 // Maximum speed in steps/sec (turntable)
#define TURN_ACCELERATION     2000.0 // Acceleration rate in steps/sec^2

// Homing Calibration Parameters (Rail)
// The limit switch is on the motor side (left side). Homing CCW (towards motor).
#define RAIL_HOMING_DIR       LOW    // Direction value to move towards the motor (LOW or HIGH)
#define RAIL_RUNNING_DIR      HIGH   // Direction value to move away from motor (positive step increment)

#define RAIL_HOME_COARSE_SPD  800.0  // Rapid search speed (steps/sec)
#define RAIL_HOME_FINE_SPD    200.0  // High-precision slow search speed (steps/sec)
#define RAIL_HOME_BACKOFF     150    // Back-off steps to clear limit switch before fine homing

// Homing Calibration Parameters (Turntable - Optional)
#define TURN_HOMING_DIR       LOW
#define TURN_HOME_COARSE_SPD  400.0
#define TURN_HOME_FINE_SPD    100.0
#define TURN_HOME_BACKOFF     100

// Soft Position Limits (in steps, after successful Homing)
#define RAIL_MIN_LIMIT        0
#define RAIL_MAX_LIMIT        5000   // Prevents rail from hitting the far right physical end stop

#endif // CONFIG_H
