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
#define ESTOP_PIN             16  // Emergency Stop Switch (GPIO 16, Active LOW with Internal Pull-Up)

// LINEAR RAIL (TB6600 Stepper Driver - EN pin removed)
#define RAIL_PUL_PIN          4   // Pulse/Step signal
#define RAIL_DIR_PIN          5   // Direction signal
#define RAIL_ENA_PIN          -1  // Enable signal removed (always enabled)
#define RAIL_LIMIT_PIN        6   // Limit switch at motor side (Active LOW with Internal Pull-Up)

// TURNTABLE (TB6600 Stepper Driver - EN pin removed)
#define TURN_PUL_PIN          7   // Pulse/Step signal
#define TURN_DIR_PIN          15  // Direction signal
#define TURN_ENA_PIN          -1  // Enable signal removed (always enabled)
#define TURN_LIMIT_PIN        -1  // Turntable Limit Switch disabled (using relative control)

// Accessories
#define TURN_LED_RELAY_PIN    18  // Turntable LED Ring Relay Control Pin (Active HIGH)


// =============================================================================
// MOTOR & GEOMETRIC CONSTANTS
// =============================================================================
// Generic Stepper Motor Parameters
#define STEPPER_STEPS_PER_REV 200   // Standard 1.8 degree stepper motor

// 1. Linear Rail Configuration
#define RAIL_MICROSTEPPING    16    // Configured on TB6600 DIP switches (16 microsteps)
#define RAIL_PULLEY_TEETH     20    // 20T Pulley
#define RAIL_BELT_PITCH       2.0f  // GT2 Belt Pitch (2mm)
#define RAIL_MM_PER_REV       (RAIL_PULLEY_TEETH * RAIL_BELT_PITCH) // 40.0mm per rev
#define RAIL_STEPS_PER_MM     ((float)(STEPPER_STEPS_PER_REV * RAIL_MICROSTEPPING) / RAIL_MM_PER_REV) // 80.0 steps/mm

// 2. Turntable Configuration
#define TURN_MICROSTEPPING    16    // Configured on TB6600 DIP switches (e.g. 16 means 3200 steps/rev)
#define TURN_GEAR_RATIO       3.0   // 3:1 mechanical reduction ratio

// Derived steps per full revolution of the turntable table:
// (200 steps/rev * 16 microsteps * 3.0 ratio = 9600 steps per 360 degrees)
#define TURN_STEPS_PER_REV    (STEPPER_STEPS_PER_REV * TURN_MICROSTEPPING * TURN_GEAR_RATIO)
#define TURN_STEPS_PER_DEGREE ((double)(TURN_STEPS_PER_REV) / 360.0)

// =============================================================================
// MOTION STYLES & CALIBRATION (TRAPEZOIDAL CONTROL)
// =============================================================================
// Speed & Acceleration Limits
#define RAIL_MAX_SPEED        16000.0 // Maximum speed in steps/sec (equivalent to 400mm/s)
#define RAIL_ACCELERATION     40000.0 // Acceleration rate in steps/sec^2
#define TURN_MAX_SPEED        600.0  // Maximum speed in steps/sec (turntable)
#define TURN_ACCELERATION     1000.0 // Acceleration rate in steps/sec^2

// Homing Calibration Parameters (Rail)
// The limit switch is on the motor side (left side).
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

// Soft Position Limits (after successful Homing)
// Rail: mm 단위로 수신 후 steps 로 변환. 상한은 420.0mm.
#define RAIL_MIN_LIMIT_MM     0.0f
#define RAIL_MAX_LIMIT_MM     420.0f

// =============================================================================
// ROS 2 TOPICS
// =============================================================================
// /motor/rail        : std_msgs/Float32  — 목표 위치 (mm)
// /motor/turntable_cmd : std_msgs/Int32  — 목표 각도 (degree, 정수)
// /motor/turntable_led : std_msgs/Bool   — LED 릴레이 ON/OFF
// /system/estop      : std_msgs/Bool     — 소프트 E-STOP (true=발동, false=해제)
// /motor/rail_done   : std_msgs/Bool     — Rail 이동 완료 알림
// /motor/turntable_done : std_msgs/Bool  — Turntable 이동 완료 알림
// /motor/status      : quvi_msgs/MotorStatus — 100ms 주기 상태
#define TOPIC_RAIL_CMD        "/motor/rail"
#define TOPIC_TURN_CMD        "/motor/turntable_cmd"
#define TOPIC_TURN_LED        "/motor/turntable_led"
#define TOPIC_ESTOP           "/system/estop"
#define TOPIC_RAIL_DONE       "/motor/rail_done"
#define TOPIC_TURN_DONE       "/motor/turntable_done"
#define TOPIC_STATUS          "/motor/status"

#endif // CONFIG_H
