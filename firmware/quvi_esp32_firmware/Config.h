#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// =============================================================================
// 통신 모드 선택
// =============================================================================
// 아래 줄을 주석 처리하면 펌웨어가 "표준 시리얼 모드"(UART) 로 전환된다.
// 이 매크로가 정의되면 ESP32-S3 는 micro-ROS 로 동작하며 호스트에 micro-ROS agent 가 있어야 한다.
#define USE_MICRO_ROS

#ifdef USE_MICRO_ROS
  // micro-ROS agent 와 반드시 일치해야 하는 전송 보드레이트.
  // 호스트: ros2 run micro_ros_agent micro_ros_agent serial --dev <port> -b 115200
  // full_system.launch.py micro_ros_baud 기본값과 일치시킨다.
  #define MICRO_ROS_BAUDRATE 115200
#else
  // 표준 시리얼(CLI) 모드 전용 — Arduino Serial Monitor 수동 테스트용.
  #define SERIAL_BAUDRATE 115200
#endif

// =============================================================================
// 하드웨어 핀 매핑 (Wemos LOLIN S3 V1.0.0)
// =============================================================================
// 상태 표시
#define ONBOARD_LED_PIN       38  // LOLIN S3 내장 WS2812B RGB LED

// 안전 및 시스템
#define ESTOP_PIN             16  // 비상정지 스위치 (GPIO 16, 내부 풀업 사용 Active LOW)

// 리니어 레일 (TB6600 스테퍼 드라이버 - EN 핀 미사용)
#define RAIL_PUL_PIN          4   // 펄스/스텝 신호
#define RAIL_DIR_PIN          5   // 방향 신호
#define RAIL_ENA_PIN          -1  // 활성화 신호 미사용 (항상 활성화 상태)
#define RAIL_LIMIT_PIN        6   // 모터 쪽 리밋 스위치 (내부 풀업 사용 Active LOW)

// 턴테이블 (TB6600 스테퍼 드라이버 - EN 핀 미사용)
#define TURN_PUL_PIN          7   // 펄스/스텝 신호
#define TURN_DIR_PIN          15  // 방향 신호
#define TURN_ENA_PIN          -1  // 활성화 신호 미사용 (항상 활성화 상태)
#define TURN_LIMIT_PIN        -1  // 턴테이블 리밋 스위치 미사용 (상대 위치 제어 방식)

// 부속 장치
#define TURN_LED_RELAY_PIN    17  // 턴테이블 LED 링 릴레이 제어 핀 (Active HIGH)


// =============================================================================
// 모터 및 기구 상수
// =============================================================================
// 스테퍼 모터 공통 파라미터
#define STEPPER_STEPS_PER_REV 200   // 표준 1.8도 스테퍼 모터

// 1. 리니어 레일 설정
#define RAIL_MICROSTEPPING    16    // TB6600 DIP 스위치 설정값 (16 마이크로스텝)
#define RAIL_PULLEY_TEETH     20    // 20T 풀리
#define RAIL_BELT_PITCH       2.0f  // GT2 벨트 피치 (2mm)
#define RAIL_MM_PER_REV       (RAIL_PULLEY_TEETH * RAIL_BELT_PITCH) // 1회전당 40.0mm
// SSoT: 이 값(80.0)의 근거 — ROS 쪽은 quvi_robot_control/topics.py 의
// RAIL_STEPS_PER_MM 이 이 값을 미러링하므로(hmi_node 등이 그걸 import),
// 위 스텝퍼/풀리/벨트 상수를 바꾸면 topics.py 도 함께 수정한다.
#define RAIL_STEPS_PER_MM     ((float)(STEPPER_STEPS_PER_REV * RAIL_MICROSTEPPING) / RAIL_MM_PER_REV) // 80.0 steps/mm

// 2. 턴테이블 설정
#define TURN_MICROSTEPPING    16    // TB6600 DIP 스위치 설정값 (예: 16이면 회전당 3200 스텝)
#define TURN_GEAR_RATIO       1.0   // 1:1 (감속 없음)

// 턴테이블 1회전당 스텝 수 계산:
// (200 steps/rev × 16 마이크로스텝 × 1.0 비율 = 360도당 3200 스텝)
#define TURN_STEPS_PER_REV    (STEPPER_STEPS_PER_REV * TURN_MICROSTEPPING * TURN_GEAR_RATIO)
#define TURN_STEPS_PER_DEGREE (TURN_STEPS_PER_REV / 360.0)

// =============================================================================
// 모션 스타일 및 캘리브레이션 (사다리꼴 속도 제어)
// =============================================================================
// 속도·가속도 제한
#define RAIL_MAX_SPEED        16000.0 // 최대 속도 (steps/sec, 400mm/s 상당)
#define RAIL_ACCELERATION     40000.0 // 가속도 (steps/sec^2)
#define TURN_MAX_SPEED        600.0  // 최대 속도 (steps/sec, 턴테이블)
#define TURN_ACCELERATION     1000.0 // 가속도 (steps/sec^2)

// 호밍 캘리브레이션 파라미터 (레일)
// 리밋 스위치는 모터 쪽(좌측)에 있다.
#define RAIL_HOMING_DIR       LOW    // 모터 방향으로 이동하는 방향값 (LOW 또는 HIGH)

#define RAIL_HOME_COARSE_SPD  4000.0 // 고속 탐색 속도 (steps/sec, 50mm/s) — fine 단계가 정밀도 담보
#define RAIL_HOME_FINE_SPD    200.0  // 정밀 저속 탐색 속도 (steps/sec)
#define RAIL_HOME_BACKOFF     150    // 파인 호밍 전 리밋 스위치에서 벗어나기 위한 백오프 스텝 수

// 소프트 위치 제한 (스텝 단위, 호밍 성공 후 기준)
#define RAIL_MIN_LIMIT        0
#define RAIL_MAX_LIMIT        33600  // 420.0f mm * 80.0 steps/mm = 33600 steps

// =============================================================================
// ROS 2 토픽
// =============================================================================
#define TOPIC_RAIL_CMD "/motor/rail"
#define TOPIC_TURN_CMD "/motor/turntable_cmd"
#define TOPIC_TURN_LED "/motor/turntable_led"
#define TOPIC_ESTOP "/system/estop"
#define TOPIC_RAIL_DONE "/motor/rail_done"
#define TOPIC_TURN_DONE "/motor/turntable_done"
#define TOPIC_STATUS "/motor/status"

#endif // CONFIG_H
