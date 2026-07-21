#!/usr/bin/env bash
# QUVI micro-ROS Agent 실행 스크립트
# ESP32-S3 (LOLIN S3) 펌웨어와 통신하는 micro-ROS agent 를 기동한다.
#
# 보드레이트는 firmware/quvi_esp32_firmware/Config.h 의 MICRO_ROS_BAUDRATE 와
# 반드시 일치해야 한다 (프로젝트 표준: 115200).
#
# 사용법:
#   scripts/run_microros_agent.sh                   # 기본 /dev/ttyESP32, 115200
#   scripts/run_microros_agent.sh /dev/ttyESP32     # 포트 지정
#   QUVI_MICROROS_BAUD=115200 scripts/run_microros_agent.sh /dev/ttyESP32
set -euo pipefail

DEV="${1:-${QUVI_MICROROS_DEV:-/dev/ttyESP32}}"
BAUD="${QUVI_MICROROS_BAUD:-115200}"

if [ ! -e "$DEV" ]; then
    echo "[run_microros_agent] 경고: 장치 $DEV 가 존재하지 않습니다." >&2
    echo "[run_microros_agent] 연결된 시리얼 장치:" >&2
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null >&2 || echo "  (없음)" >&2
fi

echo "[run_microros_agent] micro-ROS agent 시작: dev=$DEV baud=$BAUD"
exec ros2 run micro_ros_agent micro_ros_agent serial --dev "$DEV" -b "$BAUD"
