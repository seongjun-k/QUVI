#!/bin/bash
# ------------------------------------------------------------------
# QUVI System Launch Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부로 접속하여
# full_system.launch.py 메인 런치 파일을 실행합니다.
# ------------------------------------------------------------------

CONTAINER_NAME="quvi-dev"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

source "${SCRIPT_DIR}/docker/find_or_start_container.sh"

echo "[QUVI] 컨테이너(${TARGET_CONTAINER}) 내부에서 메인 프로그램 실행 중..."
if [ -t 0 ]; then
    docker exec -it "${TARGET_CONTAINER}" bash -c "
        source /opt/ros/jazzy/setup.bash
        [ -f /uros_ws/install/setup.bash ] && source /uros_ws/install/setup.bash
        if [ ! -f /workspace/install/setup.bash ]; then
            echo '/workspace/install/setup.bash 없음 — 빌드가 필요합니다.'
            echo '   → build.sh 를 먼저 실행하여 ROS 2 워크스페이스를 빌드하세요.'
            exit 1
        fi
        source /workspace/install/setup.bash
        # 대시보드 장치 변경 시 재시작 감시 루프. sentinel 없으면 1회 실행 후 종료(기존 동작).
        rm -f /workspace/data/.restart_requested
        while true; do
            ros2 launch quvi_bringup full_system.launch.py
            launch_status=\$?
            [ -f /workspace/data/.restart_requested ] || exit \$launch_status
            rm -f /workspace/data/.restart_requested
            echo '장치 설정 변경 — 시스템 재기동'
            sleep 2
        done
    "
else
    docker exec -i "${TARGET_CONTAINER}" bash -c "
        source /opt/ros/jazzy/setup.bash
        [ -f /uros_ws/install/setup.bash ] && source /uros_ws/install/setup.bash
        if [ ! -f /workspace/install/setup.bash ]; then
            echo '/workspace/install/setup.bash 없음 — build.sh 를 먼저 실행하세요.'
            exit 1
        fi
        source /workspace/install/setup.bash
        rm -f /workspace/data/.restart_requested
        while true; do
            ros2 launch quvi_bringup full_system.launch.py
            launch_status=\$?
            [ -f /workspace/data/.restart_requested ] || exit \$launch_status
            rm -f /workspace/data/.restart_requested
            echo '장치 설정 변경 — 시스템 재기동'
            sleep 2
        done
    "
fi

if [ $? -ne 0 ]; then
    echo "오류: 메인 프로그램 실행 실패."
    exit 1
fi
