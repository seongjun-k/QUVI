#!/bin/bash
# ------------------------------------------------------------------
# QUVI Demo Launch Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부에서 demo.launch.py 를
# 실행한다 — UI(hmi_node)만 기동, 로봇 구동 계열 노드는 띄우지 않는다.
# ------------------------------------------------------------------

CONTAINER_NAME="quvi-dev"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

source "${SCRIPT_DIR}/docker/find_or_start_container.sh"

echo "[QUVI] 컨테이너(${TARGET_CONTAINER}) 내부에서 데모(UI 전용) 실행 중..."
docker exec -i "${TARGET_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash
    if [ ! -f /workspace/install/setup.bash ]; then
        echo '/workspace/install/setup.bash 없음 — build.sh 를 먼저 실행하세요.'
        exit 1
    fi
    source /workspace/install/setup.bash
    ros2 launch quvi_bringup demo.launch.py
"
