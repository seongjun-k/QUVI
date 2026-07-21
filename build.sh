#!/bin/bash
# ------------------------------------------------------------------
# QUVI Docker Build Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부로 접속하여
# ROS 2 워크스페이스를 빌드한 후 컨테이너 셸(bash) 접속을 유지합니다.
# ------------------------------------------------------------------

CONTAINER_NAME="quvi-dev"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

source "${SCRIPT_DIR}/docker/find_or_start_container.sh"

echo "[QUVI] 컨테이너(${TARGET_CONTAINER}) 내부 ROS 2 워크스페이스 빌드 중..."
if [ -t 0 ]; then
    docker exec -it "${TARGET_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace && colcon build --symlink-install && source install/setup.bash && exec bash"
    if [ $? -ne 0 ]; then
        echo "오류: 컨테이너 빌드 세션 실행 실패."
        exit 1
    fi
else
    docker exec -i "${TARGET_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace && colcon build --symlink-install"
    if [ $? -ne 0 ]; then
        echo "오류: colcon 빌드 실패."
        exit 1
    fi
fi
