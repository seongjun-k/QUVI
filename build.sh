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

# --base-paths src: 워크스페이스 루트 전체를 스캔하면 data/vla_sidecar/venv 안의
# numpy/cmake 테스트용 setup.py·CMakeLists 를 ROS 패키지로 오인해 빌드가 죽는다.
# 실제 패키지는 전부 src/ 아래이므로 스캔 범위를 src 로 한정한다.
echo "[QUVI] 컨테이너(${TARGET_CONTAINER}) 내부 ROS 2 워크스페이스 빌드 중..."
if [ -t 0 ]; then
    docker exec -it "${TARGET_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace && colcon build --symlink-install --base-paths src && source install/setup.bash && exec bash"
    if [ $? -ne 0 ]; then
        echo "오류: 컨테이너 빌드 세션 실행 실패."
        exit 1
    fi
else
    docker exec -i "${TARGET_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace && colcon build --symlink-install --base-paths src"
    if [ $? -ne 0 ]; then
        echo "오류: colcon 빌드 실패."
        exit 1
    fi
fi
