#!/bin/bash
# ------------------------------------------------------------------
# QUVI Demo Bag Recorder
# 실기 검사 사이클을 데모 재생용 bag 으로 녹화한다 (demo/dashboard 참고).
# 사용법:  ./record_demo.sh pass   (양품 사이클)
#          ./record_demo.sh fail   (불량품 사이클)
# 실행 위치: 호스트. full_system(run.sh)이 떠 있는 상태에서 별도 터미널로 실행,
# 녹화 시작 후 대시보드에서 "시작"을 눌러 사이클 1회를 돌리고 Ctrl-C 로 종료.
# rerun rrd 동시 녹화가 필요하면 run.sh 대신:
#   ros2 launch quvi_bringup full_system.launch.py rerun_save_path:=/workspace/data/demo_bags/act.rrd
# ------------------------------------------------------------------

NAME="$1"
if [ "$NAME" != "pass" ] && [ "$NAME" != "fail" ]; then
    echo "사용법: $0 pass|fail"
    exit 1
fi

CONTAINER_NAME="quvi-dev"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

source "${SCRIPT_DIR}/docker/find_or_start_container.sh"

# 대시보드가 소비하는 토픽만 녹화 (demo_controller 재생 대상과 일치해야 함)
TOPICS="/camera1/image_raw/compressed /camera2/image_raw/compressed /inspect/debug_image /hmi/status"

echo "[QUVI] '${NAME}' 데모 bag 녹화 시작 — Ctrl-C 로 종료"
docker exec -it "${TARGET_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash
    source /workspace/install/setup.bash
    rm -rf /workspace/data/demo_bags/${NAME}
    mkdir -p /workspace/data/demo_bags
    ros2 bag record -o /workspace/data/demo_bags/${NAME} ${TOPICS}
"
