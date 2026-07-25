#!/bin/bash
# ------------------------------------------------------------------
# QUVI Demo Bag Recorder
# 실기 검사 사이클을 데모 재생용 bag 으로 녹화한다 (demo/dashboard 참고).
# 사용법:  ./record_demo.sh pass   (양품 사이클)
#          ./record_demo.sh fail   (불량품 사이클)
# 실행 위치: 호스트. full_system 이 떠 있는 상태에서 별도 터미널로 실행한다.
# 실행하면 대기 상태가 되고, 대시보드에서 "시작"을 누르는 순간부터 사이클이 끝날 때까지만
# 녹화한다(demo_record_gate.py) — 앞뒤 유휴 구간이 bag 에 붙지 않는다.
#
# rrd(ACT 추론 rerun)는 bag 이 아니라 launch 인자로 생성된다 — rr.save()는 노드 시작 시
# 싱크가 고정되므로(robot_control_node._init_rerun) 이 스크립트가 켜고 끌 수 없다.
# pass/fail 을 각각 남기려면 파일명을 나눠야 한다(같은 이름이면 뒤 런이 덮어쓴다).
# 재생 쪽(demo.launch.py)도 demo_bags 의 *.rrd 를 모두 rerun 에 넘겨야 둘 다 보인다.
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

# 대시보드가 소비하는 토픽만 녹화 (demo_controller 재생 대상과 일치해야 함).
# /inspection/result 는 판정 결과 표시의 원천(hmi_node._inspection_cb) — 빠지면 재생 시 판정이 안 뜬다.
TOPICS="/camera1/image_raw/compressed /camera2/image_raw/compressed /inspect/debug_image /inspection/result /hmi/status"

RRD="/workspace/data/demo_bags/${NAME}.rrd"

echo "[QUVI] rrd 는 launch 로만 생성된다 — full_system 이 아래 인자로 떠 있어야 한다:"
echo "  ros2 launch quvi_bringup full_system.launch.py rerun_save_path:=${RRD}"
echo "[QUVI] '${NAME}' 대기 중 — 대시보드에서 \"시작\"을 누르면 녹화가 시작된다 (Ctrl-C 로 중단)"

docker exec -it "${TARGET_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash
    source /workspace/install/setup.bash
    rm -rf /workspace/data/demo_bags/${NAME}
    mkdir -p /workspace/data/demo_bags
    python3 /workspace/scripts/demo_record_gate.py /workspace/data/demo_bags/${NAME} ${TOPICS}
"

echo "[QUVI] 저장 상태:"
docker exec "${TARGET_CONTAINER}" bash -c "
    if [ -s /workspace/data/demo_bags/${NAME}/metadata.yaml ]; then
        echo \"  bag  : data/demo_bags/${NAME} (\$(du -sh /workspace/data/demo_bags/${NAME} | cut -f1))\"
    else
        echo '  bag  : 없음 — \"시작\"을 누르지 않았거나 녹화가 비정상 종료됐다'
    fi
    if [ -s ${RRD} ]; then
        echo \"  rrd  : data/demo_bags/${NAME}.rrd (\$(du -h ${RRD} | cut -f1)) — full_system 종료 후 쓰기 완결\"
    else
        echo '  rrd  : 없음 — launch 에 rerun_save_path 를 안 줬거나 ACT 추론(집기)이 안 돌았다'
    fi
"
