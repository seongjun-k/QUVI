#!/bin/bash
# ------------------------------------------------------------------
# QUVI System Launch Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부로 접속하여
# full_system.launch.py 메인 런치 파일을 실행합니다.
# 인자는 launch 로 그대로 전달됩니다 (데모 녹화 예:
#   ./run.sh rerun_save_path:=/workspace/data/demo_bags/pass.rrd )
# ------------------------------------------------------------------

CONTAINER_NAME="quvi-dev"
LAUNCH_ARGS="$*"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

source "${SCRIPT_DIR}/docker/find_or_start_container.sh"

# ─── qrun 사전 정리 ───
# QUVI(quvi-dev)와 cyclo 스택(open_manipulator/zenoh)은 같은 /dev/video·DYNAMIXEL 포트를
# 두고 경합한다. cyclo 컨테이너가 남아 usb_cam이 카메라를 선점하면 QUVI 카메라가 안 뜨므로
# 실행 전에 내린다. 이미지/볼륨은 건드리지 않는다(정지만 — 재빌드 방지).
_quvi_free_container() {
    local c="$1"
    [ -z "$(docker ps -q -f name="^${c}$")" ] && return 0
    echo "[QUVI] 충돌 컨테이너 종료: ${c}"
    docker stop -t 5 "${c}" >/dev/null 2>&1
    # snap dockerd + AppArmor 환경에선 stop 시그널이 막힐 수 있다(기록된 함정).
    # 그래도 살아있으면 내부 usb_cam/respawn 래퍼만 죽여 장치라도 반납시킨다.
    if [ -n "$(docker ps -q -f name="^${c}$")" ]; then
        docker exec "${c}" bash -c "pkill -9 -f usb_cam; pkill -9 -f 'while true'" >/dev/null 2>&1
        echo "[QUVI]   (stop 미적용 → ${c} 내부 usb_cam 강제 종료로 카메라 반납)"
    fi
}
for _c in open_manipulator zenoh_daemon cyclo_manager cyclo_manager_ui lerobot_server; do
    _quvi_free_container "${_c}"
done

# 페이지 캐시 비우기(메모리 확보) — sudo 무암호 가능할 때만, 실패해도 무시.
if sudo -n true 2>/dev/null; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && echo "[QUVI] 페이지 캐시 정리 완료"
fi

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
            ros2 launch quvi_bringup full_system.launch.py ${LAUNCH_ARGS}
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
            ros2 launch quvi_bringup full_system.launch.py ${LAUNCH_ARGS}
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
