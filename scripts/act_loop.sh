#!/usr/bin/env bash
# ACT 파지 반복 실행 스크립트 — 카메라 위치 튜닝용
# ──────────────────────────────────────────────
# /robot/act_grasp 동기 서비스(파지 완료까지 블로킹)를 반복 호출하며
# 성공/실패를 집계한다. 카메라·물체 위치를 바꿔가며 파지 성공률을 확인하는 용도.
#
# 실행 위치: 호스트 터미널에서 바로 (컨테이너로 자동 진입)
#   ./scripts/act_loop.sh [횟수] [대기초]
# 전체 시스템(qrun) 꺼진 상태면 최소 노드(camera1 + robot_control, use_act=true)를
# 직접 띄우고, 스크립트 종료 시 함께 정리한다. qrun 켜져 있으면 그대로 서비스만 호출.
# 인자:
#   횟수    반복 횟수 (기본 0 = 무한, Ctrl+C 로 중단)
#   대기초  각 파지 사이 대기 (기본 2초 — 물체 재배치 시간이 필요하면 늘릴 것)
# 옵션:
#   HOME_BETWEEN=1  매 파지 후 /robot/go_home 으로 홈 복귀 후 다음 회차 진행

# 호스트에서 실행 시 컨테이너로 재진입
if [ ! -d /workspace ]; then
    exec docker exec -it quvi-dev /workspace/scripts/act_loop.sh "$@"
fi

# ROS setup.bash 는 미정의 변수를 참조하므로 set -u 는 소싱 이후에 켠다
source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash 2>/dev/null || true
set -u

COUNT="${1:-0}"
DELAY="${2:-2}"
i=0; ok=0; fail=0
SPAWNED_PIDS=()

have_service() {
    ros2 service list 2>/dev/null | grep -q '^/robot/act_grasp$'
}

cleanup() {
    if [ "${#SPAWNED_PIDS[@]}" -gt 0 ]; then
        echo "[act_loop] 직접 띄운 노드 정리 (${SPAWNED_PIDS[*]})"
        kill "${SPAWNED_PIDS[@]}" 2>/dev/null
        wait "${SPAWNED_PIDS[@]}" 2>/dev/null
    fi
}

summary() {
    echo
    echo "[act_loop] 종료 — 총 ${i}회 | 성공 ${ok} | 실패 ${fail}"
}
trap 'summary; cleanup; exit 0' INT
trap 'cleanup' EXIT

# ─── qrun 꺼진 상태면 최소 노드 직접 기동 ───
if ! have_service; then
    echo "[act_loop] /robot/act_grasp 없음 — 최소 노드 직접 기동 (camera1 + robot_control)"

    # camera1: full_system 과 동일하게 fixed_cam 장치 → /camera1 토픽 (ACT 관측 입력)
    ros2 run usb_cam usb_cam_node_exe --ros-args \
        -r __ns:=/camera1 -r __node:=camera1 \
        -p video_device:=/dev/fixed_cam \
        -p image_width:=640 -p image_height:=480 \
        -p pixel_format:=mjpeg2rgb -p framerate:=30.0 \
        -p camera_name:=sidecam \
        -p autoexposure:=false -p exposure:=110 -p brightness:=0 \
        > /tmp/act_loop_camera1.log 2>&1 &
    SPAWNED_PIDS+=($!)

    ros2 run quvi_robot_control robot_control_node --ros-args \
        -p use_real_hardware:=true -p use_act:=true \
        -p dxl_port:=/dev/ttyFollower -p dxl_baudrate:=1000000 \
        -p act_device:=cpu \
        -p sidecam_topic:=/camera1/image_raw/compressed -p use_compressed:=true \
        > /tmp/act_loop_robot.log 2>&1 &
    SPAWNED_PIDS+=($!)

    echo "[act_loop] 서비스 대기 (ACT 모델 로드 포함, 최대 120초)..."
    ready=0
    for _ in $(seq 120); do
        if have_service; then ready=1; break; fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "[act_loop] 서비스 대기 실패 — 로그 확인: /tmp/act_loop_robot.log"
        tail -20 /tmp/act_loop_robot.log
        exit 1
    fi
    echo "[act_loop] 노드 준비 완료"
fi

# ─── 반복 루프 ───
while :; do
    i=$((i + 1))
    echo "[act_loop] ── ACT 파지 #${i} 시작 ──"
    out=$(ros2 service call /robot/act_grasp std_srvs/srv/Trigger 2>&1)
    if echo "$out" | grep -q "success=True"; then
        ok=$((ok + 1))
        echo "[act_loop] #${i} 성공  (누적: 성공 ${ok} / 실패 ${fail})"
    else
        fail=$((fail + 1))
        echo "[act_loop] #${i} 실패  (누적: 성공 ${ok} / 실패 ${fail})"
        echo "$out" | tail -2
    fi

    if [ "${HOME_BETWEEN:-0}" = "1" ]; then
        echo "[act_loop] 홈 복귀"
        ros2 service call /robot/go_home std_srvs/srv/Trigger > /dev/null 2>&1
    fi

    [ "$COUNT" -gt 0 ] && [ "$i" -ge "$COUNT" ] && break
    sleep "$DELAY"
done
summary
