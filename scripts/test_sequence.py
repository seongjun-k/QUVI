#!/usr/bin/env python3
"""
웨이포인트 시퀀스 단독 테스트 스크립트 (판정 알고리즘 없이)
현재 위치 → 목표 위치를 보간(interpolation)으로 천천히 이동

사용법 (도커 안에서):
  python3 /workspace/scripts/test_sequence.py           # 자동 실행
  python3 /workspace/scripts/test_sequence.py --step    # 스텝 모드 (Enter로 다음 단계)
"""

import sys
import time

try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS
    )
except ImportError:
    print("dynamixel_sdk 미설치: pip install dynamixel-sdk --break-system-packages")
    sys.exit(1)

# ── 하드웨어 설정 ──────────────────────────────────────────────────────────────
PORT      = '/dev/ttyFollower'
BAUDRATE  = 1_000_000
PROTOCOL  = 2.0

MOTORS = {
    'shoulder_pan':  11,
    'shoulder_lift': 12,
    'elbow_flex':    13,
    'wrist_flex':    14,
    'wrist_roll':    15,
    'gripper':       16,
}

ADDR_TORQUE_ENABLE    = 64
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_POSITION    = 116
LEN_GOAL_POSITION     = 4

GRIPPER_OPEN  = 2300
GRIPPER_CLOSE = 1800

# ── 보간 설정 ──────────────────────────────────────────────────────────────────
# 총 이동 시간 = INTERP_STEPS × INTERP_DELAY
# 현재: 200 × 0.03 = 6초 (매우 느림)
INTERP_STEPS       = 200    # 보간 스텝 수 (많을수록 부드럽고 느림)
INTERP_DELAY       = 0.03   # 스텝 간격 (초)
GRIPPER_STEPS      = 80     # 그리퍼는 스텝 줄여서 조금 빠르게
SETTLE_DELAY       = 0.5    # 목표 도달 후 정착 대기
INSPECT_WAIT       = 3.0    # 판정 대기 (추후 실제 신호로 교체)

# ── 티칭 웨이포인트 ────────────────────────────────────────────────────────────
POSE_P1 = {'shoulder_pan': 2054, 'shoulder_lift': 1258, 'elbow_flex': 2800, 'wrist_flex': 2981, 'wrist_roll': 2035, 'gripper': 2150}  # 베드 위 대기
POSE_P2 = {'shoulder_pan':   12, 'shoulder_lift': 1843, 'elbow_flex': 2165, 'wrist_flex': 3123, 'wrist_roll': 2095, 'gripper': 2150}  # 180도 회전
POSE_P3 = {'shoulder_pan':   16, 'shoulder_lift': 1736, 'elbow_flex': 2413, 'wrist_flex': 3018, 'wrist_roll': 2087, 'gripper': 2150}  # 턴테이블 진입점
POSE_P4 = {'shoulder_pan':   16, 'shoulder_lift': 1841, 'elbow_flex': 2522, 'wrist_flex': 2759, 'wrist_roll': 2085, 'gripper': 2150}  # 턴테이블 놓기/집기 지점
POSE_P5 = {'shoulder_pan': 2047, 'shoulder_lift': 1854, 'elbow_flex': 2460, 'wrist_flex': 2909, 'wrist_roll': 2050, 'gripper': 2150}  # 180도 반대 회전
POSE_P6 = {'shoulder_pan': 2039, 'shoulder_lift': 1076, 'elbow_flex': 2884, 'wrist_flex': 3094, 'wrist_roll': 1993, 'gripper': 2150}  # 분류장 위치

STEP_MODE = '--step' in sys.argv
_step_num = [0]


def open_bus():
    port = PortHandler(PORT)
    pkt  = PacketHandler(PROTOCOL)
    if not port.openPort():
        print(f"포트 열기 실패: {PORT}")
        sys.exit(1)
    if not port.setBaudRate(BAUDRATE):
        print(f"보드레이트 설정 실패: {BAUDRATE}")
        sys.exit(1)
    return port, pkt


def set_torque(port, pkt, enable: bool):
    val = 1 if enable else 0
    for mid in MOTORS.values():
        pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, val)
    print(f"[토크 {'ON' if enable else 'OFF'}]")


def read_positions(port, pkt) -> dict:
    """모든 모터의 현재 위치 읽기"""
    positions = {}
    for name, mid in MOTORS.items():
        val, result, _ = pkt.read4ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS:
            if val > 2147483648:
                val -= 4294967296
            positions[name] = val
        else:
            positions[name] = 2048
    return positions


def write_pose(port, pkt, pose: dict):
    """지정한 모터들에 목표 위치 전송"""
    sw = GroupSyncWrite(port, pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    for name, val in pose.items():
        param = [val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF]
        sw.addParam(MOTORS[name], param)
    sw.txPacket()
    sw.clearParam()


def smooth_move(port, pkt, target: dict, label: str, steps: int = INTERP_STEPS):
    """현재 위치에서 target까지 보간으로 부드럽게 이동"""
    _step_num[0] += 1
    n = _step_num[0]
    print(f"  [{n:2d}] {label}  ({steps * INTERP_DELAY:.1f}초)")

    current = read_positions(port, pkt)

    for i in range(1, steps + 1):
        t = i / steps
        interp = {
            name: int(current[name] + (goal - current[name]) * t)
            for name, goal in target.items()
            if name in current
        }
        write_pose(port, pkt, interp)
        time.sleep(INTERP_DELAY)

    time.sleep(SETTLE_DELAY)

    if STEP_MODE:
        input(f"       → Enter로 다음 단계")


def gripper_open(port, pkt):
    print(f"       그리퍼: 열기")
    smooth_move(port, pkt, {'gripper': GRIPPER_OPEN}, '그리퍼 열기', steps=GRIPPER_STEPS)


def gripper_close(port, pkt):
    print(f"       그리퍼: 닫기")
    smooth_move(port, pkt, {'gripper': GRIPPER_CLOSE}, '그리퍼 닫기', steps=GRIPPER_STEPS)


def run_sequence(port, pkt):
    _step_num[0] = 0
    print("\n[시퀀스 시작]")
    print("=" * 55)
    print(f"  이동 시간: 스텝당 {INTERP_STEPS * INTERP_DELAY:.1f}초")
    print("=" * 55)

    smooth_move(port, pkt, POSE_P1, 'P1  베드 위 대기')
    smooth_move(port, pkt, POSE_P2, 'P2  180도 회전')
    smooth_move(port, pkt, POSE_P3, 'P3  턴테이블 진입점')
    smooth_move(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점')
    gripper_open(port, pkt)

    smooth_move(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (재사용)')
    smooth_move(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점 (재사용)')
    print(f"       [판정 대기 {INSPECT_WAIT}초...]")
    time.sleep(INSPECT_WAIT)
    gripper_close(port, pkt)

    smooth_move(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (재사용)')
    smooth_move(port, pkt, POSE_P5, 'P5  180도 반대 회전')
    smooth_move(port, pkt, POSE_P1, 'P1  베드 위 대기 (재사용)')
    smooth_move(port, pkt, POSE_P6, 'P6  분류장 위치')
    gripper_open(port, pkt)

    print("=" * 55)
    print("[시퀀스 완료]")


def main():
    print("=" * 55)
    print("  웨이포인트 시퀀스 테스트 (보간 이동)")
    print(f"  이동 속도: 스텝 {INTERP_STEPS}개 × {INTERP_DELAY}초 = {INTERP_STEPS * INTERP_DELAY:.1f}초/포즈")
    if STEP_MODE:
        print("  [스텝 모드] 각 단계 후 Enter로 진행")
    print(f"  포트: {PORT}")
    print("=" * 55)
    print("\n[Enter]로 시작, Ctrl+C로 중단")
    input()

    port, pkt = open_bus()
    set_torque(port, pkt, True)

    try:
        while True:
            try:
                run_sequence(port, pkt)
            except KeyboardInterrupt:
                print("\n[중단]")
                break
            print("\n다시 실행하려면 [Enter], 종료하려면 Ctrl+C")
            try:
                input()
            except KeyboardInterrupt:
                print("\n[종료]")
                break
    finally:
        port.closePort()
        print("[포트 닫음]")


if __name__ == '__main__':
    main()
