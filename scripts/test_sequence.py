#!/usr/bin/env python3
"""
웨이포인트 시퀀스 단독 테스트 스크립트 (판정 알고리즘 없이)
dynamixel_sdk 직접 사용 — lerobot/ROS 불필요

10단계 시퀀스 (P1~P6 unique, 재사용 있음):
  Step 1  : P1  베드 위 대기
  Step 2  : P2  180도 회전
  Step 3  : P3  턴테이블 진입점
  Step 4  : P4  턴테이블 놓기 지점  → 그리퍼 열기
  Step 5  : P3  턴테이블 진입점  (재사용)
  Step 6  : P4  턴테이블 놓기 지점  → 판정 대기 → 그리퍼 닫기  (재사용)
  Step 7  : P3  턴테이블 진입점  (재사용)
  Step 8  : P5  180도 반대 회전
  Step 9  : P1  베드 위 대기  (재사용)
  Step 10 : P6  분류장 위치  → 그리퍼 열기

사용법 (도커 안에서):
  python3 /workspace/scripts/test_sequence.py           # 자동 실행
  python3 /workspace/scripts/test_sequence.py --step    # 스텝 모드 (Enter로 다음 단계)
"""

import sys
import time

try:
    from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
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
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION    = 116
LEN_GOAL_POSITION     = 4

SPEED_NORMAL  = 50   # 약 20% 속도
SPEED_GRIPPER = 100

GRIPPER_OPEN  = 2300
GRIPPER_CLOSE = 1800

# ── 티칭 웨이포인트 (teach_pendant.py로 기록) ─────────────────────────────────
POSE_P1 = {'shoulder_pan': 2047, 'shoulder_lift': 1059, 'elbow_flex': 2977, 'wrist_flex': 3005, 'wrist_roll': 1994, 'gripper': 2152}  # 베드 위 대기
POSE_P2 = {'shoulder_pan':   28, 'shoulder_lift': 1025, 'elbow_flex': 2791, 'wrist_flex': 3055, 'wrist_roll': 1989, 'gripper': 2152}  # 180도 회전
POSE_P3 = {'shoulder_pan':   52, 'shoulder_lift': 1848, 'elbow_flex': 2495, 'wrist_flex': 2834, 'wrist_roll': 1993, 'gripper': 2152}  # 턴테이블 진입점
POSE_P4 = {'shoulder_pan':   20, 'shoulder_lift': 1900, 'elbow_flex': 2494, 'wrist_flex': 2833, 'wrist_roll': 1994, 'gripper': 2152}  # 턴테이블 놓기/집기 지점
POSE_P5 = {'shoulder_pan': 2068, 'shoulder_lift':  868, 'elbow_flex': 2901, 'wrist_flex': 3036, 'wrist_roll': 1997, 'gripper': 2152}  # 180도 반대 회전
POSE_P6 = {'shoulder_pan': 2016, 'shoulder_lift': 1287, 'elbow_flex': 3025, 'wrist_flex': 2753, 'wrist_roll': 2006, 'gripper': 2152}  # 분류장 위치

POSE_DELAY    = 4.0
GRIPPER_DELAY = 1.0
INSPECT_WAIT  = 3.0   # 판정 대기 (추후 실제 신호로 교체)

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


def set_speed(port, pkt, names, velocity: int):
    for name in names:
        pkt.write4ByteTxRx(port, MOTORS[name], ADDR_PROFILE_VELOCITY, velocity)


def write_pose(port, pkt, pose: dict):
    sw = GroupSyncWrite(port, pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    for name, val in pose.items():
        param = [val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF]
        sw.addParam(MOTORS[name], param)
    sw.txPacket()
    sw.clearParam()


def step(port, pkt, pose: dict, label: str, delay: float = POSE_DELAY):
    _step_num[0] += 1
    n = _step_num[0]
    print(f"  [{n:2d}] {label}")
    write_pose(port, pkt, pose)
    time.sleep(delay)
    if STEP_MODE:
        input(f"       → Enter로 다음 단계")


def gripper_open(port, pkt):
    print(f"       그리퍼: 열기")
    set_speed(port, pkt, ['gripper'], SPEED_GRIPPER)
    write_pose(port, pkt, {'gripper': GRIPPER_OPEN})
    time.sleep(GRIPPER_DELAY)
    set_speed(port, pkt, ['gripper'], SPEED_NORMAL)


def gripper_close(port, pkt):
    print(f"       그리퍼: 닫기")
    set_speed(port, pkt, ['gripper'], SPEED_GRIPPER)
    write_pose(port, pkt, {'gripper': GRIPPER_CLOSE})
    time.sleep(GRIPPER_DELAY)
    set_speed(port, pkt, ['gripper'], SPEED_NORMAL)


def run_sequence(port, pkt):
    _step_num[0] = 0
    print("\n[시퀀스 시작]")
    print("=" * 50)
    print("  스텝  │ 포즈  │ 설명")
    print("  ──────┼───────┼──────────────────────────")
    print("   1    │  P1   │ 베드 위 대기")
    print("   2    │  P2   │ 180도 회전")
    print("   3    │  P3   │ 턴테이블 진입점")
    print("   4    │  P4   │ 턴테이블 놓기 → 그리퍼 열기")
    print("   5    │  P3   │ 턴테이블 진입점 (재사용)")
    print("   6    │  P4   │ 판정 대기 → 그리퍼 닫기 (집기)")
    print("   7    │  P3   │ 턴테이블 진입점 (재사용)")
    print("   8    │  P5   │ 180도 반대 회전")
    print("   9    │  P1   │ 베드 위 대기 (재사용)")
    print("  10    │  P6   │ 분류장 위치 → 그리퍼 열기")
    print("=" * 50)

    step(port, pkt, POSE_P1, 'P1  베드 위 대기')
    step(port, pkt, POSE_P2, 'P2  180도 회전')
    step(port, pkt, POSE_P3, 'P3  턴테이블 진입점')
    step(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점')
    gripper_open(port, pkt)

    step(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (재사용)')
    step(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점 (재사용)')
    print(f"       [판정 대기 {INSPECT_WAIT}초...]")
    time.sleep(INSPECT_WAIT)
    gripper_close(port, pkt)

    step(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (재사용)')
    step(port, pkt, POSE_P5, 'P5  180도 반대 회전')
    step(port, pkt, POSE_P1, 'P1  베드 위 대기 (재사용)')
    step(port, pkt, POSE_P6, 'P6  분류장 위치')
    gripper_open(port, pkt)

    print("=" * 50)
    print("[시퀀스 완료]")


def main():
    print("=" * 50)
    print("  웨이포인트 시퀀스 테스트 (속도 20%)")
    if STEP_MODE:
        print("  [스텝 모드] 각 단계 후 Enter로 진행")
    print(f"  포트: {PORT}")
    print("=" * 50)
    print("\n[Enter]로 시작, Ctrl+C로 중단")
    input()

    port, pkt = open_bus()
    set_torque(port, pkt, True)
    set_speed(port, pkt, list(MOTORS.keys()), SPEED_NORMAL)
    print(f"[속도 설정: {SPEED_NORMAL} (약 20%)]")

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
