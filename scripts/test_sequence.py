#!/usr/bin/env python3
"""
웨이포인트 시퀀스 단독 테스트 스크립트 (판정 알고리즘 없이)
Dynamixel Profile_Velocity / Profile_Acceleration 사용
→ 하드웨어 가속·감속으로 부드럽게 이동, 떨림 없음

사용법 (도커 안에서):
  python3 /workspace/scripts/test_sequence.py           # 자동 실행
  python3 /workspace/scripts/test_sequence.py --step    # 스텝 모드
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
PORT     = '/dev/ttyFollower'
BAUDRATE = 1_000_000
PROTOCOL = 2.0

MOTORS = {
    'shoulder_pan':  11,
    'shoulder_lift': 12,
    'elbow_flex':    13,
    'wrist_flex':    14,
    'wrist_roll':    15,
    'gripper':       16,
}

ADDR_TORQUE_ENABLE       = 64
ADDR_PROFILE_ACCEL       = 108   # 4 bytes, 단위: 214.577 rev/min²
ADDR_PROFILE_VELOCITY    = 112   # 4 bytes, 단위: 0.229 RPM
ADDR_PRESENT_POSITION    = 132
ADDR_GOAL_POSITION       = 116
LEN_GOAL_POSITION        = 4

# ── 속도 설정 ─────────────────────────────────────────────────────────────────
# Profile_Velocity: 단위 0.229 RPM
#   30 → 약 6.9 RPM  (매우 느림)
#   50 → 약 11.5 RPM (느림)
#   80 → 약 18.3 RPM (보통)
PROFILE_VELOCITY       = 8     # 팔 관절 속도 (0.229 RPM 단위 → 약 1.8 RPM)
PROFILE_ACCEL          = 3     # 가속도
PROFILE_VELOCITY_GRIP  = 20    # 그리퍼 속도
PROFILE_ACCEL_GRIP     = 5

# 이동 완료 대기 시간 (Profile_Velocity 기준 예상 시간 + 여유)
MOVE_WAIT              = 15.0  # 초 (관절 이동 — 느린 속도에 맞게 충분히)
GRIPPER_WAIT           = 3.0   # 초 (그리퍼)
SETTLE                 = 0.3   # 정착 여유
INSPECT_WAIT           = 3.0   # 판정 대기 (추후 실제 신호로 교체)

# ── 티칭 웨이포인트 ────────────────────────────────────────────────────────────
POSE_P1 = {'shoulder_pan': 2054, 'shoulder_lift': 1258, 'elbow_flex': 2800, 'wrist_flex': 2981, 'wrist_roll': 2035, 'gripper': 2150}
POSE_P2 = {'shoulder_pan':   12, 'shoulder_lift': 1843, 'elbow_flex': 2165, 'wrist_flex': 3123, 'wrist_roll': 2095, 'gripper': 2150}
POSE_P3 = {'shoulder_pan':   16, 'shoulder_lift': 1736, 'elbow_flex': 2413, 'wrist_flex': 3018, 'wrist_roll': 2087, 'gripper': 2150}
POSE_P4 = {'shoulder_pan':   16, 'shoulder_lift': 1841, 'elbow_flex': 2522, 'wrist_flex': 2759, 'wrist_roll': 2085, 'gripper': 2150}
POSE_P5 = {'shoulder_pan': 2047, 'shoulder_lift': 1854, 'elbow_flex': 2460, 'wrist_flex': 2909, 'wrist_roll': 2050, 'gripper': 2150}
POSE_P6 = {'shoulder_pan': 2039, 'shoulder_lift': 1076, 'elbow_flex': 2884, 'wrist_flex': 3094, 'wrist_roll': 1993, 'gripper': 2150}

STEP_MODE = '--step' in sys.argv
_step_num = [0]


def open_bus():
    port = PortHandler(PORT)
    pkt  = PacketHandler(PROTOCOL)
    if not port.openPort():
        print(f"포트 열기 실패: {PORT}"); sys.exit(1)
    if not port.setBaudRate(BAUDRATE):
        print(f"보드레이트 설정 실패: {BAUDRATE}"); sys.exit(1)
    return port, pkt


def set_torque(port, pkt, enable: bool):
    val = 1 if enable else 0
    for mid in MOTORS.values():
        pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, val)
    print(f"[토크 {'ON' if enable else 'OFF'}]")


def apply_profile(port, pkt, motor_names, velocity: int, accel: int):
    for name in motor_names:
        mid = MOTORS[name]
        pkt.write4ByteTxRx(port, mid, ADDR_PROFILE_ACCEL, accel)
        pkt.write4ByteTxRx(port, mid, ADDR_PROFILE_VELOCITY, velocity)


def write_pose(port, pkt, pose: dict):
    sw = GroupSyncWrite(port, pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    for name, val in pose.items():
        param = [val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF]
        sw.addParam(MOTORS[name], param)
    sw.txPacket()
    sw.clearParam()


def move_to(port, pkt, pose: dict, label: str):
    _step_num[0] += 1
    print(f"  [{_step_num[0]:2d}] {label}")

    arm_motors = [k for k in pose if k != 'gripper']
    if arm_motors:
        apply_profile(port, pkt, arm_motors, PROFILE_VELOCITY, PROFILE_ACCEL)

    write_pose(port, pkt, pose)
    time.sleep(MOVE_WAIT + SETTLE)

    if STEP_MODE:
        input("       → Enter로 다음 단계")


def gripper_open(port, pkt):
    print("       그리퍼: 열기")
    apply_profile(port, pkt, ['gripper'], PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)
    write_pose(port, pkt, {'gripper': 2300})
    time.sleep(GRIPPER_WAIT)


def gripper_close(port, pkt):
    print("       그리퍼: 닫기")
    apply_profile(port, pkt, ['gripper'], PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)
    write_pose(port, pkt, {'gripper': 1800})
    time.sleep(GRIPPER_WAIT)


def run_sequence(port, pkt):
    _step_num[0] = 0
    print("\n[시퀀스 시작]")
    print("=" * 55)

    move_to(port, pkt, POSE_P1, 'P1  베드 위 대기')
    move_to(port, pkt, POSE_P2, 'P2  180도 회전')
    move_to(port, pkt, POSE_P3, 'P3  턴테이블 진입점')
    move_to(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점')
    gripper_open(port, pkt)

    move_to(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (재사용)')
    move_to(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점 (재사용)')
    print(f"       [판정 대기 {INSPECT_WAIT}초...]")
    time.sleep(INSPECT_WAIT)
    gripper_close(port, pkt)

    move_to(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (재사용)')
    move_to(port, pkt, POSE_P5, 'P5  180도 반대 회전')
    move_to(port, pkt, POSE_P1, 'P1  베드 위 대기 (재사용)')
    move_to(port, pkt, POSE_P6, 'P6  분류장 위치')
    gripper_open(port, pkt)

    print("=" * 55)
    print("[시퀀스 완료]")


def main():
    print("=" * 55)
    print("  웨이포인트 시퀀스 테스트")
    print(f"  속도: Profile_Velocity={PROFILE_VELOCITY} ({PROFILE_VELOCITY * 0.229:.1f} RPM)")
    print(f"  가속: Profile_Accel={PROFILE_ACCEL}")
    if STEP_MODE:
        print("  [스텝 모드]")
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
