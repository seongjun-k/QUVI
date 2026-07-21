#!/usr/bin/env python3
"""
웨이포인트 시퀀스 단독 테스트 스크립트 (판정 알고리즘 없이)
소프트웨어 보간(interpolation)으로 천천히 단계적 이동

dynamixel_sdk 로 /dev/ttyFollower 를 직접 제어 — ROS 노드를 거치지 않는다.
실행 위치: quvi-dev 컨테이너 (privileged 로 시리얼 포트 접근).
사용법: python3 scripts/test_sequence.py [--step]
"""

import sys
import time

try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
        COMM_SUCCESS
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
ADDR_PROFILE_ACCEL       = 108
ADDR_PROFILE_VELOCITY    = 112
ADDR_PRESENT_POSITION    = 132
ADDR_GOAL_POSITION       = 116
LEN_GOAL_POSITION        = 4
LEN_PRESENT_POSITION     = 4

# ── 속도 설정 ─────────────────────────────────────────────────────────────────
PROFILE_VELOCITY       = 8
PROFILE_ACCEL          = 3
PROFILE_VELOCITY_GRIP  = 20
PROFILE_ACCEL_GRIP     = 5

# ── 소프트웨어 보간 설정 ───────────────────────────────────────────────────────
# 현재 → 목표 위치 사이를 INTERP_STEPS개로 나눠서 전송
# INTERP_STEPS * INTERP_DELAY = 총 이동 시간
INTERP_STEPS         = 200       # 중간 목표 개수 (많을수록 부드럽게)
INTERP_DELAY         = 0.005      # 각 스텝 사이 대기 (초) → 총 200*0.005 = 1.0초
INTERP_STEPS_GRIP    = 3       # 그리퍼 보간 스텝 수
INTERP_DELAY_GRIP    = 0.001     # 그리퍼 스텝 간 대기 → 총 3*0.001 = 0.003초

SETTLE               = 0.3
INSPECT_WAIT         = 3.0

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


def read_positions(port, pkt, motor_names: list) -> dict:
    """지정한 모터들의 현재 위치를 읽어 반환."""
    result = {}
    for name in motor_names:
        mid = MOTORS[name]
        val, _, _ = pkt.read4ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        # 부호 없는 32비트 → signed 처리 (필요 시)
        if val > 0x7FFFFFFF:
            val -= 0x100000000
        result[name] = val
    return result


def write_pose(port, pkt, pose: dict):
    sw = GroupSyncWrite(port, pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    for name, val in pose.items():
        val = int(val)
        param = [val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF]
        sw.addParam(MOTORS[name], param)
    sw.txPacket()
    sw.clearParam()


def interp_move(port, pkt, target_pose: dict, steps: int, delay: float,
                velocity: int, accel: int):
    """
    현재 위치에서 target_pose까지 steps개의 중간 목표로 나눠서 이동.
    각 중간 목표 사이에 delay초 대기.
    """
    motor_names = list(target_pose.keys())

    # Profile 설정
    apply_profile(port, pkt, motor_names, velocity, accel)

    # 현재 위치 읽기
    current = read_positions(port, pkt, motor_names)

    for i in range(1, steps + 1):
        t = i / steps  # 0.0 초과 ~ 1.0
        # 선형 보간: current + t * (target - current)
        interp_pose = {
            name: int(current[name] + t * (target_pose[name] - current[name]))
            for name in motor_names
        }
        write_pose(port, pkt, interp_pose)
        time.sleep(delay)

    time.sleep(SETTLE)


def move_to(port, pkt, pose: dict, label: str):
    _step_num[0] += 1
    print(f"  [{_step_num[0]:2d}] {label}")

    arm_motors = {k: v for k, v in pose.items() if k != 'gripper'}
    grip_motors = {k: v for k, v in pose.items() if k == 'gripper'}

    # 팔 관절 보간 이동
    if arm_motors:
        interp_move(port, pkt, arm_motors,
                    INTERP_STEPS, INTERP_DELAY,
                    PROFILE_VELOCITY, PROFILE_ACCEL)

    # 그리퍼도 pose에 포함된 경우 함께 이동
    if grip_motors:
        interp_move(port, pkt, grip_motors,
                    INTERP_STEPS_GRIP, INTERP_DELAY_GRIP,
                    PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)

    if STEP_MODE:
        input("       → Enter로 다음 단계")


def gripper_open(port, pkt):
    print("       그리퍼: 열기")
    interp_move(port, pkt, {'gripper': 2300},
                INTERP_STEPS_GRIP, INTERP_DELAY_GRIP,
                PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)


def gripper_close(port, pkt):
    print("       그리퍼: 닫기")
    interp_move(port, pkt, {'gripper': 1800},
                INTERP_STEPS_GRIP, INTERP_DELAY_GRIP,
                PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)


# ── 티칭 웨이포인트 ────────────────────────────────────────────────────────────
POSE_P1 = {'shoulder_pan': 2054, 'shoulder_lift': 1258, 'elbow_flex': 2800, 'wrist_flex': 2981, 'wrist_roll': 2035, 'gripper': 2150}
POSE_P2 = {'shoulder_pan':   12, 'shoulder_lift': 1843, 'elbow_flex': 2165, 'wrist_flex': 3123, 'wrist_roll': 2095, 'gripper': 2150}
POSE_P3 = {'shoulder_pan':   16, 'shoulder_lift': 1736, 'elbow_flex': 2413, 'wrist_flex': 3018, 'wrist_roll': 2087, 'gripper': 2150}
POSE_P4 = {'shoulder_pan':   26, 'shoulder_lift': 1776, 'elbow_flex': 2537, 'wrist_flex': 2825, 'wrist_roll': 2019, 'gripper': 2067}
POSE_P5 = {'shoulder_pan': 2047, 'shoulder_lift': 1854, 'elbow_flex': 2460, 'wrist_flex': 2909, 'wrist_roll': 2050, 'gripper': 2150}
POSE_P6 = {'shoulder_pan': 2039, 'shoulder_lift': 1076, 'elbow_flex': 2884, 'wrist_flex': 3094, 'wrist_roll': 1993, 'gripper': 2150}


def run_sequence(port, pkt):
    _step_num[0] = 0
    print("\n[시퀀스 시작]")
    print("=" * 55)

    # 1. 픽업
    move_to(port, pkt, POSE_P1, 'P1  베드 위 대기')
    move_to(port, pkt, POSE_P2, 'P2  180도 회전')
    move_to(port, pkt, POSE_P3, 'P3  턴테이블 진입점')
    move_to(port, pkt, POSE_P4, 'P4  턴테이블 놓기 지점')
    gripper_open(port, pkt)                                  # 물건 놓기

    # 2. 진입점으로 빠져나온 뒤 판정 대기
    move_to(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (대기)')
    print(f"       [판정 대기 {INSPECT_WAIT}초...]")
    time.sleep(INSPECT_WAIT)

    # 3. 다시 진입해서 집기
    move_to(port, pkt, POSE_P4, 'P4  턴테이블 집기 지점')
    gripper_close(port, pkt)                                 # 물건 집기

    # 4. 복귀 및 분류
    move_to(port, pkt, POSE_P3, 'P3  턴테이블 진입점 (복귀)')
    move_to(port, pkt, POSE_P5, 'P5  180도 반대 회전')
    move_to(port, pkt, POSE_P1, 'P1  베드 위 대기 (재사용)')
    move_to(port, pkt, POSE_P6, 'P6  분류장 위치')
    gripper_open(port, pkt)                                  # 분류 완료

    print("=" * 55)
    print("[시퀀스 완료]")


POSES = {
    '1': (POSE_P1, 'P1  베드 위 대기'),
    '2': (POSE_P2, 'P2  180도 회전'),
    '3': (POSE_P3, 'P3  턴테이블 진입점'),
    '4': (POSE_P4, 'P4  턴테이블 놓기/집기 지점'),
    '5': (POSE_P5, 'P5  180도 반대 회전'),
    '6': (POSE_P6, 'P6  분류장 위치'),
}


def main():
    print("=" * 55)
    print("  웨이포인트 테스트 (소프트웨어 보간 모드)")
    print(f"  보간 스텝: {INTERP_STEPS}개 / 스텝 간격: {INTERP_DELAY}s")
    print(f"  총 이동 시간 (팔): 약 {INTERP_STEPS * INTERP_DELAY:.1f}초")
    print(f"  Profile_Velocity={PROFILE_VELOCITY} ({PROFILE_VELOCITY * 0.229:.1f} RPM)")
    if STEP_MODE:
        print("  [스텝 모드]")
    print(f"  포트: {PORT}")
    print("=" * 55)
    print("  1~6 : 해당 포즈로 단독 이동")
    print("  o/c : 그리퍼 열기/닫기")
    print("  a   : 전체 시퀀스 실행")
    print("  q   : 종료 (Ctrl+C 동일)")
    print("=" * 55)

    port, pkt = open_bus()
    set_torque(port, pkt, True)

    try:
        while True:
            try:
                key = input("\n명령 (1~6/o/c/a/q) > ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n[종료]")
                break

            if key in POSES:
                pose, label = POSES[key]
                _step_num[0] = 0
                try:
                    move_to(port, pkt, pose, label)
                except KeyboardInterrupt:
                    print("\n[이동 중단]")
            elif key == 'o':
                gripper_open(port, pkt)
            elif key == 'c':
                gripper_close(port, pkt)
            elif key == 'a':
                try:
                    run_sequence(port, pkt)
                except KeyboardInterrupt:
                    print("\n[중단]")
            elif key == 'q':
                print("[종료]")
                break
            elif key:
                print("  잘못된 입력")
    finally:
        port.closePort()
        print("[포트 닫음]")


if __name__ == '__main__':
    main()
