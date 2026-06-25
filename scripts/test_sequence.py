#!/usr/bin/env python3
"""
웨이포인트 시퀀스 단독 테스트 스크립트 (판정 알고리즘 없이)

ROS/오케스트레이터 없이 직접 모터를 제어하여 P1~P6 시퀀스를 실행합니다.

사용법:
  python3 scripts/test_sequence.py

주의:
  - qrun(도커 컨테이너) 밖, 호스트에서 실행하거나
  - 도커 안에서 실행할 경우 /dev/ttyFollower 접근 권한 확인
  - 실행 전 robot_control_node 등 다른 프로세스가 /dev/ttyFollower를 점유하지 않아야 함
"""

import sys
import time

# ── lerobot 경로 추가 ──────────────────────────────────────────────────────────
sys.path.insert(0, '/home/ksj/QUVI/lerobot')

try:
    from lerobot.robots.omx_follower import OmxFollower
    from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
except ImportError as e:
    print(f"lerobot 임포트 실패: {e}")
    print("경로 확인: /home/ksj/QUVI/lerobot")
    sys.exit(1)

# ── 하드웨어 설정 ──────────────────────────────────────────────────────────────
PORT     = '/dev/ttyFollower'
BAUDRATE = 1_000_000

GRIPPER_OPEN  = 2300
GRIPPER_CLOSE = 1800

# ── 티칭 웨이포인트 ────────────────────────────────────────────────────────────
POSE_P1 = {'shoulder_pan': 2047, 'shoulder_lift': 1059, 'elbow_flex': 2977, 'wrist_flex': 3005, 'wrist_roll': 1994, 'gripper': 2152}
POSE_P2 = {'shoulder_pan':   28, 'shoulder_lift': 1025, 'elbow_flex': 2791, 'wrist_flex': 3055, 'wrist_roll': 1989, 'gripper': 2152}
POSE_P3 = {'shoulder_pan':   52, 'shoulder_lift': 1848, 'elbow_flex': 2495, 'wrist_flex': 2834, 'wrist_roll': 1993, 'gripper': 2152}
POSE_P4 = {'shoulder_pan':   20, 'shoulder_lift': 1900, 'elbow_flex': 2494, 'wrist_flex': 2833, 'wrist_roll': 1994, 'gripper': 2152}
POSE_P5 = {'shoulder_pan': 2068, 'shoulder_lift':  868, 'elbow_flex': 2901, 'wrist_flex': 3036, 'wrist_roll': 1997, 'gripper': 2152}
POSE_P6 = {'shoulder_pan': 2016, 'shoulder_lift': 1287, 'elbow_flex': 3025, 'wrist_flex': 2753, 'wrist_roll': 2006, 'gripper': 2152}

# 각 포즈 이동 후 대기 시간 (초)
POSE_DELAY    = 1.5
GRIPPER_DELAY = 0.5


def move_to(follower: OmxFollower, pose: dict, label: str, delay: float = POSE_DELAY):
    print(f"  → {label}")
    follower.bus.sync_write('Goal_Position', pose, normalize=False)
    time.sleep(delay)


def gripper(follower: OmxFollower, position: int, label: str):
    print(f"  → 그리퍼: {label} ({position})")
    follower.bus.sync_write('Goal_Position', {'gripper': position}, normalize=False)
    time.sleep(GRIPPER_DELAY)


def run_sequence(follower: OmxFollower):
    print("\n[시퀀스 시작]")
    print("=" * 50)

    # P1: 집기 전 대기 자세
    move_to(follower, POSE_P1, 'P1: 집기 전 대기')

    # P2: 180도 회전 후 자세
    move_to(follower, POSE_P2, 'P2: 180도 회전')

    # P3: 턴테이블 위
    move_to(follower, POSE_P3, 'P3: 턴테이블 위')

    # P4: 내려가서 놓기
    move_to(follower, POSE_P4, 'P4: 내려가기')
    gripper(follower, GRIPPER_OPEN, '열기 (놓기)')

    # P5: 위로 올라가기 (판정 대기)
    move_to(follower, POSE_P5, 'P5: 위로 올라가기')
    print("  [판정 대기 중... 3초]")
    time.sleep(3.0)

    # P4: 다시 내려가서 집기
    move_to(follower, POSE_P4, 'P4: 다시 내려가기')
    gripper(follower, GRIPPER_CLOSE, '닫기 (집기)')

    # P3: 들어올리기
    move_to(follower, POSE_P3, 'P3: 들어올리기')

    # P5: 반대쪽으로 회전
    move_to(follower, POSE_P5, 'P5: 회전')

    # P6: 최종 내려놓기
    move_to(follower, POSE_P6, 'P6: 최종 위치')
    gripper(follower, GRIPPER_OPEN, '열기 (최종 놓기)')

    print("=" * 50)
    print("[시퀀스 완료]")


def main():
    print("=" * 50)
    print("  웨이포인트 시퀀스 테스트")
    print(f"  포트: {PORT}  |  보드레이트: {BAUDRATE}")
    print("=" * 50)
    print("\n[Enter]로 시작, Ctrl+C로 중단")
    input()

    print("[OmxFollower 연결 중...]")
    config = OmxFollowerConfig(port=PORT, id='test_follower')
    follower = OmxFollower(config)

    try:
        follower.connect()
        print(f"[연결 완료] 모터: {list(follower.bus.motors.keys())}")

        while True:
            try:
                run_sequence(follower)
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
        follower.disconnect()
        print("[연결 해제 완료]")


if __name__ == '__main__':
    main()
