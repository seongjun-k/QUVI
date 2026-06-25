#!/usr/bin/env python3
"""
QUVI Teach Pendant — 로봇 웨이포인트 티칭 도구

사용법:
  python3 scripts/teach_pendant.py

키 조작:
  1~6  : 현재 관절 값을 P1~P6로 저장
  p    : 현재 저장된 모든 포인트 출력
  s    : 저장된 포인트를 robot_control_node.py 붙여넣기 형식으로 출력
  q    : 종료

티칭 절차:
  1. 실행하면 토크가 자동으로 꺼짐 → 로봇을 손으로 원하는 자세로 이동
  2. 숫자 키(1~6)를 눌러 현재 자세를 저장
  3. 's' 로 최종 출력 → robot_control_node.py의 POSE_P1~P6에 붙여넣기
"""

import sys
import termios
import tty

try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler, GroupSyncRead, COMM_SUCCESS
    )
except ImportError:
    print("dynamixel_sdk 미설치. 아래 명령으로 설치하세요:")
    print("  pip install dynamixel-sdk")
    sys.exit(1)

# ── 하드웨어 설정 ──────────────────────────────────────────────
PORT      = "/dev/ttyFollower"
BAUDRATE  = 1_000_000
PROTOCOL  = 2.0

MOTORS = {
    "shoulder_pan":  11,
    "shoulder_lift": 12,
    "elbow_flex":    13,
    "wrist_flex":    14,
    "wrist_roll":    15,
    "gripper":       16,
}

# Dynamixel XL430/XL330 control table
ADDR_TORQUE_ENABLE    = 64
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION  = 4


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
    for name, mid in MOTORS.items():
        pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, val)
    state = "ON" if enable else "OFF (손으로 자유롭게 이동 가능)"
    print(f"[토크 {state}]")


def read_positions(port, pkt) -> dict:
    positions = {}
    for name, mid in MOTORS.items():
        val, result, _ = pkt.read4ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS:
            # Dynamixel 값은 unsigned. 2147483648 이상이면 음수 보정
            if val > 2147483648:
                val -= 4294967296
            positions[name] = val
        else:
            positions[name] = -1
    return positions


def fmt_pose(label: str, positions: dict) -> str:
    items = ", ".join(f"'{k}': {v}" for k, v in positions.items())
    return f"POSE_{label} = {{{items}}}"


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def print_positions(positions: dict):
    print("\n현재 관절 raw 값 (0~4095):")
    for name, val in positions.items():
        bar = "█" * int(val / 4095 * 20)
        print(f"  {name:<15}: {val:4d}  {bar}")


POSE_LABELS = {
    '1': 'P1: 베드 위 대기',
    '2': 'P2: 180도 회전',
    '3': 'P3: 턴테이블 진입점  ← 스텝 3·5·7 공용',
    '4': 'P4: 턴테이블 놓기 지점  ← 스텝 4(놓기)·6(집기) 공용',
    '5': 'P5: 180도 반대 회전',
    '6': 'P6: 분류장 위치',
}


def main():
    print("=" * 60)
    print("  QUVI Teach Pendant")
    print("=" * 60)
    print("  키  │ 저장할 포즈")
    print("  ────┼─────────────────────────────────────────")
    for k, desc in POSE_LABELS.items():
        print(f"   {k}  │ {desc}")
    print("  ────┼─────────────────────────────────────────")
    print("   p  │ 저장된 포인트 출력")
    print("   s  │ 코드 출력 (붙여넣기용)")
    print("   c  │ 현재 관절 값 확인")
    print("   q  │ 종료")
    print("=" * 60)

    port, pkt = open_bus()
    set_torque(port, pkt, False)

    waypoints: dict[str, dict] = {}

    print("\n현재 관절 값:")
    print_positions(read_positions(port, pkt))
    print("\n키를 누르세요...")

    try:
        while True:
            key = getch()

            if key in "123456":
                idx = int(key)
                label = f"P{idx}"
                desc = POSE_LABELS[key]
                pos = read_positions(port, pkt)
                waypoints[label] = pos
                print(f"\n[저장] {label} ({desc}):")
                print_positions(pos)
                print(f"  → {fmt_pose(label, pos)}")
                print("\n키를 누르세요...")

            elif key == 'p':
                print("\n[저장된 포인트]")
                if not waypoints:
                    print("  (없음)")
                for label, pos in waypoints.items():
                    print(f"  {label}: {pos}")

            elif key == 's':
                print("\n" + "=" * 55)
                print("# robot_control_node.py 에 붙여넣기:")
                print("=" * 55)
                for label, pos in sorted(waypoints.items()):
                    print(fmt_pose(label, pos))
                if not waypoints:
                    print("  (저장된 포인트 없음 — 먼저 1~6 키로 저장하세요)")
                print("=" * 55)

            elif key == 'c':
                pos = read_positions(port, pkt)
                print_positions(pos)

            elif key in ('q', '\x03'):  # q or Ctrl+C
                break

    finally:
        set_torque(port, pkt, True)
        port.closePort()
        print("\n[종료] 토크 복구 후 포트 닫음.")


if __name__ == "__main__":
    main()
