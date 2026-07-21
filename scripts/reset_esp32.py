#!/usr/bin/env python3
"""ESP32-S3 하드 리셋 스크립트.

micro-ROS agent 재기동 시 이전 세션에 붙어 있던 ESP32 는 재협상 로직이
없어 새 agent 에 다시 연결되지 못한다 (firmware 는 부팅 시에만
rclc_support_init 재시도 루프를 도는 구조). ESP32 는 CP210x/CH340
USB-시리얼(DTR/RTS 리셋 회로) 로 연결되어 있으므로, agent 기동 직전에
DTR/RTS 를 esptool 의 hard_reset 과 동일한 순서로 펄스를 주어 ESP32 를
프레시 부팅시키면 재시도 루프가 새 agent 를 잡는다.

best-effort 스크립트다: 포트가 없거나 열기 실패해도 launch 체인을
막지 않도록 항상 exit code 0 으로 종료한다.

사용법:
  python3 scripts/reset_esp32.py                      # 기본 /dev/ttyESP32
  python3 scripts/reset_esp32.py --port /dev/ttyUSB0
  python3 scripts/reset_esp32.py --pulse-sec 0.2
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    print("[reset_esp32] 경고: pyserial 이 설치되어 있지 않습니다. 리셋을 건너뜁니다.", file=sys.stderr)
    sys.exit(0)


def reset_esp32(port: str, pulse_sec: float) -> bool:
    """DTR/RTS 를 펄스하여 ESP32 를 하드 리셋한다.

    esptool hard_reset 과 동일한 순서(DTR 먼저 내려 IO0=HIGH 확보 후
    EN 펄스)를 따른다 — 순서가 바뀌면 부트로더 모드로 진입할 수 있다.
    """
    try:
        ser = serial.Serial(port)
        # open 시 리눅스가 DTR/RTS 를 동시에 assert 하므로, 먼저 IO0(DTR)을
        # HIGH 로 되돌려 정상 부팅 모드를 확보한 뒤 EN(RTS) 만 펄스한다.
        ser.dtr = False   # IO0 = HIGH (정상 부팅 모드)
        ser.rts = True    # EN = LOW (리셋)
        time.sleep(pulse_sec)
        ser.rts = False   # EN = HIGH (부팅 시작)
        ser.close()
    except Exception as exc:
        print(f"[reset_esp32] 경고: {port} 리셋 실패 ({exc}). 건너뜁니다.", file=sys.stderr)
        return False

    print(f"[reset_esp32] ESP32 하드 리셋 완료 ({port})")
    return True


def main():
    parser = argparse.ArgumentParser(description="ESP32-S3 DTR/RTS 하드 리셋")
    parser.add_argument('--port', default='/dev/ttyESP32', help='ESP32 시리얼 포트 (기본: /dev/ttyESP32)')
    parser.add_argument('--pulse-sec', type=float, default=0.1, help='EN 리셋 펄스 길이(초, 기본 0.1)')
    args = parser.parse_args()

    reset_esp32(args.port, args.pulse_sec)
    # best-effort: 실패해도 launch 체인을 막지 않도록 항상 0으로 종료
    sys.exit(0)


if __name__ == '__main__':
    main()
