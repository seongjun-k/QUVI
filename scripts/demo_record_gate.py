#!/usr/bin/env python3
"""데모 bag 녹화 게이트.

HMI "시작" 버튼(/hmi/command = "START")을 누른 시점부터 사이클이 끝날 때까지만
녹화해, bag 앞뒤에 유휴 구간이 붙지 않게 한다. record_demo.sh 가 컨테이너 안에서 호출한다.

사용법: python3 demo_record_gate.py <출력 bag 경로> <토픽> [<토픽> ...]
"""

import signal
import subprocess
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from quvi_msgs.msg import SystemStatus
from quvi_robot_control.topics import TOPIC_HMI_COMMAND, TOPIC_HMI_STATUS

# 사이클 시작 전·종료 후의 대기 상태. 여기로 복귀하면 1사이클 완료로 본다.
IDLE_STATES = ('IDLE', 'INIT')


class RecordGate(Node):
    def __init__(self, out_path, topics):
        super().__init__('demo_record_gate')
        self._out_path = out_path
        self._topics = topics
        self._proc = None
        # START 직후에도 FSM은 잠시 IDLE로 남아 있다(status 주기 0.5s) —
        # 한 번 IDLE을 벗어난 뒤에야 복귀를 종료 신호로 인정한다.
        self._left_idle = False
        self.done = False

        self.create_subscription(String, TOPIC_HMI_COMMAND, self._command_cb, 10)
        self.create_subscription(SystemStatus, TOPIC_HMI_STATUS, self._status_cb, 10)
        self.get_logger().info('대기 중 — 대시보드에서 "시작"을 누르면 녹화를 시작한다')

    def _command_cb(self, msg):
        if self._proc is not None or msg.data.strip().upper() != 'START':
            return
        self._proc = subprocess.Popen(
            ['ros2', 'bag', 'record', '-o', self._out_path, *self._topics])
        self.get_logger().info(f'녹화 시작 → {self._out_path}')

    def _status_cb(self, msg):
        if self._proc is None:
            return
        if msg.current_state not in IDLE_STATES:
            self._left_idle = True
        elif self._left_idle:
            self.done = True

    def stop(self):
        if self._proc is None:
            return
        # SIGINT 라야 rosbag2가 metadata.yaml을 정상 기록하고 닫는다 (kill 하면 bag이 깨진다).
        self._proc.send_signal(signal.SIGINT)
        try:
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self.get_logger().warn('bag record 종료 지연 — 강제 종료했다. bag이 손상됐을 수 있다')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    rclpy.init()
    node = RecordGate(sys.argv[1], sys.argv[2:])
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.done:
            node.get_logger().info('사이클 종료 감지 — 녹화를 마친다')
    except KeyboardInterrupt:
        node.get_logger().info('중단 요청 — 녹화를 마친다')
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
