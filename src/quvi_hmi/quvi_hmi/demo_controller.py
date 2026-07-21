"""
QUVI DEMO_CONTROLLER
─────────────────────
demo/dashboard 브랜치 전용. 실기 없이 HMI "▶ 시작" 클릭에 반응해
녹화된 ros2 bag(PASS/FAIL 교대)을 재생, 대시보드를 채운다.
"""

import os
import subprocess

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from quvi_robot_control.topics import TOPIC_HMI_COMMAND


class DemoController(Node):

    def __init__(self):
        super().__init__('demo_controller')
        self.declare_parameter('bag_dir', '/workspace/data/demo_bags')
        self._bag_dir = self.get_parameter('bag_dir').value
        self._toggle_fail = False  # False=다음 재생은 pass, True=fail
        self._proc = None
        self.create_subscription(String, TOPIC_HMI_COMMAND, self._command_cb, 10)
        self.get_logger().info(f'demo_controller 시작 — bag_dir={self._bag_dir}')

    def _command_cb(self, msg: String):
        command = msg.data.upper()
        if command == 'START':
            self._start_playback()
        elif command in ('STOP', 'ESTOP'):
            self._stop_playback()

    def _start_playback(self):
        if self._proc is not None and self._proc.poll() is None:
            self.get_logger().info('재생 중 — START 무시')
            return
        bag_name = 'fail' if self._toggle_fail else 'pass'
        self._toggle_fail = not self._toggle_fail
        bag_path = os.path.join(self._bag_dir, bag_name)
        if not os.path.isdir(bag_path):
            self.get_logger().warn(f'bag 없음, 무시: {bag_path}')
            return
        self.get_logger().info(f'bag 재생 시작: {bag_path}')
        self._proc = subprocess.Popen(['ros2', 'bag', 'play', bag_path])

    def _stop_playback(self):
        if self._proc is not None and self._proc.poll() is None:
            self.get_logger().info('bag 재생 정지')
            self._proc.terminate()
        self._proc = None

    def destroy_node(self):
        self._stop_playback()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DemoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
