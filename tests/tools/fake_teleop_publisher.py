"""하드웨어 없이 수집 앱을 끝까지 돌려보는 가짜 퍼블리셔 (컨테이너 quvi-dev 에서 실행).

camera1·camera3 에 30fps 합성 JPEG, /robot/teleop_record_state 에 30Hz [present6, goal6](라디안).
시나리오 반복: 홈 정지 2초 → 움직임 4초(중간에 그리퍼 닫힘) → 홈 복귀·정지 2초(그리퍼 닫힌 채) → 그리퍼 재개방 0.5초.

  source /opt/ros/jazzy/setup.bash
  PYTHONPATH=/workspace/src/quvi_robot_control:$PYTHONPATH python3 /workspace/tests/tools/fake_teleop_publisher.py
"""
import math
import random
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

from quvi_robot_control import topics
from quvi_robot_control.act_common import JOINT_NAMES, RECORD_START_POSE, RECORD_STATE_HZ, raw_to_rad

HOME = [RECORD_START_POSE[j] for j in JOINT_NAMES]
AMP = [200, 150, -200, 150, 100]   # 팔 5관절 최대 이동량(raw)
REST, MOVE, HOLD, REOPEN = 2.0, 4.0, 2.0, 0.5
CYCLE = REST + MOVE + HOLD + REOPEN
GRIPPER_CLOSED = 1900


def pose_raw(t: float) -> list:
    """사이클 안 시각 t 의 6관절 raw (노이즈 없음)."""
    t %= CYCLE
    arm = [0.0] * 5
    grip = HOME[5]
    if REST <= t < REST + MOVE:
        s = (t - REST) / MOVE
        arm = [a * (1 - math.cos(2 * math.pi * s)) / 2 for a in AMP]
        grip = HOME[5] - (HOME[5] - GRIPPER_CLOSED) * min(1.0, max(0.0, (s - 0.4) / 0.1))
    elif t >= REST + MOVE:
        grip = GRIPPER_CLOSED
        if t >= REST + MOVE + HOLD:
            grip = GRIPPER_CLOSED + (HOME[5] - GRIPPER_CLOSED) * (t - REST - MOVE - HOLD) / REOPEN
    return [h + a for h, a in zip(HOME, arm)] + [grip]


class FakeTeleop(Node):
    def __init__(self):
        super().__init__('fake_teleop_publisher')
        self._t_start = time.monotonic()
        self._state_pub = self.create_publisher(Float32MultiArray, topics.TOPIC_ROBOT_TELEOP_RECORD_STATE, 10)
        self._cam_pub = {
            'camera1': self.create_publisher(CompressedImage, '/camera1/image_raw/compressed', 10),
            'camera3': self.create_publisher(CompressedImage, '/camera3/image_raw/compressed', 10)}
        self.create_timer(1.0 / RECORD_STATE_HZ, self._pub_state)
        self.create_timer(1.0 / 30, self._pub_cams)
        self._n = 0

    def _pub_state(self):
        t = time.monotonic() - self._t_start
        present = [r + random.uniform(-1, 1) for r in pose_raw(t)]
        goal = pose_raw(t + 0.1)  # 목표는 현재보다 살짝 앞선다
        msg = Float32MultiArray()
        msg.data = [raw_to_rad(r) for r in present] + [raw_to_rad(r) for r in goal]
        self._state_pub.publish(msg)

    def _pub_cams(self):
        self._n += 1
        for i, (cam, pub) in enumerate(self._cam_pub.items()):
            img = np.full((480, 640, 3), 40 + 60 * i, np.uint8)
            x = int(50 + (self._n * 4 + i * 200) % 500)
            cv2.rectangle(img, (x, 200), (x + 60, 260), (0, 200, 255), -1)
            cv2.putText(img, f'{cam} #{self._n}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            ok, buf = cv2.imencode('.jpg', img)
            if not ok:
                continue
            msg = CompressedImage()
            msg.format = 'jpeg'
            msg.data = buf.tobytes()
            pub.publish(msg)


def main():
    rclpy.init()
    node = FakeTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
