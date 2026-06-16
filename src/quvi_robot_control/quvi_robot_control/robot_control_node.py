"""
QUVI ROBOT_CONTROL_NODE
────────────────────────────────────────────────────────────────
로봇팔(OMX AI Manipulator) + 리니어 레일 + 턴테이블을
통합 제어하는 노드.

모터 구성:
  리더  ID 1~6 : DYNAMIXEL XL330-M288T / XL330-M077T (5V)
  팔로워 ID 11~13: DYNAMIXEL XL430-W250T (12V) — 베이스, 숄더, 엘보우
  팔로워 ID 14~16: DYNAMIXEL XL330-M288T (5V)  — 리스트, 그리퍼

모터 제어:
  lerobot 공식 OmxFollower / OmxLeader 코드를 그대로 사용.
  - DynamixelMotorsBus (GroupSyncRead/Write, 캘리브레이션, 정규화)
  - OmxFollower (팔로워 로봇: connect, get_observation, send_action)
  - OmxLeader  (리더 텔레오퍼레이터: connect, get_action)

주요 기능:
  1. ACT 모방학습 파지 (LeRobot ACTPolicy)
     - /camera/handcam 이미지 + 관절 상태 → ACT 추론 → 관절 목표값 전송
  2. OMX Dynamixel 관절 제어 (lerobot 공식 API)
     - 홈 복귀, 자세 이동, 그리퍼 제어
  3. 리더-팔로워 텔레오퍼레이션 (lerobot OmxLeader → OmxFollower)
  4. 레일 이동 명령 → ESP32-S3 (/motor/rail)
  5. 턴테이블 회전 명령 → ESP32-S3 (/motor/turntable)

ROS 2 인터페이스 (Subscriber):
  /camera/handcam/compressed  sensor_msgs/CompressedImage   핸드캠 이미지
  /robot/grasp_command        quvi_msgs/GraspGoal           파지 트리거 + 목표 좌표
  /robot/rail_command         std_msgs/Int32                레일 목표 위치 코드 (0=D,1=A,2=B,3=C)
  /robot/rotate_command       std_msgs/Bool                 베이스 180° 회전 (true=뒤, false=앞)
  /robot/release_command      std_msgs/Bool                 출력물 투하
  /robot/home_command         std_msgs/Bool                 홈 복귀

ROS 2 인터페이스 (Publisher):
  /robot/joint_states         sensor_msgs/JointState        현재 관절 각도 (30 Hz)
  /motor/rail                 std_msgs/Int32                레일 목표 스텝 (→ ESP32)
  /motor/turntable            std_msgs/Int32                턴테이블 목표 각도 (→ ESP32)
  /robot/status               std_msgs/String               상태 문자열
  /robot/act_done             std_msgs/Bool                 ACT 파지 완료 신호
  /robot/grasp_done           std_msgs/Bool                 파지/투하 완료 신호
  /robot/rail_done            std_msgs/Bool                 레일 이동 완료 신호

ROS 2 서비스 (Server):
  /robot/act_grasp            std_srvs/Trigger              ACT 파지 실행 (동기)
  /robot/go_home              std_srvs/Trigger              홈 복귀 실행 (동기)
  /robot/open_gripper         std_srvs/Trigger              그리퍼 열기
  /robot/close_gripper        std_srvs/Trigger              그리퍼 닫기
"""

import math
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger

from quvi_msgs.msg import GraspGoal

# ─────────────────────────────────────────────────────────────
# lerobot 공식 코드 import
# ─────────────────────────────────────────────────────────────
LEROBOT_SRC = str(Path(__file__).resolve().parents[3] / 'lerobot' / 'src')
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

from lerobot.robots.omx_follower import OmxFollower
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.teleoperators.omx_leader import OmxLeader
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig


# ─────────────────────────────────────────────────────────────
# 상수 및 열거형
# ─────────────────────────────────────────────────────────────

class RailPosition(IntEnum):
    """레일 정지 위치 코드 (Main Orchestrator가 Int32로 전송)."""
    BED     = 0   # X=D  3D 프린터 베드
    INSPECT = 1   # X=A  검사장
    PASS    = 2   # X=B  PASS 분류함
    FAIL    = 3   # X=C  FAIL 분류함


class RobotState(IntEnum):
    """노드 내부 상태머신."""
    IDLE          = 0
    HOMING        = 1
    MOVING_RAIL   = 2
    ROTATING_BASE = 3
    ACT_GRASPING  = 4
    PLACING       = 5
    RELEASING     = 6
    TURNTABLE     = 7
    TELEOPING     = 8
    ERROR         = 99


# 관절 이름 (ROS 2 JointState 메시지용)
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# 사전 정의 자세 — raw Dynamixel 위치값 (0~4095 = 0~360°)
# dict 형태로 lerobot bus에 직접 전달 (normalize=False)
POSE_HOME = {
    'shoulder_pan': 2048, 'shoulder_lift': 1800, 'elbow_flex': 1200,
    'wrist_flex': 2048, 'wrist_roll': 2048, 'gripper': 2048,
}
POSE_FRONT = {
    'shoulder_pan': 2048, 'shoulder_lift': 1400, 'elbow_flex': 900,
    'wrist_flex': 1800, 'wrist_roll': 2048, 'gripper': 2300,
}
POSE_BACK = {
    'shoulder_pan': 2048, 'shoulder_lift': 1400, 'elbow_flex': 900,
    'wrist_flex': 1800, 'wrist_roll': 2048, 'gripper': 2300,
}
POSE_PLACE = {
    'shoulder_pan': 2048, 'shoulder_lift': 1600, 'elbow_flex': 1100,
    'wrist_flex': 2048, 'wrist_roll': 2048, 'gripper': 2300,
}

# 그리퍼 raw 위치값 (XL330-M288T)
GRIPPER_OPEN  = 2300
GRIPPER_CLOSE = 1800

# ACT 실행 주기 (Hz)
ACT_CONTROL_HZ = 30


# ─────────────────────────────────────────────────────────────
# RobotControlNode
# ─────────────────────────────────────────────────────────────

class RobotControlNode(Node):
    """로봇팔 + 레일 + 턴테이블 통합 제어 노드.
    
    모터 제어는 lerobot 공식 OmxFollower / OmxLeader를 사용.
    """

    def __init__(self):
        super().__init__('robot_control_node')

        # ─── 파라미터 선언 ───
        self._declare_params()
        self._load_params()

        # ─── 내부 상태 ───
        self._state: RobotState = RobotState.IDLE
        self._state_lock = threading.Lock()

        self._latest_handcam: Optional[np.ndarray] = None
        self._handcam_lock = threading.Lock()
        self._bridge = CvBridge()

        self._esp32_rail_done = False

        # ─── 텔레오퍼레이션 상태 ───
        self._teleop_running = False
        self._leader: Optional[OmxLeader] = None

        # ─── 콜백 그룹 (블로킹 서비스/텔레옵과 타이머가 서로를 막지 않도록) ───
        self._cb_group = ReentrantCallbackGroup()

        # ─── lerobot bus I/O 직렬화 락 ───
        # MultiThreadedExecutor 환경에서 30Hz joint 발행 타이머(read)와
        # ACT/텔레옵 루프(write)가 동일 시리얼 포트에 동시 접근하면
        # 패킷이 섞일 수 있으므로 모든 I/O를 이 락으로 직렬화한다.
        self._dxl_io_lock = threading.Lock()

        # ─── lerobot OmxFollower 초기화 ───
        self._follower: Optional[OmxFollower] = None
        self._dxl_ready = False
        if self._use_real_hardware:
            self._init_follower()

        # ─── ACT 모델 로드 ───
        self._act_policy = None
        self._act_ready = False
        if self._use_act:
            self._load_act_policy()

        # ─── ROS 통신 ───
        self._setup_ros_interfaces()

        # ─── 관절 상태 발행 타이머 (30 Hz) ───
        self._joint_pub_timer = self.create_timer(
            1.0 / ACT_CONTROL_HZ, self._publish_joint_states,
            callback_group=self._cb_group)

        self.get_logger().info(
            f'ROBOT_CONTROL_NODE 초기화 완료 (lerobot 공식 코드 사용) | '
            f'하드웨어={self._use_real_hardware} | '
            f'ACT={self._use_act} | '
            f'팔로워 포트={self._dxl_port_name} | '
            f'리더 포트={self._leader_port_name}')

    # ─────────────────────────────────────────────
    # 파라미터
    # ─────────────────────────────────────────────
    def _declare_params(self):
        # 하드웨어
        self.declare_parameter('use_real_hardware', True)
        self.declare_parameter('dxl_port', '/dev/ttyFollower')
        self.declare_parameter('leader_dxl_port', '/dev/ttyLeader')
        # ACT
        self.declare_parameter('use_act', True)
        self.declare_parameter('act_model_path',
            'outputs/train/quvi_act/checkpoints/last/pretrained_model')
        self.declare_parameter('act_chunk_size', 20)
        self.declare_parameter('act_device', 'cpu')   # 'cuda' or 'cpu'
        # 레일 위치 (mm 단위) — 조립 후 캘리브레이션으로 확정
        self.declare_parameter('rail_mm_bed',      0.0)
        self.declare_parameter('rail_mm_inspect', 12.5)
        self.declare_parameter('rail_mm_pass',    21.25)
        self.declare_parameter('rail_mm_fail',    30.0)
        # 카메라
        self.declare_parameter('handcam_topic', '/camera1/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        # 동작 타임아웃 (초)
        self.declare_parameter('rail_move_timeout_sec', 30.0)
        self.declare_parameter('grasp_timeout_sec', 20.0)
        self.declare_parameter('home_timeout_sec', 10.0)

    def _load_params(self):
        self._use_real_hardware = self.get_parameter('use_real_hardware').value
        self._dxl_port_name     = self.get_parameter('dxl_port').value
        self._leader_port_name  = self.get_parameter('leader_dxl_port').value
        self._use_act           = self.get_parameter('use_act').value
        self._act_model_path    = self.get_parameter('act_model_path').value
        self._act_chunk_size    = self.get_parameter('act_chunk_size').value
        self._act_device        = self.get_parameter('act_device').value
        self._rail_mm = {
            RailPosition.BED:     self.get_parameter('rail_mm_bed').value,
            RailPosition.INSPECT: self.get_parameter('rail_mm_inspect').value,
            RailPosition.PASS:    self.get_parameter('rail_mm_pass').value,
            RailPosition.FAIL:    self.get_parameter('rail_mm_fail').value,
        }
        self._handcam_topic  = self.get_parameter('handcam_topic').value
        self._use_compressed = self.get_parameter('use_compressed').value
        self._rail_timeout   = self.get_parameter('rail_move_timeout_sec').value
        self._grasp_timeout  = self.get_parameter('grasp_timeout_sec').value
        self._home_timeout   = self.get_parameter('home_timeout_sec').value

    # ─────────────────────────────────────────────
    # lerobot OmxFollower 초기화
    # ─────────────────────────────────────────────
    def _init_follower(self):
        """lerobot 공식 OmxFollower를 통해 팔로워 로봇팔 초기화.

        OmxFollower가 내부적으로 처리하는 것들:
          - DynamixelMotorsBus 생성 (GroupSyncRead/Write 포함)
          - 모터 ping 및 모델 검증
          - 캘리브레이션 로드/적용 (homing offset, drive mode, range limit)
          - Operating Mode 설정 (shoulder_pan=EXTENDED_POSITION, gripper=CURRENT_POSITION 등)
          - PID 게인, Profile Velocity/Acceleration 설정
          - Position Limit, Current Limit 설정
          - 토크 활성화
        """
        try:
            follower_config = OmxFollowerConfig(
                port=self._dxl_port_name,
                id='quvi_follower',
            )
            self._follower = OmxFollower(follower_config)
            self._follower.connect()
            self._dxl_ready = True
            self.get_logger().info(
                f'OmxFollower 연결 완료 | 포트={self._dxl_port_name} | '
                f'모터: {list(self._follower.bus.motors.keys())}')
        except Exception as e:
            self.get_logger().error(f'OmxFollower 초기화 실패: {e}')
            self._dxl_ready = False

    # ─────────────────────────────────────────────
    # ACT 모델 로드
    # ─────────────────────────────────────────────
    def _load_act_policy(self):
        """LeRobot ACTPolicy 로드."""
        try:
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError as e:
            self.get_logger().error(f'LeRobot/torch 미설치: {e}')
            return

        resolved_path = Path(self._act_model_path)
        if not resolved_path.is_absolute():
            resolved_path = Path('/workspace') / resolved_path
        resolved_path = resolved_path.resolve()

        self.get_logger().info(f'ACT 모델 로드 중: {resolved_path}')
        try:
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
            if not resolved_path.exists():
                raise FileNotFoundError(f'로컬 모델 디렉토리가 존재하지 않습니다: {resolved_path}')
            self._act_policy = ACTPolicy.from_pretrained(str(resolved_path))
            self._act_policy.eval()
            device = self._act_device
            self._act_policy = self._act_policy.to(device)
            self._act_device_obj = device
            self._act_ready = True
            self.get_logger().info(f'ACT 모델 로드 완료 (device={device})')
        except Exception as e:
            self.get_logger().error(f'ACT 모델 로드 실패: {e}')
            self._act_ready = False

    # ─────────────────────────────────────────────
    # ROS 인터페이스 설정
    # ─────────────────────────────────────────────
    def _setup_ros_interfaces(self):
        # ── Subscribers ──
        if self._use_compressed:
            self._handcam_sub = self.create_subscription(
                CompressedImage, self._handcam_topic,
                self._handcam_callback, 10)
        else:
            from sensor_msgs.msg import Image
            self._handcam_sub = self.create_subscription(
                Image, self._handcam_topic,
                self._handcam_callback_raw, 10)

        self._grasp_cmd_sub = self.create_subscription(
            GraspGoal, '/robot/grasp_command',
            self._grasp_cmd_callback, 10)

        self._rail_cmd_sub = self.create_subscription(
            Int32, '/robot/rail_command',
            self._rail_cmd_callback, 10)

        self._esp32_rail_done_sub = self.create_subscription(
            Bool, '/motor/rail_done',
            self._esp32_rail_done_callback, 10)

        self._rotate_cmd_sub = self.create_subscription(
            Bool, '/robot/rotate_command',
            self._rotate_cmd_callback, 10)

        self._release_cmd_sub = self.create_subscription(
            Bool, '/robot/release_command',
            self._release_cmd_callback, 10)

        self._home_cmd_sub = self.create_subscription(
            Bool, '/robot/home_command',
            self._home_cmd_callback, 10)

        self._teleop_cmd_sub = self.create_subscription(
            Bool, '/robot/teleop_command',
            self._teleop_cmd_callback, 10,
            callback_group=self._cb_group)

        self._estop_sub = self.create_subscription(
            Bool, '/system/estop',
            self._estop_cmd_callback, 10,
            callback_group=self._cb_group)

        # ── Publishers ──
        self._joint_state_pub = self.create_publisher(
            JointState, '/robot/joint_states', 10)

        self._rail_pub = self.create_publisher(
            Int32, '/motor/rail', 10)

        self._turntable_pub = self.create_publisher(
            Int32, '/motor/turntable', 10)

        self._status_pub = self.create_publisher(
            String, '/robot/status', 10)

        self._act_done_pub = self.create_publisher(
            Bool, '/robot/act_done', 10)

        self._grasp_done_pub = self.create_publisher(
            Bool, '/robot/grasp_done', 10)

        self._rail_done_pub = self.create_publisher(
            Bool, '/robot/rail_done', 10)

        # ── Services ──
        self._act_grasp_srv = self.create_service(
            Trigger, '/robot/act_grasp', self._act_grasp_service,
            callback_group=self._cb_group)

        self._go_home_srv = self.create_service(
            Trigger, '/robot/go_home', self._go_home_service,
            callback_group=self._cb_group)

        self._open_gripper_srv = self.create_service(
            Trigger, '/robot/open_gripper', self._open_gripper_service,
            callback_group=self._cb_group)

        self._close_gripper_srv = self.create_service(
            Trigger, '/robot/close_gripper', self._close_gripper_service,
            callback_group=self._cb_group)

    # ─────────────────────────────────────────────
    # 카메라 콜백
    # ─────────────────────────────────────────────
    def _handcam_callback(self, msg: CompressedImage):
        import cv2
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._handcam_lock:
                self._latest_handcam = frame

    def _handcam_callback_raw(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is not None:
            with self._handcam_lock:
                self._latest_handcam = frame

    def _esp32_rail_done_callback(self, msg: Bool):
        if msg.data:
            self._esp32_rail_done = True

    # ─────────────────────────────────────────────
    # 명령 콜백 (토픽 기반 — 비동기)
    # ─────────────────────────────────────────────
    def _grasp_cmd_callback(self, msg: GraspGoal):
        """파지 명령 수신 → ACT 파지 실행 (별도 스레드).

        주의: ACT(visuomotor)는 핸드캠 이미지+관절상태로 end-to-end 추론하므로
        목표 좌표(target_x/y)를 직접 사용하지 않는다. 좌표는 베드 위 어떤
        출력물을 대상으로 하는지에 대한 참고/로깅 용도이며, 향후 좌표 기반
        프리포지셔닝(레일/베이스 이동)을 붙일 때 사용할 수 있다.
        """
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('파지 명령 무시: 현재 동작 중')
            return
        self.get_logger().info(
            f'파지 목표 수신(참고): idx={msg.object_index} '
            f'x={msg.target_x:.1f} y={msg.target_y:.1f} '
            f'(ACT visuomotor 추론 사용, 좌표는 직접 미사용)')
        t = threading.Thread(
            target=self._execute_act_grasp, daemon=True)
        t.start()

    def _rail_cmd_callback(self, msg: Int32):
        """레일 이동 명령 수신 → 레일 이동 실행 (별도 스레드)."""
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('레일 명령 무시: 현재 동작 중')
            return
        pos_code = msg.data
        try:
            pos = RailPosition(pos_code)
        except ValueError:
            self.get_logger().error(f'알 수 없는 레일 위치 코드: {pos_code}')
            return
        t = threading.Thread(
            target=self._execute_rail_move, args=(pos,), daemon=True)
        t.start()

    def _rotate_cmd_callback(self, msg: Bool):
        """베이스 회전 명령 수신. true → 뒤 방향 / false → 앞 방향"""
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('회전 명령 무시: 현재 동작 중')
            return
        pose = POSE_BACK if msg.data else POSE_FRONT
        t = threading.Thread(
            target=self._execute_pose, args=(pose, '베이스 회전'), daemon=True)
        t.start()

    def _release_cmd_callback(self, msg: Bool):
        """투하 명령 수신."""
        if not msg.data:
            return
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('투하 명령 무시: 현재 동작 중')
            return
        t = threading.Thread(
            target=self._execute_release, daemon=True)
        t.start()

    def _home_cmd_callback(self, msg: Bool):
        """홈 복귀 명령 수신."""
        if not msg.data:
            return
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('홈 복귀 명령 무시: 현재 동작 중')
            return
        t = threading.Thread(
            target=self._execute_home, daemon=True)
        t.start()

    def _estop_cmd_callback(self, msg: Bool):
        """비상 정지 수신."""
        if msg.data:
            self.get_logger().error('비상 정지 명령 수신! 동작 강제 중단 및 에러 상태 천이')
            self._set_state(RobotState.ERROR)
            self._publish_status('ERROR: ESTOP ACTIVE')

    # ─────────────────────────────────────────────
    # 서비스 핸들러 (동기 — Main Orchestrator가 완료 대기)
    # ─────────────────────────────────────────────
    def _act_grasp_service(self, request, response):
        """ACT 파지 서비스 (완료까지 블로킹)."""
        success = self._execute_act_grasp()
        response.success = success
        response.message = 'ACT 파지 완료' if success else 'ACT 파지 실패'
        return response

    def _go_home_service(self, request, response):
        """홈 복귀 서비스."""
        success = self._execute_home()
        response.success = success
        response.message = '홈 복귀 완료' if success else '홈 복귀 실패'
        return response

    def _open_gripper_service(self, request, response):
        """그리퍼 열기 서비스 (lerobot bus 사용)."""
        self._write_raw_position({'gripper': GRIPPER_OPEN})
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 열기 완료 (OmxFollower ID16)'
        return response

    def _close_gripper_service(self, request, response):
        """그리퍼 닫기 서비스 (lerobot bus 사용)."""
        self._write_raw_position({'gripper': GRIPPER_CLOSE})
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 닫기 완료 (OmxFollower ID16)'
        return response

    # ─────────────────────────────────────────────
    # 실행 함수 — ACT 파지
    # ─────────────────────────────────────────────
    def _execute_act_grasp(self) -> bool:
        """
        ACT 모방학습 파지 실행.

        흐름:
          1. 핸드캠 이미지 + 관절 상태 → obs 딕셔너리 구성
          2. ACTPolicy.select_action(obs) → 액션 청크 (chunk_size × 6)
          3. 액션 청크를 30 Hz로 OmxFollower에 전송
          4. 완료 신호 발행
        """
        self._set_state(RobotState.ACT_GRASPING)
        self._publish_status('ACT 파지 시작')
        self.get_logger().info('ACT 파지 시작')

        if not self._act_ready:
            self.get_logger().error('ACT 모델 미로드 — 파지 불가')
            self._set_state(RobotState.IDLE)
            return False

        try:
            import torch
            start = time.time()

            # ── obs 구성 ──
            with self._handcam_lock:
                frame = self._latest_handcam

            if frame is None:
                self.get_logger().error('핸드캠 이미지 없음 — 파지 불가')
                self._set_state(RobotState.IDLE)
                return False

            import cv2
            frame_rgb = cv2.cvtColor(
                cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
            # (H, W, C) → (1, C, H, W), float32 [0,1]
            img_tensor = torch.from_numpy(
                frame_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self._act_device_obj)

            # 관절 상태: lerobot bus에서 raw 위치 읽기 → 라디안 변환 (0~4095 = 0~2π)
            raw_positions = self._read_raw_positions()
            joint_rad = [
                (raw_positions[name] / 4095.0) * 2 * math.pi
                for name in JOINT_NAMES
            ]
            state_tensor = torch.tensor(
                joint_rad, dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.to(self._act_device_obj)

            # TODO: 학습 데이터셋의 video_keys 설정에 따라 이미지 키 이름을 맞춰야 합니다.
            # 예: 'observation.images.handcam' 또는 'observation.images.top', 'observation.images.wrist' 등
            obs = {
                'observation.images.handcam': img_tensor,
                'observation.state': state_tensor,
            }

            # ── ACT 추론 ──
            with torch.no_grad():
                action_chunk = self._act_policy.select_action(obs)
                # action_chunk: Tensor shape (1, chunk_size, 6) or (chunk_size, 6)
                if action_chunk.ndim == 3:
                    action_chunk = action_chunk.squeeze(0)  # (chunk_size, 6)
                action_chunk = action_chunk.cpu().numpy()

            self.get_logger().info(
                f'ACT 추론 완료 | 청크 크기={len(action_chunk)} | '
                f'추론 시간={(time.time()-start)*1000:.1f}ms')

            # ── 액션 청크 실행 ──
            dt = 1.0 / ACT_CONTROL_HZ
            for i, action in enumerate(action_chunk):
                step_start = time.time()

                if self._use_real_hardware and self._dxl_ready:
                    # lerobot 공식 API를 통한 캘리브레이션/변환 적용 및 모터 전송
                    action_dict = {f"{name}.pos": float(action[j]) for j, name in enumerate(JOINT_NAMES)}
                    with self._dxl_io_lock:
                        self._follower.send_action(action_dict)
                else:
                    # 시뮬레이션 모드 또는 하드웨어가 없는 경우
                    goal_dict = {}
                    for j, name in enumerate(JOINT_NAMES):
                        val = float(action[j])
                        # LeRobot의 M100_100 범위 [-100, 100]를 0~4095로 변환 모사
                        if -100 <= val <= 100:
                            raw = int(((val + 100.0) / 200.0) * 4095.0)
                        else:
                            raw = int(np.clip((val / (2 * math.pi)) * 4095, 0, 4095))
                        goal_dict[name] = raw
                    self._write_raw_position(goal_dict)

                # 30 Hz 타이밍 유지
                elapsed = time.time() - step_start
                remaining = dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            total_time = time.time() - start
            self.get_logger().info(f'ACT 파지 완료 | 총 소요={total_time:.2f}s')

            # ── 완료 신호 발행 ──
            done_msg = Bool()
            done_msg.data = True
            self._act_done_pub.publish(done_msg)
            self._grasp_done_pub.publish(done_msg)

            self._set_state(RobotState.IDLE)
            self._publish_status('ACT 파지 완료')
            return True

        except Exception as e:
            self.get_logger().error(f'ACT 파지 중 오류: {e}')
            self._set_state(RobotState.ERROR)
            self._publish_status(f'ERROR: {e}')
            return False

    # ─────────────────────────────────────────────
    # 실행 함수 — 레일 이동
    # ─────────────────────────────────────────────
    def _execute_rail_move(self, position: RailPosition) -> bool:
        """
        레일을 지정 위치로 이동.
        /motor/rail 토픽으로 스텝 수 발행 → ESP32-S3가 TB6600 구동.
        """
        self._set_state(RobotState.MOVING_RAIL)
        target_mm = float(self._rail_mm[position])
        target_steps = int(target_mm * 80.0)
        pos_name = position.name
        self._publish_status(f'레일 이동: {pos_name} ({target_mm:.1f}mm -> {target_steps}스텝)')
        self.get_logger().info(f'레일 이동 명령: {pos_name} = {target_mm:.1f}mm -> {target_steps}스텝')

        if self._use_real_hardware:
            self._esp32_rail_done = False

        msg = Int32()
        msg.data = target_steps
        self._rail_pub.publish(msg)

        if self._use_real_hardware:
            # ESP32로부터 완료 신호 대기 (타임아웃 포함)
            deadline = time.time() + self._rail_timeout
            success = False
            while time.time() < deadline:
                if self._esp32_rail_done:
                    self._esp32_rail_done = False
                    success = True
                    break
                time.sleep(0.05)
            
            if not success:
                self.get_logger().error(f'레일 이동 타임아웃! ({self._rail_timeout}초)')
                self._set_state(RobotState.ERROR)
                self._publish_status('ERROR: 레일 이동 타임아웃')
                return False
        else:
            # 시뮬레이션 모드에서는 즉시 완료 처리 (대기 시간 모사)
            time.sleep(1.0)

        done_msg = Bool()
        done_msg.data = True
        self._rail_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status(f'레일 이동 완료: {pos_name}')
        return True

    # ─────────────────────────────────────────────
    # 실행 함수 — 자세 이동
    # ─────────────────────────────────────────────
    def _execute_pose(self, target_pose: dict, label: str = '') -> bool:
        """목표 자세(raw Dynamixel 위치값 dict)로 이동."""
        self._set_state(RobotState.ROTATING_BASE)
        self._publish_status(f'자세 변경: {label}')
        self.get_logger().info(f'자세 변경: {label} → {target_pose}')

        if target_pose == POSE_BACK and POSE_BACK == POSE_FRONT:
            self.get_logger().warn('경고: POSE_BACK과 POSE_FRONT의 모터 제어값이 동일합니다. 실제 회전을 위해 캘리브레이션이 필요합니다.')

        success = self._write_raw_position(target_pose)
        time.sleep(1.5)  # 자세 안정화 대기

        self._set_state(RobotState.IDLE)
        self._publish_status(f'자세 변경 완료: {label}')
        return success

    # ─────────────────────────────────────────────
    # 실행 함수 — 투하
    # ─────────────────────────────────────────────
    def _execute_release(self) -> bool:
        """분류함 위에서 그리퍼를 열어 출력물 투하."""
        self._set_state(RobotState.RELEASING)
        self._publish_status('출력물 투하')
        self.get_logger().info('출력물 투하: 그리퍼 열기 (OmxFollower gripper)')

        self._write_raw_position({'gripper': GRIPPER_OPEN})
        time.sleep(0.8)

        done_msg = Bool()
        done_msg.data = True
        self._grasp_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status('투하 완료')
        return True

    # ─────────────────────────────────────────────
    # 실행 함수 — 홈 복귀
    # ─────────────────────────────────────────────
    def _execute_home(self) -> bool:
        """전체 관절을 홈 자세로 복귀."""
        self._set_state(RobotState.HOMING)
        self._publish_status('홈 복귀')
        self.get_logger().info('홈 복귀 시작')

        success = self._write_raw_position(POSE_HOME)
        time.sleep(2.0)

        # 홈 복귀 완료 신호 발행
        done_msg = Bool()
        done_msg.data = True
        self._grasp_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status('홈 복귀 완료')
        return success

    # ─────────────────────────────────────────────
    # lerobot bus 기반 모터 I/O
    # ─────────────────────────────────────────────
    def _write_raw_position(self, positions: dict) -> bool:
        """lerobot bus를 통해 raw 위치값 전송 (normalize=False).

        Args:
            positions: {'shoulder_pan': 2048, 'gripper': 2300, ...} 형태의 dict.
                       전체 6관절 또는 일부 관절만 지정 가능.
        """
        if not self._use_real_hardware or not self._dxl_ready:
            self.get_logger().debug(f'[SIM] 관절 목표: {positions}')
            return True

        try:
            with self._dxl_io_lock:
                self._follower.bus.sync_write(
                    'Goal_Position', positions, normalize=False)
            return True
        except Exception as e:
            self.get_logger().error(f'lerobot sync_write 오류: {e}')
            return False

    def _read_raw_positions(self) -> dict:
        """lerobot bus를 통해 현재 raw 위치값 읽기 (normalize=False).

        Returns:
            {'shoulder_pan': 2048, ...} 형태의 dict.
        """
        if not self._use_real_hardware or not self._dxl_ready:
            return {name: 2048 for name in JOINT_NAMES}

        try:
            with self._dxl_io_lock:
                return self._follower.bus.sync_read(
                    'Present_Position', normalize=False)
        except Exception as e:
            self.get_logger().error(f'lerobot sync_read 오류: {e}')
            return {name: 2048 for name in JOINT_NAMES}

    # ─────────────────────────────────────────────
    # 관절 상태 발행 (30 Hz 타이머)
    # ─────────────────────────────────────────────
    def _publish_joint_states(self):
        """현재 관절 위치를 JointState 토픽으로 발행."""
        raw_positions = self._read_raw_positions()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = JOINT_NAMES
        # raw Dynamixel 위치값 → 라디안 변환 (0~4095 = 0~2π)
        msg.position = [
            (raw_positions[name] / 4095.0) * 2 * math.pi
            for name in JOINT_NAMES
        ]
        self._joint_state_pub.publish(msg)

    # ─────────────────────────────────────────────
    # 상태 관리 유틸리티
    # ─────────────────────────────────────────────
    def _get_state(self) -> RobotState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: RobotState):
        with self._state_lock:
            self._state = state

    def _publish_status(self, msg: str):
        status_msg = String()
        status_msg.data = f'[ROBOT] {msg}'
        self._status_pub.publish(status_msg)

    # ─────────────────────────────────────────────
    # 종료 처리
    # ─────────────────────────────────────────────
    def destroy_node(self):
        """노드 종료 시 lerobot을 통해 안전하게 연결 해제."""
        # 텔레옵이 돌고 있다면 정지
        if self._teleop_running:
            self._stop_teleop()

        if self._follower and self._dxl_ready:
            self.get_logger().info('OmxFollower 연결 해제 (토크 비활성화 포함)')
            try:
                self._follower.disconnect()
            except Exception as e:
                self.get_logger().warn(f'OmxFollower 해제 중 오류: {e}')
        super().destroy_node()

    # ─────────────────────────────────────────────
    # 텔레오퍼레이션 제어 (lerobot OmxLeader 사용)
    # ─────────────────────────────────────────────
    def _teleop_cmd_callback(self, msg: Bool):
        """텔레오퍼레이션 ON/OFF 명령 수신."""
        if msg.data:
            t = threading.Thread(target=self._start_teleop, daemon=True)
            t.start()
        else:
            t = threading.Thread(target=self._stop_teleop, daemon=True)
            t.start()

    def _start_teleop(self) -> bool:
        """lerobot OmxLeader를 사용한 리더-팔로워 텔레오퍼레이션 시작."""
        if self._get_state() == RobotState.TELEOPING:
            return True
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('텔레옵 무시: 현재 로봇이 IDLE 상태가 아님')
            return False

        self._set_state(RobotState.TELEOPING)
        self._publish_status('텔레오퍼레이션 활성화')
        self.get_logger().info('텔레오퍼레이션 시작 중...')

        if self._use_real_hardware:
            try:
                leader_config = OmxLeaderConfig(
                    port=self._leader_port_name,
                    id='quvi_leader',
                )
                self._leader = OmxLeader(leader_config)
                self._leader.connect()
                self.get_logger().info(
                    f'OmxLeader 연결 완료 | 포트={self._leader_port_name} | '
                    f'모터: {list(self._leader.bus.motors.keys())}')
            except Exception as e:
                self.get_logger().error(f'OmxLeader 초기화 실패: {e}')
                self._leader = None
                self._set_state(RobotState.IDLE)
                self._publish_status(f'텔레옵 에러: {e}')
                return False

        self._teleop_running = True
        self._teleop_thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self._teleop_thread.start()
        self.get_logger().info('텔레오퍼레이션 루프 시작됨')
        return True

    def _stop_teleop(self) -> bool:
        """리더-팔로워 텔레오퍼레이션 종료."""
        if not self._teleop_running:
            return True

        self.get_logger().info('텔레오퍼레이션 종료 중...')
        self._teleop_running = False
        if hasattr(self, '_teleop_thread') and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=2.0)

        if self._leader:
            try:
                self._leader.disconnect()
                self.get_logger().info('OmxLeader 연결 해제 완료')
            except Exception as e:
                self.get_logger().error(f'OmxLeader 해제 중 오류: {e}')
            self._leader = None

        self._set_state(RobotState.IDLE)
        self._publish_status('텔레오퍼레이션 종료')
        return True

    def _teleop_loop(self):
        """lerobot OmxLeader → OmxFollower 텔레오퍼레이션 루프 (100Hz).

        공식 API 사용:
          - leader.get_action()  → 리더 관절 위치 (정규화된 값)
          - follower.send_action(action) → 팔로워에 전송 (정규화 → raw 변환 자동)
        """
        dt = 1.0 / 100.0
        sim_angle = 0.0

        while self._teleop_running and rclpy.ok():
            start_time = time.time()

            if self._use_real_hardware and self._leader and self._follower:
                try:
                    # lerobot 공식 API: 정규화된 값으로 리더→팔로워 직접 매핑
                    with self._dxl_io_lock:
                        action = self._leader.get_action()
                        self._follower.send_action(action)
                except Exception as e:
                    self.get_logger().warn(f'텔레옵 루프 오류: {e}')
            else:
                # 시뮬레이션 모드: 부드러운 사인파 형태의 모의 각도 전송
                sim_angle += 0.05
                sim_pose = {
                    'shoulder_pan': 2048 + int(500 * math.sin(sim_angle)),
                    'shoulder_lift': 1800 + int(300 * math.cos(sim_angle)),
                    'elbow_flex': 1200 + int(200 * math.sin(sim_angle * 1.5)),
                    'wrist_flex': 2048,
                    'wrist_roll': 2048,
                    'gripper': GRIPPER_OPEN if math.sin(sim_angle * 0.5) > 0 else GRIPPER_CLOSE,
                }
                self._write_raw_position(sim_pose)

            # 30Hz 제어 속도 타이밍 대기
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    # 블로킹 서비스(ACT 파지/홈 복귀)와 30Hz 타이머, 텔레옵 명령이
    # 서로를 막지 않도록 MultiThreadedExecutor 사용.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
