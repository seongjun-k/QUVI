"""
QUVI ROBOT_CONTROL_NODE
────────────────────────────────────────────────────────────────
로봇팔(OMX AI Manipulator) + 리니어 레일 + 턴테이블을
통합 제어하는 노드.

모터 구성:
  ID 1~3 : DYNAMIXEL XL430-W250T  (동작 전압 12V)  — 베이스, 숄더, 엘보우
  ID 4~5 : DYNAMIXEL XL330-M288T  (동작 전압  5V)  — 리스트, 그리퍼

주요 기능:
  1. ACT 모방학습 파지 (LeRobot ACTPolicy)
     - /camera/handcam 이미지 + 관절 상태 → ACT 추론 → 관절 목표값 전송
  2. OMX Dynamixel 관절 제어
     - dynamixel_sdk 직접 제어 (Position Mode)
     - 홈 복귀, 180° 베이스 회전, 안착/투하 자세
  3. 레일 이동 명령 → ESP32-S3 (/motor/rail)
  4. 턴테이블 회전 명령 → ESP32-S3 (/motor/turntable)

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
import threading
import time
from enum import IntEnum
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


# ─────────────────────────────────────────────────────────────
# Dynamixel 공통 레지스터 주소 (Protocol 2.0)
# ─────────────────────────────────────────────────────────────
DXL_BAUDRATE          = 1_000_000
DXL_PROTOCOL          = 2.0
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE   = 11
LEN_GOAL_POSITION     = 4
LEN_PRESENT_POSITION  = 4
TORQUE_ENABLE         = 1
TORQUE_DISABLE        = 0
POSITION_MODE         = 3

# ─────────────────────────────────────────────────────────────
# 관절 ID / 모터 종류 분류
# ─────────────────────────────────────────────────────────────
# 관절 및 ID 매핑 (리더: 1~6, 팔로워: 11~16)
LEADER_IDS_XL430  = [1, 2, 3]        # 리더 12V 모터 (베이스/숄더/엘보우)
LEADER_IDS_XL330  = [4, 5, 6]        # 리더 5V 모터 (리스트1/리스트2/그리퍼)
LEADER_IDS        = LEADER_IDS_XL430 + LEADER_IDS_XL330  # [1, 2, 3, 4, 5, 6]

FOLLOWER_IDS_XL430 = [11, 12, 13]    # 팔로워 12V 모터 (베이스/숄더/엘보우)
FOLLOWER_IDS_XL330 = [14, 15, 16]    # 팔로워 5V 모터 (리스트1/리스트2/그리퍼)
FOLLOWER_IDS       = FOLLOWER_IDS_XL430 + FOLLOWER_IDS_XL330  # [11, 12, 13, 14, 15, 16]

JOINT_NAMES       = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']

# ─────────────────────────────────────────────────────────────
# 주요 자세 (Dynamixel 위치값 0~4095 = 0~360°)
# ─────────────────────────────────────────────────────────────
#                         [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
POSE_HOME  = [2048, 1800, 1200, 2048, 2048, 2048]   # 홈 (직립)
POSE_FRONT = [2048, 1400,  900, 1800, 2048, 2300]   # 베드 파지 준비 (앞 방향)
POSE_BACK  = [2048, 1400,  900, 1800, 2048, 2300]   # 검사/분류 (180° 회전)
POSE_PLACE = [2048, 1600, 1100, 2048, 2048, 2300]   # 턴테이블 안착

# 그리퍼 (ID 6 / ID 16, XL330-M288T)
GRIPPER_OPEN  = 2300   # 열림
GRIPPER_CLOSE = 1800   # 닫힘

# ACT 실행 주기 (Hz)
ACT_CONTROL_HZ = 30


# ─────────────────────────────────────────────────────────────
# RobotControlNode
# ─────────────────────────────────────────────────────────────

class RobotControlNode(Node):
    """로봇팔 + 레일 + 턴테이블 통합 제어 노드."""

    def __init__(self):
        super().__init__('robot_control_node')

        # ─── 파라미터 선언 ───
        self._declare_params()
        self._load_params()

        # ─── 내부 상태 ───
        self._state: RobotState = RobotState.IDLE
        self._state_lock = threading.Lock()

        self._latest_handcam: Optional[np.ndarray] = None
        self._latest_joint_pos: List[int] = [2048] * 6   # Dynamixel 위치값 (6자유도)
        self._handcam_lock = threading.Lock()
        self._bridge = CvBridge()

        # ─── 텔레오퍼레이션 상태 ───
        self._teleop_running = False
        self._leader_port_handler = None
        self._leader_packet_handler = None

        # ─── 콜백 그룹 (블로킹 서비스/텔레옵과 타이머가 서로를 막지 않도록) ───
        self._cb_group = ReentrantCallbackGroup()

        # ─── Dynamixel 포트 접근 직렬화 락 ───
        # MultiThreadedExecutor 환경에서 30Hz joint 발행 타이머(read)와
        # ACT/텔레옵 루프(write)가 동일 시리얼 포트에 동시 접근하면
        # 패킷이 섞일 수 있으므로 모든 저수준 I/O 를 이 락으로 직렬화한다.
        self._dxl_io_lock = threading.Lock()

        # ─── Dynamixel 초기화 ───
        self._dxl_port = None
        self._packet_handler = None
        self._port_handler = None
        self._dxl_ready = False
        if self._use_real_hardware:
            self._init_dynamixel()

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
            f'ROBOT_CONTROL_NODE 초기화 완료 | '
            f'하드웨어={self._use_real_hardware} | '
            f'ACT={self._use_act} | '
            f'DXL포트={self._dxl_port_name} | '
            f'XL430(12V)=ID{FOLLOWER_IDS_XL430} | '
            f'XL330(5V)=ID{FOLLOWER_IDS_XL330}')

    # ─────────────────────────────────────────────
    # 파라미터
    # ─────────────────────────────────────────────
    def _declare_params(self):
        # 하드웨어
        self.declare_parameter('use_real_hardware', True)
        self.declare_parameter('dxl_port', '/dev/ttyFollower')
        self.declare_parameter('leader_dxl_port', '/dev/ttyLeader')
        self.declare_parameter('dxl_baudrate', DXL_BAUDRATE)
        # ACT
        self.declare_parameter('use_act', True)
        self.declare_parameter('act_model_path',
            'outputs/train/quvi_act/checkpoints/last/pretrained_model')
        self.declare_parameter('act_chunk_size', 20)
        self.declare_parameter('act_device', 'cpu')   # 'cuda' or 'cpu'
        # 레일 위치 (스텝 수) — 조립 후 캘리브레이션으로 확정
        self.declare_parameter('rail_steps_bed',      0)
        self.declare_parameter('rail_steps_inspect', 1000)
        self.declare_parameter('rail_steps_pass',    1700)
        self.declare_parameter('rail_steps_fail',    2400)
        # 카메라 (usb_cam camera1 네임스페이스와 일치하도록 기본값 설정)
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
        self._dxl_baudrate      = self.get_parameter('dxl_baudrate').value
        self._use_act           = self.get_parameter('use_act').value
        self._act_model_path    = self.get_parameter('act_model_path').value
        self._act_chunk_size    = self.get_parameter('act_chunk_size').value
        self._act_device        = self.get_parameter('act_device').value
        self._rail_steps = {
            RailPosition.BED:     self.get_parameter('rail_steps_bed').value,
            RailPosition.INSPECT: self.get_parameter('rail_steps_inspect').value,
            RailPosition.PASS:    self.get_parameter('rail_steps_pass').value,
            RailPosition.FAIL:    self.get_parameter('rail_steps_fail').value,
        }
        self._handcam_topic  = self.get_parameter('handcam_topic').value
        self._use_compressed = self.get_parameter('use_compressed').value
        self._rail_timeout   = self.get_parameter('rail_move_timeout_sec').value
        self._grasp_timeout  = self.get_parameter('grasp_timeout_sec').value
        self._home_timeout   = self.get_parameter('home_timeout_sec').value

    # ─────────────────────────────────────────────
    # Dynamixel 초기화
    # ─────────────────────────────────────────────
    def _init_dynamixel(self):
        """
        dynamixel_sdk으로 포트 열고 모터 초기화.

        XL430-W250T (ID 1~3, 12V) : 고토크 관절 (베이스/숄더/엘보우)
        XL330-M288T (ID 4~5,  5V) : 경량 관절 (리스트/그리퍼)

        ※ 두 모터 모두 Protocol 2.0 / Position Mode 사용.
          위치값 범위(0~4095)와 레지스터 주소가 동일하므로
          동일한 초기화 루틴 적용 가능.
        """
        try:
            from dynamixel_sdk import (
                PortHandler, PacketHandler, GroupSyncWrite,
                COMM_SUCCESS
            )
        except ImportError:
            self.get_logger().error(
                'dynamixel_sdk 미설치. pip install dynamixel-sdk --break-system-packages')
            return

        self._port_handler   = PortHandler(self._dxl_port_name)
        self._packet_handler = PacketHandler(DXL_PROTOCOL)

        if not self._port_handler.openPort():
            self.get_logger().error(f'Dynamixel 포트 열기 실패: {self._dxl_port_name}')
            return

        if not self._port_handler.setBaudRate(self._dxl_baudrate):
            self.get_logger().error('Dynamixel 보드레이트 설정 실패')
            return

        # 각 관절 Position Mode 설정 + 토크 활성화 (팔로워 ID 11~16)
        for dxl_id in FOLLOWER_IDS:
            motor_type = 'XL430-W250T(12V)' if dxl_id in FOLLOWER_IDS_XL430 \
                         else 'XL330-M288T(5V)'

            # Operating Mode = Position (3)
            self._packet_handler.write1ByteTxRx(
                self._port_handler, dxl_id, ADDR_OPERATING_MODE, POSITION_MODE)

            # Torque Enable
            result, error = self._packet_handler.write1ByteTxRx(
                self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

            if result != 0:
                self.get_logger().warn(
                    f'ID{dxl_id}({motor_type}) 토크 활성화 실패 (result={result})')
            else:
                self.get_logger().info(
                    f'ID{dxl_id}({motor_type}) 토크 활성화 완료')

        # GroupSyncWrite (4바이트 Goal Position) — XL430/XL330 공통 주소
        from dynamixel_sdk import GroupSyncWrite
        self._sync_write = GroupSyncWrite(
            self._port_handler, self._packet_handler,
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        self._dxl_ready = True
        self.get_logger().info(
            'Dynamixel 초기화 완료 | '
            f'XL430-W250T(12V): ID{FOLLOWER_IDS_XL430} | '
            f'XL330-M288T(5V): ID{FOLLOWER_IDS_XL330}')

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

        self.get_logger().info(f'ACT 모델 로드 중: {self._act_model_path}')
        try:
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
            self._act_policy = ACTPolicy.from_pretrained(self._act_model_path)
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
        import cv2
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is not None:
            with self._handcam_lock:
                self._latest_handcam = frame

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
        """베이스 회전 명령 수신.
        true  → 뒤 방향 (검사/분류)
        false → 앞 방향 (베드 파지)
        """
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
        t = threading.Thread(
            target=self._execute_release, daemon=True)
        t.start()

    def _home_cmd_callback(self, msg: Bool):
        """홈 복귀 명령 수신."""
        if not msg.data:
            return
        t = threading.Thread(
            target=self._execute_home, daemon=True)
        t.start()

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
        """그리퍼 열기 서비스 (ID 16, XL330-M288T, 5V)."""
        self._set_joint_position(16, GRIPPER_OPEN)
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 열기 완료 (XL330 ID16)'
        return response

    def _close_gripper_service(self, request, response):
        """그리퍼 닫기 서비스 (ID 16, XL330-M288T, 5V)."""
        self._set_joint_position(16, GRIPPER_CLOSE)
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 닫기 완료 (XL330 ID16)'
        return response

    # ─────────────────────────────────────────────
    # 실행 함수 — ACT 파지
    # ─────────────────────────────────────────────
    def _execute_act_grasp(self) -> bool:
        """
        ACT 모방학습 파지 실행.

        흐름:
          1. 핸드캠 이미지 + 관절 상태 → obs 딕셔너리 구성
          2. ACTPolicy.select_action(obs) → 액션 청크 (chunk_size × 5)
          3. 액션 청크를 30 Hz로 Dynamixel에 순서대로 전송
             (XL430 ID1~3: 12V 고토크 / XL330 ID4~5: 5V 경량)
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

            # 관절 상태: Dynamixel 위치값 → 라디안 (0~4095 = 0~2π)
            joint_rad = [
                (pos / 4095.0) * 2 * math.pi
                for pos in self._latest_joint_pos
            ]
            state_tensor = torch.tensor(
                joint_rad, dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.to(self._act_device_obj)

            obs = {
                'observation.images.handcam': img_tensor,
                'observation.state': state_tensor,
            }

            # ── ACT 추론 ──
            with torch.no_grad():
                action_chunk = self._act_policy.select_action(obs)
                # action_chunk: Tensor shape (1, chunk_size, 5) or (chunk_size, 5)
                if action_chunk.ndim == 3:
                    action_chunk = action_chunk.squeeze(0)  # (chunk_size, 5)
                action_chunk = action_chunk.cpu().numpy()

            self.get_logger().info(
                f'ACT 추론 완료 | 청크 크기={len(action_chunk)} | '
                f'추론 시간={(time.time()-start)*1000:.1f}ms')

            # ── 액션 청크 실행 ──
            dt = 1.0 / ACT_CONTROL_HZ
            for i, action in enumerate(action_chunk):
                step_start = time.time()

                # 라디안 → Dynamixel 위치값 (0~4095)
                dxl_goals = [
                    int(np.clip(
                        (float(a) / (2 * math.pi)) * 4095,
                        0, 4095))
                    for a in action
                ]

                self._sync_send_positions(dxl_goals)

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
        target_steps = self._rail_steps[position]
        pos_name = position.name
        self._publish_status(f'레일 이동: {pos_name} ({target_steps}스텝)')
        self.get_logger().info(f'레일 이동 명령: {pos_name} = {target_steps}스텝')

        msg = Int32()
        msg.data = target_steps
        self._rail_pub.publish(msg)

        time.sleep(0.2)

        done_msg = Bool()
        done_msg.data = True
        self._rail_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status(f'레일 이동 완료: {pos_name}')
        return True

    # ─────────────────────────────────────────────
    # 실행 함수 — 자세 이동
    # ─────────────────────────────────────────────
    def _execute_pose(self, target_pose: List[int], label: str = '') -> bool:
        """목표 자세(Dynamixel 위치값 5개)로 이동."""
        self._set_state(RobotState.ROTATING_BASE)
        self._publish_status(f'자세 변경: {label}')
        self.get_logger().info(f'자세 변경: {label} → {target_pose}')

        success = self._sync_send_positions(target_pose)
        time.sleep(1.5)  # 자세 안정화 대기

        self._set_state(RobotState.IDLE)
        self._publish_status(f'자세 변경 완료: {label}')
        return success

    # ─────────────────────────────────────────────
    # 실행 함수 — 투하
    # ─────────────────────────────────────────────
    def _execute_release(self) -> bool:
        """분류함 위에서 그리퍼를 열어 출력물 투하 (ID5 XL330-M288T)."""
        self._set_state(RobotState.RELEASING)
        self._publish_status('출력물 투하')
        self.get_logger().info('출력물 투하: 그리퍼 열기 (XL330 ID5)')

        self._set_joint_position(16, GRIPPER_OPEN)
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

        success = self._sync_send_positions(POSE_HOME)
        time.sleep(2.0)

        self._set_state(RobotState.IDLE)
        self._publish_status('홈 복귀 완료')
        return success

    # ─────────────────────────────────────────────
    # Dynamixel 저수준 제어
    # ─────────────────────────────────────────────
    def _sync_send_positions(self, positions: List[int]) -> bool:
        """
        GroupSyncWrite로 6개 관절 위치 동시 전송.
        리더(1~6)와 매칭되는 팔로워(11~16)에 Goal Position 적용.
        """
        if not self._use_real_hardware or not self._dxl_ready:
            self.get_logger().debug(f'[SIM] 관절 목표: {positions}')
            self._latest_joint_pos = list(positions)
            return True

        try:
            with self._dxl_io_lock:
                self._sync_write.clearParam()
                for dxl_id, goal in zip(FOLLOWER_IDS, positions):
                    goal = int(np.clip(goal, 0, 4095))
                    data = [
                        (goal >> 0)  & 0xFF,
                        (goal >> 8)  & 0xFF,
                        (goal >> 16) & 0xFF,
                        (goal >> 24) & 0xFF,
                    ]
                    self._sync_write.addParam(dxl_id, data)

                result = self._sync_write.txPacket()
            if result != 0:
                self.get_logger().warn(f'SyncWrite 실패 (result={result})')
                return False

            self._latest_joint_pos = list(positions)
            return True

        except Exception as e:
            self.get_logger().error(f'Dynamixel SyncWrite 오류: {e}')
            return False

    def _set_joint_position(self, dxl_id: int, position: int) -> bool:
        """단일 관절 위치 전송."""
        if not self._use_real_hardware or not self._dxl_ready:
            idx = FOLLOWER_IDS.index(dxl_id) if dxl_id in FOLLOWER_IDS else -1
            if idx != -1:
                self._latest_joint_pos[idx] = position
            return True

        try:
            with self._dxl_io_lock:
                result, _ = self._packet_handler.write4ByteTxRx(
                    self._port_handler, dxl_id, ADDR_GOAL_POSITION, position)
            idx = FOLLOWER_IDS.index(dxl_id) if dxl_id in FOLLOWER_IDS else -1
            if idx != -1:
                self._latest_joint_pos[idx] = position
            return result == 0
        except Exception as e:
            self.get_logger().error(f'단일 관절 전송 오류 (ID{dxl_id}): {e}')
            return False

    def _read_joint_positions(self) -> List[int]:
        """현재 관절 위치 읽기."""
        if not self._use_real_hardware or not self._dxl_ready:
            return list(self._latest_joint_pos)

        positions = []
        for i, dxl_id in enumerate(FOLLOWER_IDS):
            try:
                with self._dxl_io_lock:
                    val, result, _ = self._packet_handler.read4ByteTxRx(
                        self._port_handler, dxl_id, ADDR_PRESENT_POSITION)
                positions.append(
                    int(val) if result == 0 else self._latest_joint_pos[i])
            except Exception:
                positions.append(self._latest_joint_pos[i])
        return positions

    # ─────────────────────────────────────────────
    # 관절 상태 발행 (30 Hz 타이머)
    # ─────────────────────────────────────────────
    def _publish_joint_states(self):
        """현재 관절 위치를 JointState 토픽으로 발행."""
        positions = self._read_joint_positions()
        self._latest_joint_pos = positions

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = JOINT_NAMES
        # Dynamixel 위치값 → 라디안 변환 (0~4095 = 0~2π)
        msg.position = [
            (pos / 4095.0) * 2 * math.pi for pos in positions
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
        """노드 종료 시 토크 비활성화."""
        # 텔레옵이 돌고 있다면 정지
        if self._teleop_running:
            self._stop_teleop()

        if self._use_real_hardware and self._dxl_ready:
            self.get_logger().info('Dynamixel 토크 비활성화')
            for dxl_id in JOINT_IDS:
                try:
                    self._packet_handler.write1ByteTxRx(
                        self._port_handler, dxl_id,
                        ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                except Exception:
                    pass
            self._port_handler.closePort()
        super().destroy_node()

    # ─────────────────────────────────────────────
    # 텔레오퍼레이션 제어 함수
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
        """리더-팔로워 텔레오퍼레이션 시작."""
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
                from dynamixel_sdk import PortHandler, PacketHandler
                self._leader_port_handler = PortHandler(self._leader_port_name)
                self._leader_packet_handler = PacketHandler(DXL_PROTOCOL)

                if not self._leader_port_handler.openPort():
                    self.get_logger().error(f'리더 암 다이나믹셀 포트 열기 실패: {self._leader_port_name}')
                    self._set_state(RobotState.IDLE)
                    self._publish_status('텔레옵 에러: 리더 포트 열기 실패')
                    return False

                if not self._leader_port_handler.setBaudRate(self._dxl_baudrate):
                    self.get_logger().error('리더 암 다이나믹셀 보드레이트 설정 실패')
                    self._leader_port_handler.closePort()
                    self._leader_port_handler = None
                    self._set_state(RobotState.IDLE)
                    self._publish_status('텔레옵 에러: 보드레이트 설정 실패')
                    return False

                # 리더 암 모터 토크 해제 (Torque Disable) -> 사람이 손으로 조작 가능하도록 함
                for dxl_id in LEADER_IDS:
                    self._leader_packet_handler.write1ByteTxRx(
                        self._leader_port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                    self.get_logger().info(f'리더 암 ID{dxl_id} 토크 해제 완료')

                # 팔로워 암 모터 토크 활성화 (다시 한 번 확인)
                if self._dxl_ready:
                    for dxl_id in FOLLOWER_IDS:
                        self._packet_handler.write1ByteTxRx(
                            self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

            except Exception as e:
                self.get_logger().error(f'리더 암 초기화 중 오류: {e}')
                if self._leader_port_handler:
                    try:
                        self._leader_port_handler.closePort()
                    except Exception:
                        pass
                self._leader_port_handler = None
                self._leader_packet_handler = None
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
            self._teleop_thread.join(timeout=1.0)

        if self._use_real_hardware and self._leader_port_handler:
            try:
                # 리더 암 모터 토크 다시 활성화하여 락(고정) 처리 -> 갑자기 아래로 툭 떨어지는 것을 방지
                if self._leader_packet_handler:
                    for dxl_id in LEADER_IDS:
                        self._leader_packet_handler.write1ByteTxRx(
                            self._leader_port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
                        self.get_logger().info(f'리더 암 ID{dxl_id} 토크 복구 완료')
                
                # 리더 암 포트 닫기
                self._leader_port_handler.closePort()
                self.get_logger().info('리더 암 포트 닫힘')
            except Exception as e:
                self.get_logger().error(f'리더 암 포트 닫기 중 오류: {e}')

        self._leader_port_handler = None
        self._leader_packet_handler = None

        self._set_state(RobotState.IDLE)
        self._publish_status('텔레오퍼레이션 종료')
        return True

    def _teleop_loop(self):
        """리더-팔로워 동기화 제어 주기 루프 (30Hz)."""
        dt = 1.0 / ACT_CONTROL_HZ
        sim_angle = 0.0

        while self._teleop_running and rclpy.ok():
            start_time = time.time()
            positions = []

            if self._use_real_hardware and self._leader_port_handler and self._leader_packet_handler:
                for dxl_id in LEADER_IDS:
                    try:
                        val, result, error = self._leader_packet_handler.read4ByteTxRx(
                            self._leader_port_handler, dxl_id, ADDR_PRESENT_POSITION)
                        if result == 0:
                            positions.append(int(val))
                        else:
                            idx = LEADER_IDS.index(dxl_id)
                            positions.append(self._latest_joint_pos[idx])
                    except Exception:
                        idx = LEADER_IDS.index(dxl_id)
                        positions.append(self._latest_joint_pos[idx])
            else:
                # 시뮬레이션 모드: 부드러운 사인파 형태의 모의 각도 전송
                sim_angle += 0.05
                base_sim = 2048 + int(500 * math.sin(sim_angle))
                shoulder_sim = 1800 + int(300 * math.cos(sim_angle))
                elbow_sim = 1200 + int(200 * math.sin(sim_angle * 1.5))
                wrist_p = 2048
                wrist_r = 2048
                gripper_sim = GRIPPER_OPEN if math.sin(sim_angle * 0.5) > 0 else GRIPPER_CLOSE
                positions = [base_sim, shoulder_sim, elbow_sim, wrist_p, wrist_r, gripper_sim]

            # 팔로워에 동시 전송 (6자유도 매핑)
            if len(positions) == 6:
                self._sync_send_positions(positions)

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
