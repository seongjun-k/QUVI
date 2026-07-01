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
     - /camera/sidecam 이미지 + 관절 상태 → ACT 추론 → 관절 목표값 전송
  2. OMX Dynamixel 관절 제어 (lerobot 공식 API)
     - 홈 복귀, 자세 이동, 그리퍼 제어
  3. 리더-팔로워 텔레오퍼레이션 (lerobot OmxLeader → OmxFollower)
  4. 레일 이동 명령 → ESP32-S3 (/motor/rail Float32 mm)
  5. 턴테이블 회전 명령 → ESP32-S3 (/motor/turntable)

ROS 2 인터페이스 (Subscriber):
  /camera/sidecam/compressed  sensor_msgs/CompressedImage   사이드캠 이미지
  /robot/grasp_command        quvi_msgs/GraspGoal           파지 트리거 + 목표 좌표
  /robot/rail_command         std_msgs/Int32                레일 목표 위치 코드 (0=D,1=A,2=B,3=C)
  /robot/rotate_command       std_msgs/Bool                 베이스 180° 회전 (true=뒤, false=앞)
  /robot/release_command      std_msgs/Bool                 출력물 투하
  /robot/home_command         std_msgs/Bool                 홈 복귀

ROS 2 인터페이스 (Publisher):
  /robot/joint_states         sensor_msgs/JointState        현재 관절 각도 (30 Hz)
  /motor/rail                 std_msgs/Float32              레일 목표 위치 mm (→ ESP32)
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
from std_msgs.msg import Bool, Float32, Int32, String
from std_srvs.srv import Trigger

import quvi_robot_control.topics as topics

from quvi_msgs.msg import GraspGoal

# ─── lerobot 공식 코드 import ───
import os
possible_paths = [
    '/workspace/lerobot/src',
    str(Path(__file__).resolve().parents[3] / 'lerobot' / 'src'),
    str(Path(__file__).resolve().parents[4] / 'lerobot' / 'src'),
    '/home/ksj/QUVI/lerobot/src'
]
for p in possible_paths:
    if os.path.isdir(p):
        if p not in sys.path:
            sys.path.insert(0, p)
        break

from lerobot.robots.omx_follower import OmxFollower
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.teleoperators.omx_leader import OmxLeader
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig

import quvi_robot_control.topics as topics


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


# 그리퍼 raw 위치값 (XL330-M288T)
GRIPPER_OPEN  = 2300
GRIPPER_CLOSE = 1800


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
    'shoulder_pan': 0, 'shoulder_lift': 1400, 'elbow_flex': 900,
    'wrist_flex': 1800, 'wrist_roll': 2048, 'gripper': 2300,
}
POSE_PLACE = {
    'shoulder_pan': 2048, 'shoulder_lift': 1600, 'elbow_flex': 1100,
    'wrist_flex': 2048, 'wrist_roll': 2048, 'gripper': 2300,
}

POSE_LIFT_ARM = {
    'shoulder_lift': 1800, 'elbow_flex': 1200, 'wrist_flex': 2048, 'wrist_roll': 2048,
}

POSE_PLACE_CHAMBER = {
    'shoulder_lift': 1400, 'elbow_flex': 900, 'wrist_flex': 1800,
}

POSE_CHAMBER_PRE_PICK = {
    'shoulder_pan': 0, 'shoulder_lift': 1400, 'elbow_flex': 900, 'wrist_flex': 1800, 'gripper': GRIPPER_OPEN,
}

# ── 티칭 웨이포인트 (teach_pendant.py 로 기록 후 여기에 붙여넣기) ──────────────
# scripts/teach_pendant.py 실행 → 원하는 자세로 이동 → 1~6 키 → 's' 로 출력
# 출력된 POSE_P* 값을 아래에 붙여넣으면 해당 자세로 이동합니다.
POSE_P1 = {'shoulder_pan': 2054, 'shoulder_lift': 1258, 'elbow_flex': 2800, 'wrist_flex': 2981, 'wrist_roll': 2035, 'gripper': 2150}  # 베드 위 대기
POSE_P2 = {'shoulder_pan':   12, 'shoulder_lift': 1843, 'elbow_flex': 2165, 'wrist_flex': 3123, 'wrist_roll': 2095, 'gripper': 2150}  # 180도 회전
POSE_P3 = {'shoulder_pan':   16, 'shoulder_lift': 1736, 'elbow_flex': 2413, 'wrist_flex': 3018, 'wrist_roll': 2087, 'gripper': 2150}  # 턴테이블 진입점
POSE_P4 = {'shoulder_pan':   16, 'shoulder_lift': 1841, 'elbow_flex': 2522, 'wrist_flex': 2759, 'wrist_roll': 2085, 'gripper': 2150}  # 턴테이블 놓기 지점
POSE_P5 = {'shoulder_pan': 2047, 'shoulder_lift': 1854, 'elbow_flex': 2460, 'wrist_flex': 2909, 'wrist_roll': 2050, 'gripper': 2150}  # 180도 반대 회전 (경유)
POSE_P6 = {'shoulder_pan': 2039, 'shoulder_lift': 1076, 'elbow_flex': 2884, 'wrist_flex': 3094, 'wrist_roll': 1993, 'gripper': 2150}  # 분류장 위치

# ── Dynamixel 프로파일 (⚠️ 시간기반 프로파일 모드) ───────────────────────────
# omx_follower.configure() 가 Drive_Mode Bit2(시간기반 프로파일)를 설정하므로
#   Profile_Velocity     = 이동 완료 시간(ms)   ← 속도 아님!
#   Profile_Acceleration = 가속 시간(ms)
# 값이 "작을수록 빠르고", "클수록 느리고 부드럽다". 제약: 가속시간 ≤ 이동시간/2.
# (과거 코드는 이를 속도(0.229 RPM)로 오해해 vel=1~8 을 '저속'으로 썼으나
#  실제로는 1~8ms = 사실상 최대속도였다. 시퀀스 난폭 동작의 원인. 계획서 C7.)
# ※ 아래 ms 값은 보수적 초기값이며 하드웨어에서 미세조정한다.
#
# 일반 동작 (빈 손: 검사장 이동, 홈 복귀 등)
PROFILE_VELOCITY      = 1200   # 이동시간 ms (부드러운 중속)
PROFILE_ACCEL         = 400    # 가속시간 ms
PROFILE_VELOCITY_GRIP = 800    # 그리퍼 이동시간 ms
PROFILE_ACCEL_GRIP    = 200    # 그리퍼 가속시간 ms
#
# P1~P6 시퀀스 (물체 파지 중 — 낙하/파손 방지 위해 더 느리게)
PROFILE_VELOCITY_SEQ  = 2000   # 이동시간 ms (저속 안전 이송)
PROFILE_ACCEL_SEQ     = 600    # 가속시간 ms
#
# ACT 추론 (학습 시 configure() 기본값 50ms 로 녹화됨 → 동일 유지해 동역학 일치)
PROFILE_VELOCITY_ACT  = 50     # 이동시간 ms
PROFILE_ACCEL_ACT     = 25     # 가속시간 ms

# ACT 실행 주기 (Hz)
ACT_CONTROL_HZ = 30

# ── 관절 안전 범위 (P1: ACT/시퀀스 폭주 방지) ────────────────────────────────
# 정규화 단위 안전 범위 — send_action 경로(ACT·텔레옵).
#   shoulder_pan, wrist_roll : DEGREES 모드. 1회전이 [-180°, +180°] 에 매핑되며
#     EXTENDED_POSITION 이라 unnormalize 시 클램프가 없어 다회전 폭주가 가능하다.
#     ±178° 로 제한해 한 바퀴 안에 묶는다.
#   shoulder_lift/elbow_flex/wrist_flex : RANGE_M100_100 → ±100 (unnormalize 도 자동 클램프하나 명시).
#   gripper : RANGE_0_100 → 0~100.
NORM_SAFE_RANGE = {
    'shoulder_pan':  (-178.0, 178.0),
    'shoulder_lift': (-100.0, 100.0),
    'elbow_flex':    (-100.0, 100.0),
    'wrist_flex':    (-100.0, 100.0),
    'wrist_roll':    (-178.0, 178.0),
    'gripper':       (0.0, 100.0),
}
# raw 단위 안전 범위 — sync_write 직접 경로(시퀀스·홈). 모터 Min/Max_Position_Limit 와 동일.
RAW_SAFE_RANGE = {
    'shoulder_lift': (830, 3129),
    'elbow_flex':    (1024, 3140),
    'wrist_flex':    (0, 4095),
}


# ─────────────────────────────────────────────────────────────
# RobotControlNode
# ─────────────────────────────────────────────────────────────

class RobotControlNode(Node):
    """로봇팔 + 레일 + 턴테이블 통합 제어 노드.
    
    모터 제어는 lerobot 공식 OmxFollower / OmxLeader를 사용.
    """

    def __init__(self, **kwargs):
        super().__init__('robot_control_node', **kwargs)

        # ─── 파라미터 선언 ───
        self._declare_params()
        self._load_params()

        # ─── 내부 상태 ───
        self._state: RobotState = RobotState.IDLE
        self._state_lock = threading.Lock()

        self._latest_sidecam: Optional[np.ndarray] = None
        self._sidecam_lock = threading.Lock()
        self._bridge = CvBridge()

        self._esp32_rail_done = False

        # ─── 텔레오퍼레이션 상태 ───
        self._teleop_running = False
        self._leader: Optional[OmxLeader] = None
        self._teleop_offsets = {}

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

        # ─── 관절 상태 발행 타이머 (10 Hz) ───
        self._joint_pub_timer = self.create_timer(
            1.0 / 10.0, self._publish_joint_states,
            callback_group=self._cb_group)

        # ─── 상태 주기적 브로드캐스트 타이머 (1 Hz) ───
        self._status_pub_timer = self.create_timer(
            1.0, self._broadcast_status_periodically,
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
        self.declare_parameter('use_act', False)
        self.declare_parameter('act_model_path',
            '/physical_ai_tools/lerobot/outputs/train/GUVI0625100FF/checkpoints/100000/pretrained_model')
        self.declare_parameter('act_chunk_size', 20)
        self.declare_parameter('act_device', 'cpu')   # 'cuda' or 'cpu'
        # 안전(P1): send_action(ACT·텔레옵) 1스텝 최대 상대이동량(정규화 단위).
        # 값을 낮출수록 폭주 방지 강도가 높다. 검증 후 단계적으로 상향한다.
        self.declare_parameter('act_max_relative_target', 8.0)
        # 레일 위치 (mm 단위) — 조립 후 캘리브레이션으로 확정
        self.declare_parameter('rail_mm_bed',      381.25)
        self.declare_parameter('rail_mm_inspect', 12.5)
        self.declare_parameter('rail_mm_pass',    25.0)
        self.declare_parameter('rail_mm_fail',    125.0)
        # 카메라
        self.declare_parameter('sidecam_topic', '/camera1/image_raw/compressed')
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
        # ensure_safe_goal_position 은 float/dict 만 허용하므로 반드시 float 로 전달.
        self._act_max_rel_target = float(self.get_parameter('act_max_relative_target').value)
        self._rail_mm = {
            RailPosition.BED:     self.get_parameter('rail_mm_bed').value,
            RailPosition.INSPECT: self.get_parameter('rail_mm_inspect').value,
            RailPosition.PASS:    self.get_parameter('rail_mm_pass').value,
            RailPosition.FAIL:    self.get_parameter('rail_mm_fail').value,
        }
        self._sidecam_topic  = self.get_parameter('sidecam_topic').value
        self._use_compressed = self.get_parameter('use_compressed').value
        self._rail_timeout   = self.get_parameter('rail_move_timeout_sec').value
        self._grasp_timeout  = self.get_parameter('grasp_timeout_sec').value
        self._home_timeout   = self.get_parameter('home_timeout_sec').value

    # ─────────────────────────────────────────────
    # lerobot OmxFollower 초기화
    # ─────────────────────────────────────────────
    def _init_follower(self):
        """lerobot 공식 OmxFollower를 통해 팔로워 로봇팔 초기화."""
        try:
            follower_config = OmxFollowerConfig(
                port=self._dxl_port_name,
                id='quvi_follower',
                # 안전(P1 C2): send_action 1스텝 상대이동 캡. ACT 이상치·첫 추론
                # 슬램을 하드웨어 직전에서 차단. sync_write 직접 경로(시퀀스)에는 영향 없음.
                max_relative_target=self._act_max_rel_target,
            )
            self._follower = OmxFollower(follower_config)
            self._follower.connect()
            self._dxl_ready = True
            self.get_logger().info(
                f'OmxFollower 연결 완료 | 포트={self._dxl_port_name} | '
                f'상대이동캡={self._act_max_rel_target} | '
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
            import sys
            for _lerobot_src in ['/workspace/lerobot/src', '/physical_ai_tools/lerobot/src']:
                if _lerobot_src not in sys.path:
                    sys.path.insert(0, _lerobot_src)
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
            self._sidecam_sub = self.create_subscription(
                CompressedImage, self._sidecam_topic,
                self._sidecam_callback, 10)
        else:
            from sensor_msgs.msg import Image
            self._sidecam_sub = self.create_subscription(
                Image, self._sidecam_topic,
                self._sidecam_callback_raw, 10)

        self._grasp_cmd_sub = self.create_subscription(
            GraspGoal, topics.TOPIC_ROBOT_GRASP_CMD,
            self._grasp_cmd_callback, 10)

        self._rail_cmd_sub = self.create_subscription(
            Int32, topics.TOPIC_ROBOT_RAIL_CMD,
            self._rail_cmd_callback, 10)

        self._esp32_rail_done_sub = self.create_subscription(
            Bool, topics.TOPIC_MOTOR_RAIL_DONE,
            self._esp32_rail_done_callback, 10)

        self._rotate_cmd_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_ROTATE_CMD,
            self._rotate_cmd_callback, 10)

        self._release_cmd_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_RELEASE_CMD,
            self._release_cmd_callback, 10)

        self._home_cmd_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_HOME_CMD,
            self._home_cmd_callback, 10)

        self._place_chamber_cmd_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_PLACE_CHAMBER_CMD,
            self._place_chamber_cmd_callback, 10)

        self._pick_chamber_cmd_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_PICK_CHAMBER_CMD,
            self._pick_chamber_cmd_callback, 10)

        self._teleop_cmd_sub = self.create_subscription(
            Bool, '/robot/teleop_command',
            self._teleop_cmd_callback, 10,
            callback_group=self._cb_group)

        self._estop_sub = self.create_subscription(
            Bool, '/system/estop',
            self._estop_cmd_callback, 10,
            callback_group=self._cb_group)

        self._reset_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_RESET_CMD,
            self._reset_cmd_callback, 10,
            callback_group=self._cb_group)

        # ── Publishers ──
        self._joint_state_pub = self.create_publisher(
            JointState, '/robot/joint_states', 10)

        self._rail_pub = self.create_publisher(
            Int32, topics.TOPIC_MOTOR_RAIL_CMD, 10)

        self._status_pub = self.create_publisher(
            String, topics.TOPIC_ROBOT_STATUS, 10)

        self._act_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_ACT_DONE, 10)

        self._grasp_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_GRASP_DONE, 10)

        self._release_done_pub = self.create_publisher(
            Bool, '/robot/release_done', 10)

        self._home_done_pub = self.create_publisher(
            Bool, '/robot/home_done', 10)

        self._rail_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_RAIL_DONE, 10)

        self._rotate_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_ROTATE_DONE, 10)

        self._place_chamber_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_PLACE_CHAMBER_DONE, 10)

        self._pick_chamber_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_PICK_CHAMBER_DONE, 10)

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
    def _sidecam_callback(self, msg: CompressedImage):
        import cv2
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._sidecam_lock:
                self._latest_sidecam = frame

    def _sidecam_callback_raw(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is not None:
            with self._sidecam_lock:
                self._latest_sidecam = frame

    def _esp32_rail_done_callback(self, msg: Bool):
        if msg.data:
            self._esp32_rail_done = True

    # ─────────────────────────────────────────────
    # 명령 콜백 (토픽 기반 — 비동기)
    # ─────────────────────────────────────────────
    def _try_start_command(self, target_state: RobotState, target_func, *args):
        """lock 안에서 상태를 선점하고 스레드를 시작하여 레이스 컨디션을 방지한다."""
        with self._state_lock:
            if self._state != RobotState.IDLE:
                self.get_logger().warn(f'명령 무시: 현재 {self._state.name} 동작 중')
                return False
            self._state = target_state
            self._publish_status(self._state.name)
        t = threading.Thread(target=target_func, args=args, daemon=True)
        t.start()
        return True

    def _grasp_cmd_callback(self, msg: GraspGoal):
        self.get_logger().info(
            f'파지 목표 수신(참고): idx={msg.object_index} '
            f'x={msg.target_x:.1f} y={msg.target_y:.1f} '
            f'(ACT visuomotor 추론 사용, 좌표는 직접 미사용)')
        self._try_start_command(RobotState.ACT_GRASPING, self._execute_act_grasp)

    def _rail_cmd_callback(self, msg: Int32):
        """레일 이동 명령 수신 (Int32 위치 코드) → 레일 이동 실행 (별도 스레드)."""
        pos_code = msg.data
        try:
            pos = RailPosition(pos_code)
        except ValueError:
            self.get_logger().error(f'알 수 없는 레일 위치 코드: {pos_code}')
            return
        self._try_start_command(RobotState.MOVING_RAIL, self._execute_rail_move, pos)

    def _rotate_cmd_callback(self, msg: Bool):
        pan_val = 0 if msg.data else 2048
        pose = {'shoulder_pan': pan_val}
        self._try_start_command(RobotState.ROTATING_BASE, self._execute_pose, pose, f'베이스 회전(pan={pan_val})')

    def _release_cmd_callback(self, msg: Bool):
        if not msg.data:
            return
        self._try_start_command(RobotState.RELEASING, self._execute_release)

    def _home_cmd_callback(self, msg: Bool):
        if not msg.data:
            return
        self._try_start_command(RobotState.HOMING, self._execute_home)

    def _place_chamber_cmd_callback(self, msg: Bool):
        if not msg.data:
            return
        self._try_start_command(RobotState.RELEASING, self._execute_place_in_chamber)

    def _pick_chamber_cmd_callback(self, msg: Bool):
        if not msg.data:
            return
        self._try_start_command(RobotState.ACT_GRASPING, self._execute_pick_from_chamber)

    def _estop_cmd_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().error('비상 정지 명령 수신! 동작 강제 중단 및 에러 상태 천이')
            self._safe_estop_cleanup()

    def _reset_cmd_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info('로봇 리셋 및 Dynamixel 포트 재연결 시도...')
            t = threading.Thread(target=self._execute_reset, daemon=True)
            t.start()

    # ─────────────────────────────────────────────
    # 서비스 핸들러
    # ─────────────────────────────────────────────
    def _act_grasp_service(self, request, response):
        success = self._execute_act_grasp()
        response.success = success
        response.message = 'ACT 파지 완료' if success else 'ACT 파지 실패'
        return response

    def _go_home_service(self, request, response):
        success = self._execute_home()
        response.success = success
        response.message = '홈 복귀 완료' if success else '홈 복귀 실패'
        return response

    def _open_gripper_service(self, request, response):
        self._write_raw_position({'gripper': GRIPPER_OPEN})
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 열기 완료 (OmxFollower ID16)'
        return response

    def _close_gripper_service(self, request, response):
        self._write_raw_position({'gripper': GRIPPER_CLOSE})
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 닫기 완료 (OmxFollower ID16)'
        return response

    # ─────────────────────────────────────────────
    # 실행 함수 — ACT 파지
    # ─────────────────────────────────────────────
    def _execute_act_grasp(self) -> bool:
        self._set_state(RobotState.ACT_GRASPING)

        # ACT를 사용하지 않거나 로드가 안된 경우 룰베이스(티칭 기반) 파지 수행
        if not self._use_act or not self._act_ready:
            self.get_logger().info('ACT 미사용 또는 로드 안됨 — 룰베이스(티칭 기반) 파지를 실행합니다.')
            self._publish_status('룰베이스 파지 시작')
            
            try:
                # 사용자가 수동으로 물려준 상태이므로, 바로 POSE_P1(베드 위 대기)로 이동합니다.
                self.get_logger().info('1. POSE_P1 위치로 즉시 이동 (그리퍼는 닫힘 상태 유지)')
                
                # POSE_P1 자세로 이동하되, 그리퍼는 사용자가 물려준 물체를 쥐도록 GRIPPER_CLOSE(1800)로 덮어씌웁니다.
                p1_pose = {k: v for k, v in POSE_P1.items()}
                p1_pose['gripper'] = GRIPPER_CLOSE
                
                self._write_raw_position(p1_pose)
                self._wait_motion_done(p1_pose)

                done_msg = Bool()
                done_msg.data = True
                self._act_done_pub.publish(done_msg)
                self._grasp_done_pub.publish(done_msg)

                self._set_state(RobotState.IDLE)
                self._publish_status('룰베이스 파지 완료')
                return True
            except Exception as e:
                self.get_logger().error(f'룰베이스 파지 중 오류: {e}')
                self._set_state(RobotState.ERROR)
                self._publish_status(f'ERROR: {e}')
                return False

        try:
            import torch
            import cv2
            # P2: 이전 파지 에피소드의 낡은 액션 큐를 비운다. reset() 없이는 두 번째
            # 파지부터 직전 관측 기반의 남은 액션이 먼저 실행돼 예기치 않게 움직인다.
            self._act_policy.reset()
            # P3(C6): ACT용 프로파일을 명시적으로 설정해 직전 동작(시퀀스=2000ms 등)의
            # 프로파일 누수를 제거한다. 학습 녹화 시 configure() 기본값(50ms)이 적용됐으므로
            # 동일하게 맞춰 학습 동역학과 일치시킨다.
            self._apply_motor_profile(JOINT_NAMES, PROFILE_VELOCITY_ACT, PROFILE_ACCEL_ACT)
            ACT_GRASP_DURATION = 10.0
            dt = 1.0 / ACT_CONTROL_HZ
            start = time.time()
            chunk_count = 0

            while (time.time() - start) < ACT_GRASP_DURATION:
                if self._get_state() == RobotState.ERROR:
                    self.get_logger().error('ESTOP 감지 — ACT 청크 실행을 중단합니다.')
                    self._safe_estop_cleanup()
                    return False

                with self._sidecam_lock:
                    frame = self._latest_sidecam

                if frame is None:
                    self.get_logger().error('사이드캠 이미지 없음 — 파지 불가')
                    self._set_state(RobotState.IDLE)
                    return False

                frame_rgb = cv2.cvtColor(
                    cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(
                    frame_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(self._act_device_obj)

                acquired = self._dxl_io_lock.acquire(blocking=False)
                if not acquired:
                    continue
                try:
                    norm_positions = self._follower.bus.sync_read('Present_Position', normalize=True)
                except Exception as e:
                    self.get_logger().error(f'관절 위치 읽기 실패 — 통신 오류. 파지를 중단합니다.')
                    self._set_state(RobotState.ERROR)
                    self._publish_status('ERROR: Dynamixel 통신 오류')
                    return False
                finally:
                    self._dxl_io_lock.release()

                joint_vals = [norm_positions[name] for name in JOINT_NAMES]
                state_tensor = torch.tensor(
                    joint_vals, dtype=torch.float32).unsqueeze(0)
                state_tensor = state_tensor.to(self._act_device_obj)

                obs = {
                    'observation.images.camera1': img_tensor,
                    'observation.state': state_tensor,
                }

                infer_start = time.time()
                with torch.no_grad():
                    action_chunk = self._act_policy.select_action(obs)
                    if action_chunk.ndim == 3:
                        action_chunk = action_chunk.squeeze(0)
                    action_chunk = action_chunk.cpu().numpy()
                chunk_count += 1
                self.get_logger().info(
                    f'ACT 청크#{chunk_count} 추론 완료 | 크기={len(action_chunk)} | '
                    f'추론시간={(time.time()-infer_start)*1000:.1f}ms | '
                    f'경과={time.time()-start:.1f}/{ACT_GRASP_DURATION:.0f}s')

                for i, action in enumerate(action_chunk):
                    if (time.time() - start) >= ACT_GRASP_DURATION:
                        break
                    if self._get_state() == RobotState.ERROR:
                        self.get_logger().error(f'ESTOP 감지 — ACT 실행 중단 (청크#{chunk_count} 스텝 {i})')
                        self._safe_estop_cleanup()
                        return False

                    step_start = time.time()
                    if self._use_real_hardware and self._dxl_ready:
                        action_dict = {f"{name}.pos": float(action[j]) for j, name in enumerate(JOINT_NAMES)}
                        action_dict = self._clip_safe_targets(action_dict, is_raw=False)
                        with self._dxl_io_lock:
                            self._follower.send_action(action_dict)
                    else:
                        goal_dict = {}
                        for j, name in enumerate(JOINT_NAMES):
                            val = float(action[j])
                            if -100 <= val <= 100:
                                raw = int(((val + 100.0) / 200.0) * 4095.0)
                            else:
                                raw = int(np.clip((val / (2 * math.pi)) * 4095, 0, 4095))
                            goal_dict[name] = raw
                        self._write_raw_position(goal_dict)

                    elapsed = time.time() - step_start
                    remaining = dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            total_time = time.time() - start
            self.get_logger().info(f'ACT 파지 완료 | 총 소요={total_time:.2f}s | 청크 수={chunk_count}')

            done_msg = Bool()
            done_msg.data = True
            self._act_done_pub.publish(done_msg)
            self._grasp_done_pub.publish(done_msg)

            self._set_state(RobotState.IDLE)
            self._publish_status('ACT 파지 완료')
            return True

        except Exception as e:
            self.get_logger().error(f'ACT 파지 중 오류: {e}')
            if self._follower and hasattr(self._follower, 'bus') and hasattr(self._follower.bus, 'port_handler'):
                self._follower.bus.port_handler.is_using = False
            self._set_state(RobotState.ERROR)
            self._publish_status(f'ERROR: {e}')
            return False

    # ─────────────────────────────────────────────
    # 실행 함수 — 레일 이동
    # ─────────────────────────────────────────────
    def _execute_rail_move(self, position: RailPosition) -> bool:
        """
        레일을 지정 위치로 이동.
        /motor/rail 토픽으로 steps(Int32) 발행 → ESP32-S3가 TB6600 구동.
        """
        self._set_state(RobotState.MOVING_RAIL)
        target_mm = float(self._rail_mm[position])
        pos_name = position.name
        self._publish_status(f'레일 이동: {pos_name} ({target_mm:.2f}mm)')
        self.get_logger().info(f'레일 이동 명령: {pos_name} = {target_mm:.2f}mm')

        if self._use_real_hardware:
            self._esp32_rail_done = False

        msg = Int32()
        msg.data = int(target_mm * 80)
        self._rail_pub.publish(msg)

        if self._use_real_hardware:
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
        self._set_state(RobotState.ROTATING_BASE)
        self._publish_status(f'자세 변경: {label}')
        self.get_logger().info(f'자세 변경: {label} → {target_pose}')

        if target_pose == POSE_BACK and POSE_BACK == POSE_FRONT:
            self.get_logger().warn('경고: POSE_BACK과 POSE_FRONT의 모터 제어값이 동일합니다.')

        success = self._write_raw_position(target_pose)
        time.sleep(1.5)

        self._set_state(RobotState.IDLE)
        self._publish_status(f'자세 변경 완료: {label}')

        done_msg = Bool()
        done_msg.data = True
        self._rotate_done_pub.publish(done_msg)

        return success

    # ─────────────────────────────────────────────
    # 실행 함수 — 투하
    # ─────────────────────────────────────────────
    def _execute_release(self) -> bool:
        self._set_state(RobotState.RELEASING)
        self._publish_status('웨이포인트 시퀀스 시작 (P1~P6)')
        self.get_logger().info('웨이포인트 시퀀스 시작')

        success = self._execute_taught_sequence()

        done_msg = Bool()
        done_msg.data = success
        self._release_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status('웨이포인트 시퀀스 완료' if success else '웨이포인트 시퀀스 실패')
        return success

    # ─────────────────────────────────────────────
    # 실행 함수 — 홈 복귀
    # ─────────────────────────────────────────────
    def _execute_home(self) -> bool:
        self._set_state(RobotState.HOMING)
        self._publish_status('홈 복귀')
        self.get_logger().info('홈 복귀 시작')

        success = self._write_raw_position(POSE_HOME)
        self._wait_motion_done(POSE_HOME)

        done_msg = Bool()
        done_msg.data = True
        self._home_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status('홈 복귀 완료')
        return success

    def _execute_place_in_chamber(self) -> bool:
        self._set_state(RobotState.RELEASING)
        self._publish_status('검사장 안착 시퀀스 시작')
        self.get_logger().info('검사장 안착 시퀀스 시작')

        self._write_raw_position(POSE_LIFT_ARM)
        self._wait_motion_done(POSE_LIFT_ARM)

        self._write_raw_position({'shoulder_pan': 0})
        self._wait_motion_done({'shoulder_pan': 0})

        self._write_raw_position(POSE_PLACE_CHAMBER)
        self._wait_motion_done(POSE_PLACE_CHAMBER)

        self._write_raw_position({'gripper': GRIPPER_OPEN})
        self._wait_motion_done({'gripper': GRIPPER_OPEN})

        success = self._write_raw_position(POSE_HOME)
        self._wait_motion_done(POSE_HOME)

        done_msg = Bool()
        done_msg.data = True
        self._place_chamber_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status('검사장 안착 완료')
        return success

    def _execute_pick_from_chamber(self) -> bool:
        self._set_state(RobotState.ACT_GRASPING)
        self._publish_status('검사장 재파지 시퀀스 시작')
        self.get_logger().info('검사장 재파지 시퀀스 시작')

        self._write_raw_position(POSE_CHAMBER_PRE_PICK)
        self._wait_motion_done(POSE_CHAMBER_PRE_PICK)

        self._write_raw_position({'gripper': GRIPPER_CLOSE})
        self._wait_motion_done({'gripper': GRIPPER_CLOSE})

        self._write_raw_position(POSE_LIFT_ARM)
        self._wait_motion_done(POSE_LIFT_ARM)

        success = self._write_raw_position({'shoulder_pan': 2048})
        self._wait_motion_done({'shoulder_pan': 2048})  # 180° 복귀 완료 확인 후 done 발행

        done_msg = Bool()
        done_msg.data = True
        self._pick_chamber_done_pub.publish(done_msg)

        self._set_state(RobotState.IDLE)
        self._publish_status('검사장 재파지 완료')
        return success

    def _clip_safe_targets(self, positions: dict, is_raw: bool) -> dict:
        """관절 목표값을 안전 범위로 클램프한다 (P1 C3·C4).

        is_raw=False : send_action 정규화 입력(ACT·텔레옵). DEGREES 관절
                       (shoulder_pan, wrist_roll)의 다회전 폭주를 ±178° 로 차단.
                       → NORM_SAFE_RANGE 적용.
        is_raw=True  : sync_write raw 입력(시퀀스·홈). 모터 Min/Max_Position_Limit
                       와 동일한 범위로 클램프. → RAW_SAFE_RANGE 적용.

        키는 'name' 과 'name.pos' 양식을 모두 지원한다.
        범위에 없는 관절(raw 경로의 shoulder_pan/wrist_roll/gripper 등)은 그대로 둔다.
        """
        table = RAW_SAFE_RANGE if is_raw else NORM_SAFE_RANGE
        out = dict(positions)
        for key, val in positions.items():
            joint = key[:-4] if key.endswith('.pos') else key
            rng = table.get(joint)
            if rng is None:
                continue
            lo, hi = rng
            if val < lo or val > hi:
                clamped = min(hi, max(lo, val))
                self.get_logger().warn(
                    f'[안전클램프] {joint} {val:.2f} → {clamped:.2f} '
                    f'({"raw" if is_raw else "norm"} {lo}~{hi})',
                    throttle_duration_sec=2.0)
                out[key] = clamped
        return out

    # ─────────────────────────────────────────────
    # 모터 프로파일 적용 (시간기반: velocity=이동시간 ms, accel=가속시간 ms)
    # ─────────────────────────────────────────────
    def _apply_motor_profile(self, joint_names: list, velocity: int, accel: int):
        if not self._use_real_hardware or not self._dxl_ready or not self._follower:
            return
        try:
            with self._dxl_io_lock:
                vel_dict   = {n: velocity for n in joint_names}
                accel_dict = {n: accel    for n in joint_names}
                self._follower.bus.sync_write('Profile_Acceleration', accel_dict, normalize=False)
                self._follower.bus.sync_write('Profile_Velocity',     vel_dict,   normalize=False)
        except Exception as e:
            self.get_logger().warn(f'프로파일 설정 실패 (계속 진행): {e}')

    # ─────────────────────────────────────────────
    # 모터 이동 완료 폴링
    # ─────────────────────────────────────────────
    def _wait_motion_done(self, goal: dict, timeout: float = 10.0, tol: int = 20):
        """Present_Position 폴링으로 목표 위치 도달을 확인.

        goal: _write_raw_position에 넘긴 것과 동일한 {joint: raw_value} 딕셔너리.
        tol:  허용 오차 (raw 단위, 20 ≈ 1.75°).
        """
        if not self._use_real_hardware or not self._dxl_ready:
            return
        time.sleep(0.05)  # Goal_Position 반영 여유
        deadline = time.time() + timeout
        while time.time() < deadline:
            positions = self._read_raw_positions()
            if positions is None:
                time.sleep(0.05)
                continue
            if all(abs(positions.get(j, 0) - v) <= tol for j, v in goal.items()):
                time.sleep(0.1)  # 기계 안정화
                return
            time.sleep(0.05)
        self.get_logger().warn(f'[_wait_motion_done] {timeout}s 타임아웃 — 강제 진행')

    # ─────────────────────────────────────────────
    # lerobot bus 기반 모터 I/O
    # ─────────────────────────────────────────────
    def _write_raw_position(self, positions: dict,
                            velocity: int = PROFILE_VELOCITY,
                            accel: int = PROFILE_ACCEL,
                            grip_velocity: int = PROFILE_VELOCITY_GRIP,
                            grip_accel: int = PROFILE_ACCEL_GRIP) -> bool:
        positions = self._clip_safe_targets(positions, is_raw=True)
        if not self._use_real_hardware or not self._dxl_ready:
            self.get_logger().debug(f'[SIM] 관절 목표: {positions}')
            return True

        try:
            arm_joints_to_apply = [k for k in positions.keys() if k != 'gripper']
            if arm_joints_to_apply:
                self._apply_motor_profile(arm_joints_to_apply, velocity, accel)
            if 'gripper' in positions:
                self._apply_motor_profile(['gripper'], grip_velocity, grip_accel)

            with self._dxl_io_lock:
                self._follower.bus.sync_write(
                    'Goal_Position', positions, normalize=False)
            return True
        except Exception as e:
            self.get_logger().error(f'lerobot sync_write 오류: {e}')
            if self._follower and hasattr(self._follower, 'bus') and hasattr(self._follower.bus, 'port_handler'):
                self._follower.bus.port_handler.is_using = False
            return False

    def _read_raw_positions(self) -> Optional[dict]:
        if not self._use_real_hardware or not self._dxl_ready:
            return {name: 2048 for name in JOINT_NAMES}

        acquired = self._dxl_io_lock.acquire(blocking=False)
        if not acquired:
            return None

        try:
            return self._follower.bus.sync_read(
                'Present_Position', normalize=False)
        except Exception as e:
            self.get_logger().error(f'lerobot sync_read 오류: {e}')
            if self._follower and hasattr(self._follower, 'bus') and hasattr(self._follower.bus, 'port_handler'):
                self._follower.bus.port_handler.is_using = False
            return None
        finally:
            self._dxl_io_lock.release()

    # ─────────────────────────────────────────────
    # 관절 상태 발행 (30 Hz 타이머)
    # ─────────────────────────────────────────────
    def _publish_joint_states(self):
        raw_positions = self._read_raw_positions()
        if raw_positions is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = JOINT_NAMES
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

    def _broadcast_status_periodically(self):
        """1Hz 주기로 HMI 상태 채널에 현재 상태를 브로드캐스트하여 오케스트레이터 등의 기동 타이밍 이슈를 방지한다."""
        if self._use_act and self._act_ready:
            if self._get_state() == RobotState.IDLE:
                self._publish_status('ACT_READY')
            else:
                self._publish_status(self._get_state().name)
        else:
            self._publish_status(self._get_state().name)

    def _safe_estop_cleanup(self):
        try:
            if self._dxl_ready and self._follower:
                if hasattr(self._follower, 'bus'):
                    torque_off = {name: 0 for name in JOINT_NAMES}
                    with self._dxl_io_lock:
                        self._follower.bus.sync_write('Torque_Enable', torque_off, normalize=False)
                self._dxl_ready = False
        except Exception as e:
            self.get_logger().error(f'ESTOP cleanup 중 오류: {e}')

        self._set_state(RobotState.ERROR)
        self._publish_status('ERROR: ESTOP ACTIVE — 토크 해제됨')

    def _execute_reset(self) -> bool:
        with self._dxl_io_lock:
            if self._follower:
                try:
                    self._follower.disconnect()
                except Exception:
                    pass
                self._follower = None
            self._dxl_ready = False
            
            if self._use_real_hardware:
                self._init_follower()
                
            if not self._use_real_hardware or self._dxl_ready:
                self._set_state(RobotState.IDLE)
                self._publish_status('ACT_READY')
                self.get_logger().info('로봇 리셋 및 Dynamixel 재연결 완료 -> IDLE 상태 전이')
                return True
            else:
                self._set_state(RobotState.ERROR)
                self._publish_status('ERROR: RESET FAILED')
                return False

    # ─────────────────────────────────────────────
    # P1~P6 티칭 시퀀스 실행 (저속 프로파일)
    # ─────────────────────────────────────────────
    def _execute_taught_sequence(self) -> bool:
        """
        P1~P6 웨이포인트 저속 실행 (시간기반 프로파일 PROFILE_VELOCITY_SEQ=2000ms).
        레일 이동은 오케스트레이터가 이미 처리하므로 팔 동작만 담당.

        순서 (test_sequence.py 기반):
          P1 → P2 → P3 → P4(놓기) → P3(후퇴) → P4(재집기)
          → P3(올리기) → P5(반대 회전) → P1(경유) → P6(최종 놓기)
        """
        def move_arm(pose: dict, label: str) -> bool:
            self.get_logger().info(f'이동: {label}')
            arm_only = {k: v for k, v in pose.items() if k != 'gripper'}
            success = self._write_raw_position(
                arm_only,
                velocity=PROFILE_VELOCITY_SEQ,
                accel=PROFILE_ACCEL_SEQ,
            )
            self._wait_motion_done(arm_only)
            return success

        def grip_open():
            self.get_logger().info('그리퍼 열기')
            self._write_raw_position(
                {'gripper': GRIPPER_OPEN},
                grip_velocity=PROFILE_VELOCITY_GRIP,
                grip_accel=PROFILE_ACCEL_GRIP,
            )
            self._wait_motion_done({'gripper': GRIPPER_OPEN})

        def grip_close():
            self.get_logger().info('그리퍼 닫기')
            self._write_raw_position(
                {'gripper': GRIPPER_CLOSE},
                grip_velocity=PROFILE_VELOCITY_GRIP,
                grip_accel=PROFILE_ACCEL_GRIP,
            )
            self._wait_motion_done({'gripper': GRIPPER_CLOSE})

        # P1 → P2 → P3 → P4 → 놓기
        for pose, label in [(POSE_P1, 'P1: 베드 위 대기'),
                            (POSE_P2, 'P2: 180도 회전'),
                            (POSE_P3, 'P3: 턴테이블 진입점'),
                            (POSE_P4, 'P4: 턴테이블 놓기 지점')]:
            if not move_arm(pose, label):
                return False
        grip_open()

        # P3으로 후퇴 후 P4로 재진입 (집기 준비 — 검사는 오케스트레이터가 이미 완료)
        if not move_arm(POSE_P3, 'P3: 후퇴'):
            return False
        if not move_arm(POSE_P4, 'P4: 재집기 지점'):
            return False
        grip_close()

        # P3 올리기 → P5 반대 회전 → P1 경유 → P6 최종 놓기
        if not move_arm(POSE_P3, 'P3: 들어올리기'):
            return False
        if not move_arm(POSE_P5, 'P5: 180도 반대 회전'):
            return False
        if not move_arm(POSE_P1, 'P1: 경유'):
            return False
        if not move_arm(POSE_P6, 'P6: 분류장 위치'):
            return False
        grip_open()

        return True

    # ─────────────────────────────────────────────
    # 종료 처리
    # ─────────────────────────────────────────────
    def destroy_node(self):
        if self._teleop_running:
            self._stop_teleop()

        if self._follower and self._dxl_ready:
            self.get_logger().info('OmxFollower 연결 해제 (토크 비활성화 포함)')
            try:
                self._follower.disconnect()
            except Exception as e:
                self.get_logger().warn(f'OmxFollower 해제 중 오류: {e}')
            finally:
                self._dxl_ready = False
                self._follower = None
        super().destroy_node()

    # ─────────────────────────────────────────────
    # 텔레오퍼레이션 제어
    # ─────────────────────────────────────────────
    def _teleop_cmd_callback(self, msg: Bool):
        if msg.data:
            t = threading.Thread(target=self._start_teleop, daemon=True)
            t.start()
        else:
            t = threading.Thread(target=self._stop_teleop, daemon=True)
            t.start()

    def _start_teleop(self) -> bool:
        if self._get_state() == RobotState.TELEOPING:
            return True
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('텔레옵 무시: 현재 로봇이 IDLE 상태가 아님')
            return False

        self._set_state(RobotState.TELEOPING)
        self._publish_status('텔레오퍼레이션 활성화')
        self.get_logger().info('텔레오퍼레이션 시작 중...')

        if self._use_real_hardware:
            max_retries = 3
            retry_delay = 0.8
            connected = False
            for attempt in range(1, max_retries + 1):
                try:
                    leader_config = OmxLeaderConfig(
                        port=self._leader_port_name,
                        id='quvi_leader',
                    )
                    self._leader = OmxLeader(leader_config)
                    self._leader.connect()
                    connected = True
                    self.get_logger().info(
                        f'OmxLeader 연결 완료 (시도 {attempt}/{max_retries}) | 포트={self._leader_port_name}')
                    break
                except Exception as e:
                    self.get_logger().warn(f'OmxLeader 연결 실패 (시도 {attempt}/{max_retries}): {e}')
                    self._leader = None
                    if attempt < max_retries:
                        time.sleep(retry_delay)
            if not connected:
                self.get_logger().error(f'OmxLeader 연결 최종 실패 (총 {max_retries}회 시도)')
                self._set_state(RobotState.IDLE)
                self._publish_status('텔레옵 에러: 리더 포트 연결 불가')
                return False

        self._teleop_running = True

        if self._use_real_hardware and self._leader and self._follower:
            try:
                self.get_logger().info('텔레옵 시작 전 리더-팔로워 위치 정렬 중...')
                with self._dxl_io_lock:
                    target_action = self._leader.get_action()
                
                with self._dxl_io_lock:
                    current_norm_positions = self._follower.bus.sync_read("Present_Position")
                
                if current_norm_positions and target_action:
                    self._teleop_offsets = {}
                    for joint in ['shoulder_pan', 'wrist_roll']:
                        leader_key = f"{joint}.pos"
                        if leader_key in target_action and joint in current_norm_positions:
                            leader_val = target_action[leader_key]
                            follower_val = current_norm_positions[joint]
                            diff = leader_val - follower_val
                            wrapped_diff = (diff + 180.0) % 360.0 - 180.0
                            self._teleop_offsets[joint] = leader_val - (follower_val + wrapped_diff)

                    corrected_target_action = dict(target_action)
                    for joint, offset in self._teleop_offsets.items():
                        leader_key = f"{joint}.pos"
                        if leader_key in corrected_target_action:
                            corrected_target_action[leader_key] -= offset

                    steps = 30
                    dt = 1.5 / steps
                    target_goals = {key.removesuffix(".pos"): val for key, val in corrected_target_action.items() if key.endswith(".pos")}
                    
                    for step in range(1, steps + 1):
                        if not self._teleop_running:
                            break
                        alpha = step / float(steps)
                        interpolated_action = {}
                        for joint in JOINT_NAMES:
                            start_val = current_norm_positions.get(joint, 0.0)
                            target_val = target_goals.get(joint, start_val)
                            interp_val = start_val + alpha * (target_val - start_val)
                            interpolated_action[f"{joint}.pos"] = interp_val
                        
                        with self._dxl_io_lock:
                            self._follower.send_action(interpolated_action)
                        time.sleep(dt)
                self.get_logger().info('리더-팔로워 정렬 완료.')
            except Exception as align_err:
                self.get_logger().warn(f'텔레옵 시작 정렬 중 에러 (무시하고 진행): {align_err}')
                if self._follower and hasattr(self._follower, 'bus') and hasattr(self._follower.bus, 'port_handler'):
                    self._follower.bus.port_handler.is_using = False
                if self._leader and hasattr(self._leader, 'bus') and hasattr(self._leader.bus, 'port_handler'):
                    self._leader.bus.port_handler.is_using = False

        self._teleop_thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self._teleop_thread.start()
        self.get_logger().info('텔레오퍼레이션 루프 시작됨')
        return True

    def _stop_teleop(self) -> bool:
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
        dt = 1.0 / 50.0
        sim_angle = 0.0

        while self._teleop_running and rclpy.ok():
            start_time = time.time()

            if self._use_real_hardware and self._leader and self._follower:
                try:
                    with self._dxl_io_lock:
                        action = self._leader.get_action()
                    
                    if hasattr(self, '_teleop_offsets') and self._teleop_offsets:
                        action = dict(action)
                        for joint, offset in self._teleop_offsets.items():
                            key = f"{joint}.pos"
                            if key in action:
                                action[key] -= offset

                    action = self._clip_safe_targets(action, is_raw=False)
                    with self._dxl_io_lock:
                        self._follower.send_action(action)
                except Exception as e:
                    self.get_logger().warn(f'텔레옵 루프 오류: {e}')
                    if self._follower and hasattr(self._follower, 'bus') and hasattr(self._follower.bus, 'port_handler'):
                        self._follower.bus.port_handler.is_using = False
                    if self._leader and hasattr(self._leader, 'bus') and hasattr(self._leader.bus, 'port_handler'):
                        self._leader.bus.port_handler.is_using = False
            else:
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

            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
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
