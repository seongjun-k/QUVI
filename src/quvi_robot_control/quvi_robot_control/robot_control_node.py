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
  4. 레일 이동 명령 → ESP32-S3 (/motor/rail Int32 steps)
  5. 턴테이블 회전 명령 → ESP32-S3 (/motor/turntable_cmd)

ROS 2 인터페이스 (Subscriber, 토픽명은 topics.py SSoT — 일부 예외는 quvi_hmi/hmi_node.py 자체 상수):
  <sidecam_topic 파라미터>   sensor_msgs/(Compressed)Image  사이드캠 이미지 (기본 /camera1/image_raw/compressed)
  /robot/grasp_command       quvi_msgs/GraspGoal            파지 트리거 (ACT는 이미지로 추론)
  /robot/rail_command        std_msgs/Int32                 레일 목표 위치 코드 (0=BED,1=INSPECT,2=PASS,3=FAIL)
  /motor/rail_done           std_msgs/Bool                  ESP32 레일 이동 완료 (MOVING_RAIL 중에만 수락)
  /robot/release_command     std_msgs/Bool                  웨이포인트 시퀀스(P1~P6) 실행
  /robot/home_command        std_msgs/Bool                  홈 복귀
  /robot/place_in_chamber    std_msgs/Bool                  검사장 안착 시퀀스
  /robot/pick_in_chamber     std_msgs/Bool                  검사장 재파지 시퀀스
  /robot/teleop_command      std_msgs/Bool                  텔레옵 시작/종료
  /system/estop              std_msgs/Bool                  비상 정지 → 토크 차단 + ERROR 전환
  /robot/reset_command       std_msgs/Bool                  리셋 + Dynamixel 포트 재연결
  /robot/act_model_select    std_msgs/String                ACT 모델 경로 선택(대시보드) → 백그라운드 재로드
  /hmi/command               std_msgs/String                STOP/ESTOP/START/RESET → 소프트 중단 제어

ROS 2 인터페이스 (Publisher):
  /robot/joint_states           sensor_msgs/JointState       현재 관절 각도 (10 Hz)
  /motor/rail                   std_msgs/Int32               레일 목표 위치 steps (→ ESP32)
  /robot/status                 std_msgs/String              상태 문자열 (1 Hz 주기 브로드캐스트 포함)
  /robot/grasp_done             std_msgs/Bool                파지 완료 신호
  /robot/release_done           std_msgs/Bool                웨이포인트 시퀀스 완료 신호
  /robot/home_done              std_msgs/Bool                홈 복귀 완료 신호
  /robot/rail_done              std_msgs/Bool                레일 이동 완료 신호
  /robot/place_in_chamber_done  std_msgs/Bool                검사장 안착 완료 신호
  /robot/pick_in_chamber_done   std_msgs/Bool                검사장 재파지 완료 신호
  /robot/act_models             std_msgs/String              스캔된 ACT 모델 목록 (latched)
  /robot/act_current            std_msgs/String              현재 ACT 모델·로드 상태 (latched)

ROS 2 서비스 (Server):
  /robot/act_grasp            std_srvs/Trigger              ACT 파지 실행 (동기)
  /robot/go_home              std_srvs/Trigger              홈 복귀 실행 (동기)
  /robot/open_gripper         std_srvs/Trigger              그리퍼 열기
  /robot/close_gripper        std_srvs/Trigger              그리퍼 닫기
"""

import json
import math
import queue
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger

import quvi_robot_control.topics as topics
from quvi_robot_control.utils import decode_compressed

from quvi_msgs.msg import GraspGoal

# ─── lerobot 공식 코드 import ───
import os
possible_paths = [
    '/workspace/lerobot/src',
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
    ACT_GRASPING  = 4
    RELEASING     = 6
    TELEOPING     = 8
    ERROR         = 99


# 그리퍼 raw 위치값 (XL330-M288T)
GRIPPER_OPEN  = 2500
GRIPPER_CLOSE = 1800


# 관절 이름 (ROS 2 JointState 메시지용)
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# 사전 정의 자세 — raw Dynamixel 위치값 (0~4095 = 0~360°)
# dict 형태로 lerobot bus에 직접 전달 (normalize=False)

# ── 티칭 웨이포인트 (teach_pendant.py 로 기록 후 여기에 붙여넣기) ──────────────
# scripts/teach_pendant.py 실행 → 원하는 자세로 이동 → 1~6 키 → 's' 로 출력
# 출력된 POSE_P* 값을 아래에 붙여넣으면 해당 자세로 이동합니다.
POSE_P1 = {'shoulder_pan': 2053, 'shoulder_lift':  978, 'elbow_flex': 3050, 'wrist_flex': 3056, 'wrist_roll': 2029, 'gripper': 2100}  # 베드 위 대기
POSE_P2 = {'shoulder_pan':  -13, 'shoulder_lift': 1126, 'elbow_flex': 2752, 'wrist_flex': 3012, 'wrist_roll': 2006, 'gripper': 2100}  # 180도 회전
POSE_P3 = {'shoulder_pan':   -1, 'shoulder_lift': 1760, 'elbow_flex': 2244, 'wrist_flex': 3120, 'wrist_roll': 2009, 'gripper': 2100}  # 턴테이블 진입점
POSE_P4 = {'shoulder_pan':   16, 'shoulder_lift': 1907, 'elbow_flex': 2464, 'wrist_flex': 2822, 'wrist_roll': 2015, 'gripper': 2100}  # 턴테이블 놓기 지점
POSE_P5 = {'shoulder_pan': 2119, 'shoulder_lift': 1200, 'elbow_flex': 2824, 'wrist_flex': 3168, 'wrist_roll': 2126, 'gripper': 2100}  # 180도 반대 회전 (경유)
POSE_P6 = {'shoulder_pan': 2110, 'shoulder_lift': 1341, 'elbow_flex': 3016, 'wrist_flex': 2816, 'wrist_roll': 2124, 'gripper': 2100}  # 분류장 위치

# 홈 자세는 P1과 동일하되 그리퍼는 완전 개방 — ACT 파지 시작 자세 (2026-07-10 사용자 지정)
POSE_HOME = dict(POSE_P1, gripper=GRIPPER_OPEN)

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

# 그리퍼 개폐 대기 (#2). 그리퍼는 전류기반 위치제어라 물체를 쥐면 목표 위치에
# 도달하지 못하므로 위치 수렴 폴링 대신 고정 시간만 대기한다.
GRIPPER_SETTLE_SEC    = 1.2

# ACT 실행 주기 (Hz)
ACT_CONTROL_HZ = 30

# 대시보드에서 마지막으로 선택한 ACT 모델 경로 저장 파일 —
# 재기동 시 파라미터 기본값 대신 이 경로를 우선 복원한다.
ACT_LAST_MODEL_FILE = '/workspace/data/act_last_model.json'

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

# ── ACT rad↔raw 변환 상수 (학습 데이터셋 단위 — lerobot 정규화와 별개) ─────────
DXL_TICKS_PER_REV = 4096.0   # XL430/XL330 1회전 틱 수 (변환 분모)
DXL_TICK_CENTER   = 2048.0   # 중립(0 rad) 틱
DXL_RAW_MAX       = 4095     # Goal_Position 클램프 상한


def raw_to_rad(raw: float) -> float:
    """raw Dynamixel 위치값 → 라디안 (ACT 학습 데이터셋 단위)."""
    return (raw - DXL_TICK_CENTER) * (2.0 * math.pi) / DXL_TICKS_PER_REV


def rad_to_raw(rad: float) -> float:
    """라디안(ACT 정책 출력) → raw Dynamixel 위치값."""
    return DXL_TICK_CENTER + rad * DXL_TICKS_PER_REV / (2.0 * math.pi)


def _arm_only(pose: dict) -> dict:
    """그리퍼 제외 팔 관절만 반환.

    그리퍼는 전류기반 위치제어라 물체를 쥐면 pose 의 목표값에 도달하지 못해
    _wait_motion_done 이 풀타임아웃하고 파지력도 느슨해진다 — 시퀀스 이동은
    팔 관절만 보내고 그리퍼는 명시 명령으로만 제어한다.
    """
    return {k: v for k, v in pose.items() if k != 'gripper'}


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
        # 명령 세대 토큰 — RESET/ESTOP 후 잔존 동작 스레드가 상태를 덮어쓰지 못하게 함
        self._cmd_gen = 0

        # STOP/ESTOP 소프트 중단 이벤트 (#1/#3). 진행 중 동작 스레드가 이 이벤트를
        # 확인하고 즉시 빠져나온다. 새 명령 수락 시 해제.
        self._abort_event = threading.Event()

        self._latest_sidecam: Optional[np.ndarray] = None
        self._sidecam_lock = threading.Lock()

        self._esp32_rail_done = False

        # ─── 텔레오퍼레이션 상태 ───
        self._teleop_running = False
        self._leader: Optional[OmxLeader] = None
        self._teleop_offsets = {}

        # ─── 콜백 그룹 (블로킹 서비스/텔레옵과 타이머가 서로를 막지 않도록) ───
        self._cb_group = ReentrantCallbackGroup()

        # ─── lerobot bus I/O 직렬화 락 ───
        # MultiThreadedExecutor 환경에서 10Hz joint 발행 타이머(read)와
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
        self._act_loading = False
        self._act_reload_lock = threading.Lock()
        self._act_models_cache = []
        # 저장된 마지막 선택이 있으면 use_act=false 로 기동해도 복원 로드한다 —
        # 복원 성공 시 자동 ON (대시보드 선택 시 자동 ON 과 동일 정책)
        restored = self._restore_last_act_model()
        if self._use_act or restored:
            if self._load_act_policy() and restored:
                self._use_act = True

        # ─── 발표용 rerun 웹 뷰어 (기본 비활성) ───
        # bounded queue — full이면 조용히 drop, ACT 추론 루프를 막지 않기 위함.
        self._rerun_queue = queue.Queue(maxsize=2)
        if self._rerun_enable:
            self._init_rerun()

        # ─── ROS 통신 ───
        self._setup_ros_interfaces()

        # ─── ACT 모델 목록/현재상태 최초 발행 (대시보드 연동) ───
        self._publish_act_models()
        self._publish_act_current()

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
        self.declare_parameter('act_device', 'cpu')   # 'cuda' or 'cpu'
        # ACT 모델 탐색 루트 (대시보드 선택용). 학습 출력 train 폴더.
        self.declare_parameter('act_models_root',
            '/physical_ai_tools/lerobot/outputs/train')
        # 안전(P1): send_action(ACT·텔레옵) 1스텝 최대 상대이동량(정규화 단위).
        # 값을 낮출수록 폭주 방지 강도가 높다. 검증 후 단계적으로 상향한다.
        self.declare_parameter('act_max_relative_target', 8.0)
        # 발표용 시각화: ACT 추론 실시간 rerun 웹 뷰어 (실패 시 자동 강등)
        self.declare_parameter('rerun_enable', True)
        # 데모 녹화용: 설정 시 웹 뷰어 대신 rrd 파일로 저장 (rerun 0.22는 싱크가 단일이라 동시 불가)
        self.declare_parameter('rerun_save_path', '')
        # 레일 위치 (mm 단위) — 조립 후 캘리브레이션으로 확정
        self.declare_parameter('rail_mm_bed',      381.25)
        self.declare_parameter('rail_mm_inspect', 12.5)
        self.declare_parameter('rail_mm_pass',    25.0)
        self.declare_parameter('rail_mm_fail',    125.0)
        # 카메라
        self.declare_parameter('sidecam_topic', '/camera1/image_raw/compressed')
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
        self._act_device        = self.get_parameter('act_device').value
        self._act_models_root   = self.get_parameter('act_models_root').value
        # ensure_safe_goal_position 은 float/dict 만 허용하므로 반드시 float 로 전달.
        self._act_max_rel_target = float(self.get_parameter('act_max_relative_target').value)
        self._rerun_enable      = self.get_parameter('rerun_enable').value
        self._rerun_save_path   = self.get_parameter('rerun_save_path').value
        self._rail_mm = {
            RailPosition.BED:     self.get_parameter('rail_mm_bed').value,
            RailPosition.INSPECT: self.get_parameter('rail_mm_inspect').value,
            RailPosition.PASS:    self.get_parameter('rail_mm_pass').value,
            RailPosition.FAIL:    self.get_parameter('rail_mm_fail').value,
        }
        self._sidecam_topic  = self.get_parameter('sidecam_topic').value
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
    # 발표용 rerun 웹 뷰어 (ACT 추론 실시간 모니터링)
    # ─────────────────────────────────────────────
    def _init_rerun(self):
        # rerun 미설치 환경에서도 노드가 죽지 않도록 try/except로 감싸고,
        # 실패 시 rerun_enable을 강등해 이후 추론 루프도 완전히 비활성화한다.
        try:
            import rerun as rr
            rr.init('quvi_act', spawn=False)
            if self._rerun_save_path:
                # rerun 0.22는 RecordingStream당 싱크가 하나뿐 — save와 serve_web 동시 불가
                rr.save(self._rerun_save_path)
                self.get_logger().info(
                    f'rerun 저장 모드 — 웹 뷰어 비활성, 경로={self._rerun_save_path}')
            else:
                rr.serve_web(web_port=9090, ws_port=9877, open_browser=False)
                self.get_logger().info('rerun 웹 뷰어 시작 — http://<host>:9090')
            self._rerun = rr
            threading.Thread(
                target=self._rerun_log_worker, daemon=True).start()
        except Exception as e:
            self.get_logger().warn(f'rerun 초기화 실패 — 비활성화: {e}')
            self._rerun_enable = False

    def _rerun_log_worker(self):
        # 큐에서 (프레임, 관절상태, 액션청크)를 받아 rr.log() 수행.
        # 인코딩/로깅 비용을 ACT 추론 루프에서 분리하기 위한 전담 데몬 스레드.
        rr = self._rerun
        while True:
            frame, joint_vals, action_chunk = self._rerun_queue.get()
            try:
                # frame은 BGR 원본 — rerun은 RGB 해석이라 여기(전담 스레드)서 변환
                rr.log('camera/sidecam', rr.Image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
                for i, v in enumerate(joint_vals):
                    rr.log(f'joints/state/{JOINT_NAMES[i]}', rr.Scalar(float(v)))
                for i, v in enumerate(action_chunk[-1] if len(action_chunk) else []):
                    rr.log(f'joints/action/{JOINT_NAMES[i]}', rr.Scalar(float(v)))
            except Exception as e:
                self.get_logger().warn(f'rerun 로깅 실패(무시): {e}')

    # ─────────────────────────────────────────────
    # ACT 모델 로드
    # ─────────────────────────────────────────────
    def _load_act_policy(self, model_path: str = None) -> bool:
        """LeRobot ACTPolicy 로드. model_path 미지정 시 현재 self._act_model_path 사용.

        성공 시 self._act_policy 교체 + self._act_model_path 갱신 + True 반환.
        실패 시 기존 정책을 유지하고 False 반환.
        """
        try:
            for _lerobot_src in ['/workspace/lerobot/src', '/physical_ai_tools/lerobot/src']:
                if _lerobot_src not in sys.path:
                    sys.path.insert(0, _lerobot_src)
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError as e:
            self.get_logger().error(f'LeRobot/torch 미설치: {e}')
            return False

        target = model_path if model_path else self._act_model_path
        resolved_path = Path(target)
        if not resolved_path.is_absolute():
            resolved_path = Path('/workspace') / resolved_path
        resolved_path = resolved_path.resolve()

        self.get_logger().info(f'ACT 모델 로드 중: {resolved_path}')
        try:
            if not resolved_path.exists():
                raise FileNotFoundError(f'로컬 모델 디렉토리가 존재하지 않습니다: {resolved_path}')
            policy = ACTPolicy.from_pretrained(str(resolved_path))
            policy.eval()
            device = self._act_device
            policy = policy.to(device)
            # 성공 후 원자적으로 교체
            self._act_policy = policy
            self._act_device_obj = device
            self._act_model_path = str(resolved_path)
            self._act_ready = True
            self._save_last_act_model(str(resolved_path))
            self.get_logger().info(f'ACT 모델 로드 완료: {resolved_path} (device={device})')
            return True
        except Exception as e:
            self.get_logger().error(f'ACT 모델 로드 실패: {e}')
            return False

    def _restore_last_act_model(self) -> bool:
        """직전 세션에서 선택한 모델 경로 복원. 성공 시 True, 파일 없거나 경로 소실 시 False."""
        try:
            saved = json.loads(Path(ACT_LAST_MODEL_FILE).read_text())['path']
            if Path(saved).is_dir():
                self._act_model_path = saved
                self.get_logger().info(f'마지막 선택 ACT 모델 복원: {saved}')
                return True
            self.get_logger().warn(f'저장된 ACT 모델 경로 소실 — 기본값 사용: {saved}')
        except (OSError, KeyError, ValueError):
            pass  # 최초 기동 등 파일 없음 — 기본값 사용
        return False

    def _save_last_act_model(self, path: str):
        """로드 성공한 모델 경로를 저장 (재기동 시 복원용). 실패해도 로드는 유효."""
        try:
            Path(ACT_LAST_MODEL_FILE).write_text(
                json.dumps({'path': path}, ensure_ascii=False))
        except OSError as e:
            self.get_logger().warn(f'ACT 모델 경로 저장 실패: {e}')

    # ─────────────────────────────────────────────
    # ACT 모델 스캔 / 런타임 선택 (HMI 대시보드 연동)
    # ─────────────────────────────────────────────
    def _scan_act_models(self) -> list:
        """학습 출력 폴더에서 호환 가능한 pretrained_model 을 탐색한다.

        구조: <root>/<run_name>/checkpoints/<step>/pretrained_model/config.json
        각 run 은 가장 최신(step 최대) 체크포인트를 대표로 사용한다.
        호환 조건: input 에 observation.images.camera1 + observation.state,
                   output action shape == [6].
        반환: [{'name','path','step'}] (name 오름차순)
        """
        roots = []
        if self._act_models_root:
            roots.append(Path(self._act_models_root))
        # act_model_path 로부터 train 루트 유추 (.../train/<run>/checkpoints/<step>/pretrained_model)
        try:
            p = Path(self._act_model_path)
            if 'checkpoints' in p.parts:
                idx = p.parts.index('checkpoints')
                roots.append(Path(*p.parts[:idx - 1]))  # train 루트
        except Exception:
            pass

        seen_roots = set()
        models = []
        for root in roots:
            root = root.resolve()
            if root in seen_roots or not root.is_dir():
                continue
            seen_roots.add(root)
            for run_dir in sorted(root.iterdir()):
                ckpt_root = run_dir / 'checkpoints'
                if not ckpt_root.is_dir():
                    continue
                # step 최대(최신) 체크포인트 선택
                steps = [d for d in ckpt_root.iterdir()
                         if d.is_dir() and (d / 'pretrained_model' / 'config.json').is_file()]
                if not steps:
                    continue
                def _step_key(d):
                    try:
                        return int(d.name)
                    except ValueError:
                        return -1
                latest = max(steps, key=_step_key)
                pm = latest / 'pretrained_model'
                try:
                    cfg = json.load(open(pm / 'config.json'))
                    inp = cfg.get('input_features', {})
                    act = cfg.get('output_features', {}).get('action', {}).get('shape')
                    compatible = ('observation.images.camera1' in inp
                                  and 'observation.state' in inp and act == [6])
                    if not compatible:
                        self.get_logger().warn(f'ACT 모델 호환 불가(건너뜀): {run_dir.name}')
                        continue
                    models.append({'name': run_dir.name,
                                   'path': str(pm.resolve()),
                                   'step': latest.name})
                except Exception as e:
                    self.get_logger().warn(f'ACT 모델 config 읽기 실패({run_dir.name}): {e}')
        models.sort(key=lambda m: m['name'])
        return models

    def _publish_act_models(self):
        """스캔한 모델 목록을 latched 토픽으로 발행."""
        try:
            models = self._scan_act_models()
        except Exception as e:
            self.get_logger().error(f'ACT 모델 스캔 실패: {e}')
            models = []
        self._act_models_cache = models
        self._act_models_pub.publish(
            String(data=json.dumps(models, ensure_ascii=False)))

    def _publish_act_current(self):
        """현재 모델·로드 상태를 latched 토픽으로 발행."""
        name = ''
        for m in self._act_models_cache:
            if m['path'] == self._act_model_path:
                name = m['name']
                break
        payload = {
            'path': self._act_model_path,
            'name': name,
            'ready': bool(self._act_ready),
            'loading': bool(self._act_loading),
            'use_act': bool(self._use_act),
        }
        self._act_current_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False)))

    def _on_act_model_select(self, msg: String):
        """HMI 에서 모델 선택 수신 → 백그라운드 재로드."""
        path = msg.data.strip()
        if not path:
            return
        t = threading.Thread(target=self._reload_act_policy, args=(path,), daemon=True)
        t.start()

    def _reload_act_policy(self, path: str):
        """런타임 ACT 모델 재로드. IDLE 상태에서만 허용, 중복 로드 방지."""
        if not self._act_reload_lock.acquire(blocking=False):
            self.get_logger().warn('ACT 재로드 이미 진행 중 — 요청 무시')
            return
        try:
            if self._get_state() != RobotState.IDLE:
                self.get_logger().warn(
                    f'ACT 재로드 거부: 로봇이 IDLE 아님(현재 {self._get_state().name})')
                self._publish_status('ACT 모델 변경 거부: 동작 중')
                return
            self._act_loading = True
            self._publish_act_current()
            _parts = Path(path).parts
            _run = _parts[-4] if len(_parts) >= 4 else path   # .../<run>/checkpoints/<step>/pretrained_model
            self._publish_status(f'ACT 모델 로딩 중: {_run}')
            ok = self._load_act_policy(path)
            if ok:
                self._use_act = True   # 선택 시 자동 ON (결정사항)
                self._publish_status('ACT 모델 로드 완료 (ACT 사용 ON)')
            else:
                self._publish_status('ACT 모델 로드 실패')
        finally:
            self._act_loading = False
            self._publish_act_current()
            self._act_reload_lock.release()

    # ─────────────────────────────────────────────
    # ROS 인터페이스 설정
    # ─────────────────────────────────────────────
    def _setup_ros_interfaces(self):
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()

    def _setup_subscribers(self):
        # ── Subscribers ──
        self._sidecam_sub = self.create_subscription(
            CompressedImage, self._sidecam_topic,
            self._sidecam_callback, 10)

        self._grasp_cmd_sub = self.create_subscription(
            GraspGoal, topics.TOPIC_ROBOT_GRASP_CMD,
            self._grasp_cmd_callback, 10)

        self._rail_cmd_sub = self.create_subscription(
            Int32, topics.TOPIC_ROBOT_RAIL_CMD,
            self._rail_cmd_callback, 10)

        self._esp32_rail_done_sub = self.create_subscription(
            Bool, topics.TOPIC_MOTOR_RAIL_DONE,
            self._esp32_rail_done_callback, 10)

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
            Bool, topics.TOPIC_ROBOT_TELEOP_CMD,
            self._teleop_cmd_callback, 10,
            callback_group=self._cb_group)

        self._estop_sub = self.create_subscription(
            Bool, topics.TOPIC_ESTOP,
            self._estop_cmd_callback, 10,
            callback_group=self._cb_group)

        self._reset_sub = self.create_subscription(
            Bool, topics.TOPIC_ROBOT_RESET_CMD,
            self._reset_cmd_callback, 10,
            callback_group=self._cb_group)

        # ACT 모델 선택 수신 (대시보드) → 백그라운드 재로드
        self._act_model_select_sub = self.create_subscription(
            String, topics.TOPIC_ACT_MODEL_SELECT,
            self._on_act_model_select, 10,
            callback_group=self._cb_group)

        # HMI 명령 수신 → STOP/ESTOP 시 진행 중 동작 소프트 중단 (#1)
        self._hmi_cmd_sub = self.create_subscription(
            String, topics.TOPIC_HMI_COMMAND,
            self._hmi_command_callback, 10,
            callback_group=self._cb_group)

    def _setup_publishers(self):
        # ── Publishers ──
        # latched(TRANSIENT_LOCAL): 늦게 붙는 HMI 구독자도 최신 상태를 즉시 받는다.
        from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
        _latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._act_models_pub = self.create_publisher(
            String, topics.TOPIC_ACT_MODELS, _latched)
        self._act_current_pub = self.create_publisher(
            String, topics.TOPIC_ACT_CURRENT, _latched)

        self._joint_state_pub = self.create_publisher(
            JointState, topics.TOPIC_ROBOT_JOINT_STATES, 10)

        self._rail_pub = self.create_publisher(
            Int32, topics.TOPIC_MOTOR_RAIL_CMD, 10)

        self._status_pub = self.create_publisher(
            String, topics.TOPIC_ROBOT_STATUS, 10)

        self._grasp_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_GRASP_DONE, 10)

        self._release_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_RELEASE_DONE, 10)

        self._home_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_HOME_DONE, 10)

        self._rail_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_RAIL_DONE, 10)

        self._place_chamber_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_PLACE_CHAMBER_DONE, 10)

        self._pick_chamber_done_pub = self.create_publisher(
            Bool, topics.TOPIC_ROBOT_PICK_CHAMBER_DONE, 10)

    def _setup_services(self):
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
        frame = decode_compressed(msg)
        if frame is not None:
            with self._sidecam_lock:
                self._latest_sidecam = frame

    def _esp32_rail_done_callback(self, msg: Bool):
        # 레일 이동 중일 때만 done 을 수락한다 (#7). 오케스트레이터가 STARTUP 에서
        # /motor/rail 로 직접 보낸 이동의 done 이 robot_control 의 플래그를 엉뚱하게
        # 세팅하는 것을 방지한다. (그 경우는 오케스트레이터가 직접 처리한다.)
        if msg.data and self._get_state() == RobotState.MOVING_RAIL:
            self._esp32_rail_done = True

    # ─────────────────────────────────────────────
    # 소프트 중단 (STOP) 처리 (#1/#3)
    # ─────────────────────────────────────────────
    def _hmi_command_callback(self, msg: String):
        """HMI 명령 수신. STOP/ESTOP 시 진행 중 동작을 소프트 중단한다.

        ESTOP 은 별도 /system/estop 으로 토크 차단까지 하지만, 여기서도
        중단 이벤트를 세워 동작 스레드가 즉시 빠져나오게 한다.
        """
        cmd = msg.data.strip().upper()
        if cmd in ('STOP', 'ESTOP'):
            self.get_logger().warn(f'HMI {cmd} 수신 — 진행 중 동작 소프트 중단')
            self._abort_event.set()
            threading.Thread(target=self._soft_stop, daemon=True).start()
        elif cmd in ('START', 'RESET'):
            self._abort_event.clear()

    def _should_abort(self) -> bool:
        """진행 중 동작을 중단해야 하는지 (STOP 이벤트 또는 ERROR 상태)."""
        return self._abort_event.is_set() or self._get_state() == RobotState.ERROR

    def _soft_stop(self):
        """진행 중 이동을 현재 위치로 즉시 정지 (토크 유지). STOP 소프트 정지용.

        ESTOP(_safe_estop_cleanup)로 토크가 이미 꺼졌다면 no-op.
        """
        if not self._use_real_hardware or not self._dxl_ready:
            return
        positions = self._read_raw_positions()
        if not positions:
            return
        arm = {k: v for k, v in positions.items() if k != 'gripper'}
        if not arm:
            return
        try:
            with self._dxl_io_lock:
                self._follower.bus.sync_write('Goal_Position', arm, normalize=False)
            self.get_logger().info('소프트 정지: 현재 위치로 목표 고정')
        except Exception as e:
            self.get_logger().warn(f'소프트 정지 실패: {e}')

    def _act_check_abort(self, estop_msg: str) -> bool:
        """ACT 파지 루프 내 ESTOP/소프트 STOP 중단 여부 확인 (청크·스텝 루프 공용)."""
        if self._get_state() == RobotState.ERROR:
            self.get_logger().error(estop_msg)
            self._safe_estop_cleanup()
            return True
        if self._abort_event.is_set():   # 소프트 STOP (#1)
            self.get_logger().warn('STOP 감지 — ACT 파지 중단 (토크 유지)')
            self._soft_stop()
            self._set_state(RobotState.IDLE)
            return True
        return False

    # ─────────────────────────────────────────────
    # 명령 콜백 (토픽 기반 — 비동기)
    # ─────────────────────────────────────────────
    def _try_acquire_state(self, target_state: RobotState, allow_error: bool = False) -> bool:
        """lock 안에서 IDLE(옵션: ERROR 포함) 선점. 성공 시 상태 세트 + _cmd_gen 증가."""
        with self._state_lock:
            allowed = (RobotState.IDLE, RobotState.ERROR) if allow_error else (RobotState.IDLE,)
            if self._state not in allowed:
                self.get_logger().warn(f'명령 무시: 현재 {self._state.name} 동작 중')
                return False
            self._state = target_state
            self._cmd_gen += 1
            self._publish_status(self._state.name)
        self._abort_event.clear()   # 새 명령 수락 → 이전 STOP 중단 해제
        return True

    def _try_start_command(self, target_state: RobotState, target_func, *args):
        """lock 안에서 상태를 선점하고 스레드를 시작하여 레이스 컨디션을 방지한다."""
        if not self._try_acquire_state(target_state):
            return False
        t = threading.Thread(target=target_func, args=args, daemon=True)
        t.start()
        return True

    def _grasp_cmd_callback(self, msg: GraspGoal):
        self.get_logger().info(
            f'파지 목표 수신: idx={msg.object_index} (ACT visuomotor 추론)')
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
            self.get_logger().error('비상 정지 명령 수신! 동작 강제 중단 및 에러 상태 전환')
            self._abort_event.set()   # _should_abort가 보는 이벤트 — state==ERROR만으로 커버 안 되는 가드 보강 (#7)
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
        if not self._try_acquire_state(RobotState.ACT_GRASPING):
            response.success = False
            response.message = f'현재 {self._state.name} 동작 중 — 명령 거부'
            return response
        success = self._execute_act_grasp()
        response.success = success
        response.message = 'ACT 파지 완료' if success else 'ACT 파지 실패'
        return response

    def _go_home_service(self, request, response):
        # 홈 복귀는 복구 수단 — act_loop.sh HOME_BETWEEN=1 이 파지 실패(ERROR) 후 호출하므로
        # ERROR 상태에서도 허용 (수동 복구 경로도 동일)
        if not self._try_acquire_state(RobotState.HOMING, allow_error=True):
            response.success = False
            response.message = f'현재 {self._state.name} 동작 중 — 명령 거부'
            return response
        success = self._execute_home()
        response.success = success
        response.message = '홈 복귀 완료' if success else '홈 복귀 실패'
        return response

    def _open_gripper_service(self, request, response):
        if self._state not in (RobotState.IDLE, RobotState.ERROR):
            response.success = False
            response.message = f'현재 {self._state.name} 동작 중 — 그리퍼 명령 거부'
            return response
        self._write_raw_position({'gripper': GRIPPER_OPEN})
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 열기 완료 (OmxFollower ID16)'
        return response

    def _close_gripper_service(self, request, response):
        # 짧은 sync_write라 상태 선점(_try_acquire_state)까지는 하지 않고 가드만 둔다
        if self._state not in (RobotState.IDLE, RobotState.ERROR):
            response.success = False
            response.message = f'현재 {self._state.name} 동작 중 — 그리퍼 명령 거부'
            return response
        self._write_raw_position({'gripper': GRIPPER_CLOSE})
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 닫기 완료 (OmxFollower ID16)'
        return response

    # ─────────────────────────────────────────────
    # 실행 함수 — ACT 파지
    # ─────────────────────────────────────────────
    def _execute_rule_based_grasp(self) -> bool:
        """ACT 미사용/미로드 시 룰베이스(티칭 기반) 파지."""
        gen = self._cmd_gen
        self.get_logger().info('ACT 미사용 또는 로드 안됨 — 룰베이스(티칭 기반) 파지를 실행합니다.')
        self._publish_status('룰베이스 파지 시작')

        try:
            # 사용자가 수동으로 물려준 상태이므로, 바로 POSE_P1(베드 위 대기)로 이동합니다.
            self.get_logger().info('1. POSE_P1 위치로 즉시 이동 (그리퍼는 닫힘 상태 유지)')

            # POSE_P1 자세로 이동하되, 그리퍼는 사용자가 물려준 물체를 쥐도록 GRIPPER_CLOSE(1800)로 덮어씌웁니다.
            p1_pose = dict(POSE_P1, gripper=GRIPPER_CLOSE)

            self._write_raw_position(p1_pose)
            # 그리퍼는 물체를 쥐어 목표(GRIPPER_CLOSE)에 도달 못 하므로 대기 대상에서 제외 — 안 하면 매 회 10s 풀타임아웃.
            self._wait_motion_done(_arm_only(p1_pose))

            self._grasp_done_pub.publish(Bool(data=True))

            self._set_state_if_current(RobotState.IDLE, gen)
            self._publish_status('룰베이스 파지 완료')
            return True
        except Exception as e:
            self.get_logger().error(f'룰베이스 파지 중 오류: {e}')
            self._set_state_if_current(RobotState.ERROR, gen)
            self._publish_status(f'ERROR: {e}')
            return False

    def _act_image_tensor(self, frame):
        """사이드캠 BGR 프레임 → ACT 정책 입력용 정규화 이미지 텐서."""
        import torch
        frame_rgb = cv2.cvtColor(
            cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(
            frame_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self._act_device_obj)
        return img_tensor

    def _act_send_action_step(self, action) -> None:
        """ACT 정책 출력 1스텝을 실HW/SIM 으로 전송 (raw 변환·상대이동 캡·클램프 포함)."""
        if self._use_real_hardware and self._dxl_ready:
            # 정책 출력은 라디안 — raw = 2048 + rad·4096/2π 로 변환해
            # 직접 전송한다. send_action 정규화(±100/DEGREES) 경로는
            # 학습 단위와 불일치해 2·3축이 raw≈2048(뻗은 자세)로 발산했다.
            goal_raw = {
                name: int(round(rad_to_raw(float(action[j]))))
                for j, name in enumerate(JOINT_NAMES)}
            with self._dxl_io_lock:
                present = self._follower.bus.sync_read('Present_Position', normalize=False)
                # send_action 의 max_relative_target(정규화 단위) 대체 —
                # 200 norm = 4096 틱으로 환산해 1스텝 상대이동을 캡.
                cap = max(1, int(self._act_max_rel_target * 4096.0 / 200.0))
                for name in goal_raw:
                    p = present.get(name, goal_raw[name])
                    goal_raw[name] = int(max(p - cap, min(p + cap, goal_raw[name])))
                goal_raw = {n: int(max(0, min(4095, v))) for n, v in goal_raw.items()}
                goal_raw = self._clip_safe_targets(goal_raw, is_raw=True)
                self._follower.bus.sync_write('Goal_Position', goal_raw, normalize=False)
        else:
            goal_dict = {
                name: int(np.clip(rad_to_raw(float(action[j])), 0, DXL_RAW_MAX))
                for j, name in enumerate(JOINT_NAMES)}
            self._write_raw_position(goal_dict)

    def _execute_act_grasp(self) -> bool:
        gen = self._cmd_gen
        self._set_state_if_current(RobotState.ACT_GRASPING, gen)

        # ACT를 사용하지 않거나 로드가 안된 경우 룰베이스(티칭 기반) 파지 수행
        if not self._use_act or not self._act_ready:
            return self._execute_rule_based_grasp()

        try:
            import torch
            # P2: 이전 파지 에피소드의 낡은 액션 큐를 비운다. reset() 없이는 두 번째
            # 파지부터 직전 관측 기반의 남은 액션이 먼저 실행돼 예기치 않게 움직인다.
            self._act_policy.reset()
            # P3(C6): ACT용 프로파일을 명시적으로 설정해 직전 동작(시퀀스=2000ms 등)의
            # 프로파일 누수를 제거한다. 학습 녹화 시 configure() 기본값(50ms)이 적용됐으므로
            # 동일하게 맞춰 학습 동역학과 일치시킨다.
            self._apply_motor_profile(JOINT_NAMES, PROFILE_VELOCITY_ACT, PROFILE_ACCEL_ACT)
            ACT_GRASP_DURATION = 7.0
            dt = 1.0 / ACT_CONTROL_HZ
            start = time.time()
            chunk_count = 0

            while (time.time() - start) < ACT_GRASP_DURATION:
                if self._act_check_abort('ESTOP 감지 — ACT 청크 실행을 중단합니다.'):
                    return False

                with self._sidecam_lock:
                    frame = self._latest_sidecam

                if frame is None:
                    self.get_logger().error('사이드캠 이미지 없음 — 파지 불가')
                    self._set_state_if_current(RobotState.IDLE, gen)
                    return False

                img_tensor = self._act_image_tensor(frame)

                acquired = self._dxl_io_lock.acquire(blocking=False)
                if not acquired:
                    continue
                try:
                    raw_positions = self._follower.bus.sync_read('Present_Position', normalize=False)
                except Exception as e:
                    self.get_logger().error(f'관절 위치 읽기 실패 — 통신 오류: {e}. 파지를 중단합니다.')
                    self._set_state_if_current(RobotState.ERROR, gen)
                    self._publish_status('ERROR: Dynamixel 통신 오류')
                    return False
                finally:
                    self._dxl_io_lock.release()

                # 학습 데이터셋(robot_type=omx_f, ROS 스택 녹화)의 state/action 은
                # 라디안 단위 — lerobot 정규화(±100/DEGREES)가 아니다. 관측을
                # rad = (raw-2048)·2π/4096 으로 변환해 학습 분포와 일치시킨다.
                joint_vals = [raw_to_rad(raw_positions[name])
                              for name in JOINT_NAMES]
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

                if self._rerun_enable:
                    try:
                        self._rerun_queue.put_nowait((frame, joint_vals, action_chunk))
                    except queue.Full:
                        pass

                for i, action in enumerate(action_chunk):
                    if (time.time() - start) >= ACT_GRASP_DURATION:
                        break
                    if self._act_check_abort(f'ESTOP 감지 — ACT 실행 중단 (청크#{chunk_count} 스텝 {i})'):
                        return False

                    step_start = time.time()
                    self._act_send_action_step(action)

                    elapsed = time.time() - step_start
                    remaining = dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            total_time = time.time() - start
            self.get_logger().info(f'ACT 파지 완료 | 총 소요={total_time:.2f}s | 청크 수={chunk_count}')

            self._grasp_done_pub.publish(Bool(data=True))

            self._set_state_if_current(RobotState.IDLE, gen)
            self._publish_status('ACT 파지 완료')
            return True

        except Exception as e:
            self.get_logger().error(f'ACT 파지 중 오류: {e}')
            if self._follower and hasattr(self._follower, 'bus') and hasattr(self._follower.bus, 'port_handler'):
                self._follower.bus.port_handler.is_using = False
            self._set_state_if_current(RobotState.ERROR, gen)
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
        gen = self._cmd_gen
        self._set_state_if_current(RobotState.MOVING_RAIL, gen)
        target_mm = float(self._rail_mm[position])
        pos_name = position.name
        self._publish_status(f'레일 이동: {pos_name} ({target_mm:.2f}mm)')
        self.get_logger().info(f'레일 이동 명령: {pos_name} = {target_mm:.2f}mm')

        if self._use_real_hardware:
            self._esp32_rail_done = False

        self._rail_pub.publish(Int32(data=int(target_mm * topics.RAIL_STEPS_PER_MM)))

        if self._use_real_hardware:
            deadline = time.time() + self._rail_timeout
            success = False
            while time.time() < deadline:
                if self._should_abort():
                    self.get_logger().warn('레일 이동 중단 감지 (STOP/ESTOP) — done 발행 없이 종료')
                    self._set_state_if_current(RobotState.IDLE, gen)
                    self._publish_status('레일 이동 중단')
                    return False
                if self._esp32_rail_done:
                    self._esp32_rail_done = False
                    success = True
                    break
                time.sleep(0.05)

            if not success:
                self.get_logger().error(f'레일 이동 타임아웃! ({self._rail_timeout}초)')
                self._set_state_if_current(RobotState.ERROR, gen)
                self._publish_status('ERROR: 레일 이동 타임아웃')
                return False
        else:
            time.sleep(1.0)

        self._rail_done_pub.publish(Bool(data=True))

        self._set_state_if_current(RobotState.IDLE, gen)
        self._publish_status(f'레일 이동 완료: {pos_name}')
        return True

    # ─────────────────────────────────────────────
    # 실행 함수 — 투하
    # ─────────────────────────────────────────────
    def _execute_release(self) -> bool:
        gen = self._cmd_gen
        self._set_state_if_current(RobotState.RELEASING, gen)
        self._publish_status('웨이포인트 시퀀스 시작 (P1~P6)')
        self.get_logger().info('웨이포인트 시퀀스 시작')

        success = self._execute_taught_sequence()

        self._release_done_pub.publish(Bool(data=success))

        self._set_state_if_current(RobotState.IDLE, gen)
        self._publish_status('웨이포인트 시퀀스 완료' if success else '웨이포인트 시퀀스 실패')
        return success

    # ─────────────────────────────────────────────
    # 실행 함수 — 홈 복귀
    # ─────────────────────────────────────────────
    def _execute_home(self) -> bool:
        gen = self._cmd_gen
        self._set_state_if_current(RobotState.HOMING, gen)
        self._publish_status('홈 복귀')
        self.get_logger().info('홈 복귀 시작')

        # 수동 배치·정지 후 토크가 꺼져 있을 수 있어 홈 이동 전 전 관절 토크 인가
        # (2026-07-10: START 시 토크 오프 상태면 홈 이동이 무시돼 ACT 파지 실패)
        if self._use_real_hardware and self._dxl_ready:
            try:
                with self._dxl_io_lock:
                    self._follower.bus.sync_write(
                        'Torque_Enable', {n: 1 for n in JOINT_NAMES}, normalize=False)
            except Exception as e:
                self.get_logger().warn(f'홈 복귀 전 토크 인가 실패: {e}')

        # 그리퍼(전류기반)는 _wait_motion_done 대상에서 제외 — 개방 목표에 못
        # 도달하면 10s 풀타임아웃으로 home done 발행이 늦어 베드 이동이 지연된다
        success = self._write_raw_position(POSE_HOME)
        self._wait_motion_done(_arm_only(POSE_HOME))
        self._wait_gripper()
        success = success and not self._should_abort()   # abort 시 실패로 보고 (#5)

        self._home_done_pub.publish(Bool(data=success))

        self._set_state_if_current(RobotState.IDLE, gen)
        self._publish_status('홈 복귀 완료')
        return success

    def _execute_place_in_chamber(self) -> bool:
        gen = self._cmd_gen
        self._set_state_if_current(RobotState.RELEASING, gen)
        self._publish_status('검사장 안착 시퀀스 시작')
        self.get_logger().info('검사장 안착 시퀀스 시작')

        # 물체 파지 중 이송 — 낙하/파손 방지 저속(SEQ) 프로파일
        self._write_raw_position(_arm_only(POSE_P1), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P1))

        time.sleep(2.0)  # 대기

        self._write_raw_position(_arm_only(POSE_P2), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P2))

        self._write_raw_position(_arm_only(POSE_P3), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P3))

        self._write_raw_position(_arm_only(POSE_P4), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P4))

        self._write_raw_position({'gripper': GRIPPER_OPEN})
        self._wait_gripper()

        # 검사 동안 팔은 P3에서 후퇴 대기
        success = self._write_raw_position(_arm_only(POSE_P3), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P3))
        success = success and not self._should_abort()   # abort 시 실패로 보고 (#5)

        self._place_chamber_done_pub.publish(Bool(data=success))

        self._set_state_if_current(RobotState.IDLE, gen)
        self._publish_status('검사장 안착 완료')
        return success

    def _execute_pick_from_chamber(self) -> bool:
        gen = self._cmd_gen
        self._set_state_if_current(RobotState.ACT_GRASPING, gen)
        self._publish_status('검사장 재파지 시퀀스 시작')
        self.get_logger().info('검사장 재파지 시퀀스 시작')

        # P3에서 완전 개방 후 팔 관절만 P4로 접근
        self._write_raw_position({'gripper': GRIPPER_OPEN})
        self._wait_gripper()

        # 물체 파지 중 이송 — 낙하/파손 방지 저속(SEQ) 프로파일
        self._write_raw_position(_arm_only(POSE_P4), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P4))

        self._write_raw_position({'gripper': GRIPPER_CLOSE})
        self._wait_gripper()

        self._write_raw_position(_arm_only(POSE_P3), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P3))

        success = self._write_raw_position(_arm_only(POSE_P5), velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ)
        self._wait_motion_done(_arm_only(POSE_P5))  # P5가 회전 포함 — 복귀 완료 확인 후 done 발행
        success = success and not self._should_abort()   # abort 시 실패로 보고 (#5)

        self._pick_chamber_done_pub.publish(Bool(data=success))

        self._set_state_if_current(RobotState.IDLE, gen)
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
    def _wait_motion_done(self, goal: dict, timeout: float = 10.0, tol: int = 40):
        """Present_Position 폴링으로 목표 위치 도달을 확인.

        goal: _write_raw_position에 넘긴 것과 동일한 {joint: raw_value} 딕셔너리.
        tol:  허용 오차 (raw 단위, 40 ≈ 3.5°). 경유 자세는 정밀 도달이 불필요하고,
              부하 자세에서 I게인 0으로 정상상태 오차가 20틱을 넘을 수 있다.

        주의: 그리퍼(전류기반 위치제어)는 물체를 쥐면 목표에 도달하지 못하므로
        이 함수로 대기하면 안 된다. 그리퍼는 _wait_gripper() 를 사용할 것 (#2).
        """
        if not self._use_real_hardware or not self._dxl_ready:
            return
        time.sleep(0.05)  # Goal_Position 반영 여유
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._should_abort():   # STOP/ESTOP (#1/#3)
                self._soft_stop()
                return
            positions = self._read_raw_positions()
            if positions is None:
                time.sleep(0.05)
                continue
            if all(abs(positions.get(j, 0) - v) <= tol for j, v in goal.items()):
                time.sleep(0.1)  # 기계 안정화
                return
            time.sleep(0.05)
        # 어긋난 관절 진단 — 어느 축이 수렴하지 못했는지 특정
        positions = self._read_raw_positions() or {}
        off = {j: int(positions.get(j, 0) - v) for j, v in goal.items()
               if abs(positions.get(j, 0) - v) > tol}
        self.get_logger().warn(
            f'[_wait_motion_done] {timeout}s 타임아웃 — 강제 진행 | 오차 초과 관절(현재-목표): {off}')

    def _wait_gripper(self, settle: float = GRIPPER_SETTLE_SEC):
        """그리퍼 개폐 완료 대기 — 고정 지연 (#2).

        그리퍼는 전류기반 위치제어라 물체를 쥐면 목표 위치에 도달하지 못한다.
        따라서 위치 수렴 폴링(_wait_motion_done) 대신 고정 시간만 대기한다.
        STOP/ESTOP 중단에 반응하도록 잘게 나눠 대기한다.
        """
        end = time.time() + settle
        while time.time() < end:
            if self._should_abort():
                return
            time.sleep(0.05)

    # ─────────────────────────────────────────────
    # lerobot bus 기반 모터 I/O
    # ─────────────────────────────────────────────
    def _write_raw_position(self, positions: dict,
                            velocity: int = PROFILE_VELOCITY,
                            accel: int = PROFILE_ACCEL,
                            grip_velocity: int = PROFILE_VELOCITY_GRIP,
                            grip_accel: int = PROFILE_ACCEL_GRIP) -> bool:
        # STOP/ESTOP 중에는 새 목표를 쓰지 않는다 (#1/#3). 홈/안착/재파지 등
        # 모든 선형 시퀀스가 이 경로를 쓰므로 여기서 일괄 차단한다.
        # (_soft_stop 은 bus.sync_write 를 직접 호출하므로 이 가드의 영향을 받지 않는다.)
        if self._abort_event.is_set():
            return False
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
    # 관절 상태 발행 (10 Hz 타이머)
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
            (raw_positions[name] / DXL_TICKS_PER_REV) * 2 * math.pi
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

    def _set_state_if_current(self, state: RobotState, gen: int):
        """자신이 최신 명령(gen)일 때만 상태를 쓴다 — RESET/ESTOP 후 잔존 스레드의 덮어쓰기 방지.
        # ponytail: gen 은 실행 함수 진입 시점 스냅샷 — 스폰~진입 사이 RESET 이 끼는 마이크로초 창은 기존과 동일 위험으로 수용"""
        if gen != self._cmd_gen:
            self.get_logger().warn(f'잔존 스레드의 상태 변경 무시: {state.name} (gen {gen} != {self._cmd_gen})')
            return
        self._set_state(state)

    def _publish_status(self, msg: str):
        self._status_pub.publish(String(data=f'[ROBOT] {msg}'))

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
        self._teleop_running = False   # 텔레옵 루프 즉시 종료 (ESTOP 중 send_action 지속 방지) (#1)
        try:
            if self._dxl_ready and self._follower:
                if hasattr(self._follower, 'bus'):
                    torque_off = {name: 0 for name in JOINT_NAMES}
                    with self._dxl_io_lock:
                        self._follower.bus.sync_write('Torque_Enable', torque_off, normalize=False)
                self._dxl_ready = False
        except Exception as e:
            self.get_logger().error(f'ESTOP cleanup 중 오류: {e}')

        self._cmd_gen += 1
        self._set_state(RobotState.ERROR)
        self._publish_status('ERROR: ESTOP ACTIVE — 토크 해제됨')

    def _execute_reset(self) -> bool:
        self._cmd_gen += 1
        self._abort_event.clear()   # 리셋 시 이전 STOP 중단 해제 (#1)
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

        순서: P1(경유) → P6(분류장) → 그리퍼 열기
        (P1→P4 안착/재파지 구간은 _execute_place_in_chamber /
        _execute_pick_from_chamber 로 이전돼 중복 제거)
        """
        def move_arm(pose: dict, label: str) -> bool:
            if self._should_abort():   # STOP/ESTOP 시 새 목표 발행 안 함 (#1/#3)
                self.get_logger().warn(f'시퀀스 중단 감지 — {label} 생략')
                return False
            self.get_logger().info(f'이동: {label}')
            arm_only = _arm_only(pose)
            success = self._write_raw_position(
                arm_only,
                velocity=PROFILE_VELOCITY_SEQ,
                accel=PROFILE_ACCEL_SEQ,
            )
            self._wait_motion_done(arm_only)
            return success and not self._should_abort()

        def grip_open():
            if self._should_abort():
                return
            self.get_logger().info('그리퍼 열기')
            self._write_raw_position(
                {'gripper': GRIPPER_OPEN},
                grip_velocity=PROFILE_VELOCITY_GRIP,
                grip_accel=PROFILE_ACCEL_GRIP,
            )
            self._wait_gripper()

        # P1 경유 → P6 분류장 → 그리퍼 열기
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
        with self._state_lock:
            if self._state == RobotState.TELEOPING:
                return True
            if self._state != RobotState.IDLE:
                self.get_logger().warn('텔레옵 무시: 현재 로봇이 IDLE 상태가 아님')
                return False
            self._state = RobotState.TELEOPING   # 체크와 전환을 락 안에서 원자적으로 (동시 진입 선점 방지)

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

        # 직전 동작(홈 1200ms·시퀀스 2000ms)의 프로파일 누수 제거 — 남아 있으면
        # 50Hz 목표를 모터가 1.2~2초에 걸쳐 추종해 리더를 뒤늦게 따라온다.
        # ACT 경로(P3 C6)와 동일하게 텔레옵도 50ms 프로파일을 명시 설정한다.
        self._apply_motor_profile(JOINT_NAMES, PROFILE_VELOCITY_ACT, PROFILE_ACCEL_ACT)

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
        # ESTOP이 _teleop_running만 끈 경우에도 leader가 남아 있으면 포트 해제를 보장한다.
        if not self._teleop_running and self._leader is None:
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

        while self._teleop_running and rclpy.ok():
            if self._should_abort():   # STOP/ESTOP → 루프 종료 (RESET 후 급작동 방지) (#1)
                self.get_logger().warn('텔레옵 중단 감지 (STOP/ESTOP) — 루프 종료')
                self._teleop_running = False
                break
            start_time = time.time()

            if self._use_real_hardware and self._leader and self._follower:
                try:
                    with self._dxl_io_lock:
                        action = self._leader.get_action()
                    
                    if self._teleop_offsets:
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
