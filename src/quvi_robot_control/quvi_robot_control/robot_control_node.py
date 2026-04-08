"""
QUVI ROBOT_CONTROL_NODE
────────────────────────────────────────────────────────────────
로봇팔(OMX Dynamixel XL330) + 리니어 레일 + 턴테이블을
통합 제어하는 노드.

주요 기능:
  1. ACT 모방학습 파지 (LeRobot ACTPolicy)
     - /camera/handcam 이미지 + 관절 상태 → ACT 추론 → 관절 목표값 전송
  2. OMX Dynamixel XL330 관절 제어
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
    BED    = 0   # X=D  3D 프린터 베드
    INSPECT = 1  # X=A  검사장
    PASS   = 2   # X=B  PASS 분류함
    FAIL   = 3   # X=C  FAIL 분류함


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
    ERROR         = 99


# Dynamixel XL330 기본 상수 (Position Mode)
DXL_BAUDRATE            = 1_000_000
DXL_PROTOCOL            = 2.0
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
ADDR_OPERATING_MODE     = 11
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4
TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0
POSITION_MODE           = 3

# 관절 ID 매핑 (OMX AI Manipulator 5DOF)
# ID1=베이스, ID2=숄더, ID3=엘보우, ID4=리스트, ID5=그리퍼
JOINT_IDS               = [1, 2, 3, 4, 5]
JOINT_NAMES             = ['base', 'shoulder', 'elbow', 'wrist', 'gripper']

# 주요 자세 (Dynamixel 위치값, 0~4095 = 0~360°)
# 값은 조립 후 실측 캘리브레이션으로 보정 필요
POSE_HOME    = [2048, 1800, 1200, 2048, 2048]  # 홈 (직립)
POSE_FRONT   = [2048, 1400,  900, 1800, 2300]  # 베드 파지 준비 (앞 방향)
POSE_BACK    = [4096 - 2048, 1400, 900, 1800, 2300]  # 검사/분류 (180° 회전)
POSE_PLACE   = [4096 - 2048, 1600, 1100, 2048, 2300]  # 턴테이블 안착
GRIPPER_OPEN  = 2300  # 그리퍼 열림
GRIPPER_CLOSE = 1800  # 그리퍼 닫힘

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
        self._latest_joint_pos: List[int] = [2048] * 5   # Dynamixel 위치값
        self._handcam_lock = threading.Lock()
        self._bridge = CvBridge()

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
            1.0 / ACT_CONTROL_HZ, self._publish_joint_states)

        self.get_logger().info(
            f'ROBOT_CONTROL_NODE 초기화 완료 | '
            f'하드웨어={self._use_real_hardware} | '
            f'ACT={self._use_act} | '
            f'DXL포트={self._dxl_port_name}')

    # ─────────────────────────────────────────────
    # 파라미터
    # ─────────────────────────────────────────────
    def _declare_params(self):
        # 하드웨어
        self.declare_parameter('use_real_hardware', True)
        self.declare_parameter('dxl_port', '/dev/ttyACM0')
        self.declare_parameter('dxl_baudrate', DXL_BAUDRATE)
        # ACT
        self.declare_parameter('use_act', True)
        self.declare_parameter('act_model_path',
            'outputs/train/quvi_act/checkpoints/last/pretrained_model')
        self.declare_parameter('act_chunk_size', 20)
        self.declare_parameter('act_device', 'cpu')   # 'cuda' or 'cpu'
        # 레일 위치 (스텝 수) — 조립 후 캘리브레이션으로 확정
        self.declare_parameter('rail_steps_bed',     0)
        self.declare_parameter('rail_steps_inspect', 1000)
        self.declare_parameter('rail_steps_pass',    1700)
        self.declare_parameter('rail_steps_fail',    2400)
        # 카메라
        self.declare_parameter('handcam_topic', '/camera/handcam/compressed')
        self.declare_parameter('use_compressed', True)
        # 동작 타임아웃 (초)
        self.declare_parameter('rail_move_timeout_sec', 30.0)
        self.declare_parameter('grasp_timeout_sec', 20.0)
        self.declare_parameter('home_timeout_sec', 10.0)

    def _load_params(self):
        self._use_real_hardware  = self.get_parameter('use_real_hardware').value
        self._dxl_port_name      = self.get_parameter('dxl_port').value
        self._dxl_baudrate       = self.get_parameter('dxl_baudrate').value
        self._use_act            = self.get_parameter('use_act').value
        self._act_model_path     = self.get_parameter('act_model_path').value
        self._act_chunk_size     = self.get_parameter('act_chunk_size').value
        self._act_device         = self.get_parameter('act_device').value
        self._rail_steps = {
            RailPosition.BED:     self.get_parameter('rail_steps_bed').value,
            RailPosition.INSPECT: self.get_parameter('rail_steps_inspect').value,
            RailPosition.PASS:    self.get_parameter('rail_steps_pass').value,
            RailPosition.FAIL:    self.get_parameter('rail_steps_fail').value,
        }
        self._handcam_topic      = self.get_parameter('handcam_topic').value
        self._use_compressed     = self.get_parameter('use_compressed').value
        self._rail_timeout       = self.get_parameter('rail_move_timeout_sec').value
        self._grasp_timeout      = self.get_parameter('grasp_timeout_sec').value
        self._home_timeout       = self.get_parameter('home_timeout_sec').value

    # ─────────────────────────────────────────────
    # Dynamixel 초기화
    # ─────────────────────────────────────────────
    def _init_dynamixel(self):
        """dynamixel_sdk으로 포트 열고 XL330 5개 토크 활성화."""
        try:
            from dynamixel_sdk import (
                PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
                COMM_SUCCESS
            )
        except ImportError:
            self.get_logger().error(
                'dynamixel_sdk 미설치. pip install dynamixel-sdk')
            return

        self._port_handler   = PortHandler(self._dxl_port_name)
        self._packet_handler = PacketHandler(DXL_PROTOCOL)

        if not self._port_handler.openPort():
            self.get_logger().error(f'Dynamixel 포트 열기 실패: {self._dxl_port_name}')
            return

        if not self._port_handler.setBaudRate(self._dxl_baudrate):
            self.get_logger().error('Dynamixel 보드레이트 설정 실패')
            return

        # 각 관절 Position Mode 설정 + 토크 활성화
        for dxl_id in JOINT_IDS:
            # Operating Mode = Position
            self._packet_handler.write1ByteTxRx(
                self._port_handler, dxl_id, ADDR_OPERATING_MODE, POSITION_MODE)
            # Torque Enable
            result, error = self._packet_handler.write1ByteTxRx(
                self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if result != 0:
                self.get_logger().warn(f'ID{dxl_id} 토크 활성화 실패 (result={result})')
            else:
                self.get_logger().info(f'ID{dxl_id} 토크 활성화 완료')

        # GroupSyncWrite (4바이트 Goal Position)
        from dynamixel_sdk import GroupSyncWrite
        self._sync_write = GroupSyncWrite(
            self._port_handler, self._packet_handler,
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        self._dxl_ready = True
        self.get_logger().info('Dynamixel XL330 초기화 완료')

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
            self.get_logger().info(
                f'ACT 모델 로드 완료 (device={device})')
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
            Trigger, '/robot/act_grasp', self._act_grasp_service)

        self._go_home_srv = self.create_service(
            Trigger, '/robot/go_home', self._go_home_service)

        self._open_gripper_srv = self.create_service(
            Trigger, '/robot/open_gripper', self._open_gripper_service)

        self._close_gripper_srv = self.create_service(
            Trigger, '/robot/close_gripper', self._close_gripper_service)

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
        """파지 명령 수신 → ACT 파지 실행 (별도 스레드)."""
        if self._get_state() != RobotState.IDLE:
            self.get_logger().warn('파지 명령 무시: 현재 동작 중')
            return
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
        """그리퍼 열기 서비스."""
        self._set_joint_position(5, GRIPPER_OPEN)  # ID5 = 그리퍼
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 열기 완료'
        return response

    def _close_gripper_service(self, request, response):
        """그리퍼 닫기 서비스."""
        self._set_joint_position(5, GRIPPER_CLOSE)
        time.sleep(0.5)
        response.success = True
        response.message = '그리퍼 닫기 완료'
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

        # /robot/rail_done 토픽을 받아 완료 확인 (타임아웃 방식)
        # Main Orchestrator가 rail_done을 직접 구독해도 됨
        # 여기서는 단순 지연으로 처리 (추후 /motor/rail_ack로 교체 가능)
        time.sleep(0.2)  # 명령 전송 후 ESP32 처리 시작 대기

        # rail_done 발행 — ESP32 완료 신호가 없으면 타임아웃 후 완료 간주
        # 실제 배포 시: /motor/rail_ack 토픽 구독 후 done 발행 권장
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
        """분류함 위에서 그리퍼를 열어 출력물 투하."""
        self._set_state(RobotState.RELEASING)
        self._publish_status('출력물 투하')
        self.get_logger().info('출력물 투하: 그리퍼 열기')

        self._set_joint_position(5, GRIPPER_OPEN)
        time.sleep(0.8)  # 그리퍼 열림 대기

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
        time.sleep(2.0)  # 홈 자세 안정화

        self._set_state(RobotState.IDLE)
        self._publish_status('홈 복귀 완료')
        return success

    # ─────────────────────────────────────────────
    # Dynamixel 저수준 제어
    # ─────────────────────────────────────────────
    def _sync_send_positions(self, positions: List[int]) -> bool:
        """
        GroupSyncWrite로 5개 관절 위치 동시 전송.
        hardware 없으면 시뮬레이션(로그만 출력).
        """
        if not self._use_real_hardware or not self._dxl_ready:
            self.get_logger().debug(f'[SIM] 관절 목표: {positions}')
            self._latest_joint_pos = list(positions)
            return True

        try:
            self._sync_write.clearParam()
            for dxl_id, goal in zip(JOINT_IDS, positions):
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
            idx = dxl_id - 1
            if 0 <= idx < 5:
                self._latest_joint_pos[idx] = position
            return True

        try:
            result, _ = self._packet_handler.write4ByteTxRx(
                self._port_handler, dxl_id, ADDR_GOAL_POSITION, position)
            idx = dxl_id - 1
            if 0 <= idx < 5:
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
        for dxl_id in JOINT_IDS:
            try:
                val, result, _ = self._packet_handler.read4ByteTxRx(
                    self._port_handler, dxl_id, ADDR_PRESENT_POSITION)
                positions.append(int(val) if result == 0 else self._latest_joint_pos[dxl_id - 1])
            except Exception:
                positions.append(self._latest_joint_pos[dxl_id - 1])
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
        # Dynamixel 위치값 → 라디안 변환
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


# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
