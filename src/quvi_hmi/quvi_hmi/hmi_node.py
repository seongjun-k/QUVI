"""
QUVI HMI_NODE
─────────────
Flask + WebSocket 기반 Web HMI 대시보드.

기능:
  - 실시간 시스템 상태 모니터링 (WebSocket)
  - 카메라 MJPEG 스트리밍 (사이드캠 / 검사챔버 / 검사 디버그)
  - 검사 결과 히스토리 + 통계 (PASS/FAIL 카운트, 그래프)
  - 시작/정지/비상정지 제어 버튼
  - 검사 로그 이미지 뷰어

토픽:
  구독: /hmi/status, /inspection/result,
        /camera1/..., /camera2/..., /inspect/debug_image
  발행: /hmi/command (시작/정지/비상정지)
"""

import base64
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, String, Int32

from quvi_msgs.msg import InspectionResult, SystemStatus

# Flask + SocketIO
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO


# 레일 스테이션 맵 (index → {name, mm})
# 캘리브레이션 후 mm 값을 여기서만 수정하면 UI에 자동 반영된다.
RAIL_STATION_MAP = [
    {'name': 'INSPECT (A)', 'mm': 12.5},
    {'name': 'PASS (B)',    'mm': 25.0},
    {'name': 'FAIL (C)',    'mm': 125.0},
    {'name': 'BED (D)',     'mm': 381.25},
]

# Config.h 기준 변환 상수: STEPPER_STEPS_PER_REV(200) × RAIL_MICROSTEPPING(16) / RAIL_MM_PER_REV(40mm)
RAIL_STEPS_PER_MM = 80.0
RAIL_MAX_STEPS    = 33600  # RAIL_MAX_LIMIT (420mm × 80 steps/mm)

# ─── 장치 매핑 (대시보드 선택 → config 파일 → 런치가 읽음) ───
DEVICE_CONFIG_PATH = '/workspace/data/device_config.json'
RESTART_SENTINEL   = '/workspace/data/.restart_requested'
# 역할별 기본 장치(=udev 심링크). full_system.launch.py 의 기본값과 일치해야 한다.
DEVICE_DEFAULTS = {
    'sidecam_device':   '/dev/sidecam',
    'fixed_cam_device': '/dev/fixed_cam',
    'dxl_port':         '/dev/ttyFollower',
    'leader_dxl_port':  '/dev/ttyLeader',
    'micro_ros_port':   '/dev/ttyESP32',
}
DEVICE_ROLES = [
    {'key': 'sidecam_device',   'label': '사이드캠 (camera1)',   'type': 'video'},
    {'key': 'fixed_cam_device', 'label': '고정캠 (camera2)',     'type': 'video'},
    {'key': 'dxl_port',         'label': '로봇 Follower',        'type': 'serial'},
    {'key': 'leader_dxl_port',  'label': '로봇 Leader',          'type': 'serial'},
    {'key': 'micro_ros_port',   'label': 'ESP (micro-ROS)',      'type': 'serial'},
]


class HmiNode(Node):
    """ROS 2 ↔ Flask 브리지 노드."""

    def __init__(self):
        super().__init__('hmi_node')

        # ─── 파라미터 ───
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 5000)
        self.declare_parameter('debug', False)
        self.declare_parameter('sidecam_topic', '/camera1/image_raw/compressed')
        self.declare_parameter('camera2_topic', '/camera2/image_raw/compressed')
        self.declare_parameter('inspect_debug_topic', '/inspect/debug_image')
        self.declare_parameter('jpeg_quality', 70)
        self.declare_parameter('stream_fps', 15)

        self._host = self.get_parameter('host').value
        self._port = self.get_parameter('port').value
        self._debug = self.get_parameter('debug').value
        self._jpeg_quality = self.get_parameter('jpeg_quality').value
        self._stream_fps = self.get_parameter('stream_fps').value

        self._bridge = CvBridge()

        # ─── 공유 상태 (thread-safe) ───
        self._lock = threading.Lock()
        self._system_status = {
            'current_state': 'IDLE',
            'total_objects': 0,
            'processed_count': 0,
            'pass_count': 0,
            'fail_count': 0,
            'grasp_ready': False,
            'inspect_ready': False,
            'motor_ready': False,
            'teleop_active': False,
            'error_message': '',
            'joint_positions': [0.0] * 6,
            'rail_position': 0.0,   # mm 단위 float (마지막 명령 목표값)
            'turntable_angle': 0,
            'rail_station_map': RAIL_STATION_MAP,
            'led_state': False,          # LED(턴테이블 링 조명) 현재 상태
        }
        self._inspection_history = []  # 최근 100건
        self._camera_frames = {
            'sidecam': None,
            'camera2': None,
            'inspect_debug': None,
        }
        self._jpeg_cache = {
            'sidecam': None,
            'camera2': None,
            'inspect_debug': None,
        }

        # ─── ACT 모델 선택 (대시보드) ───
        self._act_models = []       # [{'name','path','step'}]
        self._act_current = {}      # {'path','name','ready','loading','use_act'}
        self._act_model_select_pub = self.create_publisher(
            String, '/robot/act_model_select', 10)

        # ─── ROS 2 Subscribers ───
        self.create_subscription(
            SystemStatus, '/hmi/status', self._status_cb, 10)
        # ACT 모델 목록/현재상태 (robot_control_node 가 latched 로 발행)
        from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
        _latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, '/robot/act_models', self._act_models_cb, _latched)
        self.create_subscription(
            String, '/robot/act_current', self._act_current_cb, _latched)
        self.create_subscription(
            InspectionResult, '/inspection/result', self._inspection_cb, 10)
        self.create_subscription(
            JointState, '/robot/joint_states', self._joint_states_cb, 10)
        # NOTE: /motor/rail 은 명령 토픽(HMI→ESP32)이므로 구독하지 않음.
        # 구독하면 자신이 발행한 명령을 즉시 수신하여 실제 위치처럼 표시되는
        # 루프백 문제가 발생한다. rail_position 은 send_rail_command() 에서
        # 직접 갱신하여 "마지막 명령 목표값"으로 표시한다.
        #
        # NOTE: /motor/turntable_cmd 도 명령 토픽(HMI→ESP32)이므로 구독하지 않음.
        # 구독하면 send_turntable_command()가 발행한 명령을 _turntable_cb가
        # 수신해 덮어쓰는 루프백이 발생한다. turntable_angle 은
        # send_turntable_command() 에서 직접 갱신한다.

        # 카메라 스트림
        sidecam_topic = self.get_parameter('sidecam_topic').value
        cam2_topic = self.get_parameter('camera2_topic').value
        inspect_topic = self.get_parameter('inspect_debug_topic').value

        self.create_subscription(
            CompressedImage, sidecam_topic,
            lambda msg: self._cam_cb(msg, 'sidecam'), 5)
        self.create_subscription(
            CompressedImage, cam2_topic,
            lambda msg: self._cam_cb(msg, 'camera2'), 5)
        self.create_subscription(
            Image, inspect_topic,
            lambda msg: self._cam_raw_cb(msg, 'inspect_debug'), 5)

        # ─── ROS 2 Publishers ───
        self._cmd_pub = self.create_publisher(String, '/hmi/command', 10)
        self._inspect_trigger_pub = self.create_publisher(Bool, '/inspection/trigger', 10)
        self._teleop_pub = self.create_publisher(Bool, '/robot/teleop_command', 10)
        self._estop_pub = self.create_publisher(Bool, '/system/estop', 10)
        # [fix] /motor/rail 퍼블리셔 타입: Float32 → Int32
        # ESP32 rail_subscription_callback이 std_msgs/Int32로 수신하므로 타입을 일치시킴.
        # send_rail_command()에서 mm → steps 변환 후 발행한다.
        self._rail_pub      = self.create_publisher(Int32,  '/motor/rail', 10)
        self._led_pub       = self.create_publisher(Bool,   '/motor/turntable_led', 10)
        self._turntable_pub = self.create_publisher(Int32,  '/motor/turntable_cmd', 10)
        self._ref_capture_pub = self.create_publisher(Bool, '/inspection/capture_reference', 10)

        # ─── 기준 이미지 캡처 턴테이블 동기화 ───
        self._ref_turntable_done_event = threading.Event()
        self.create_subscription(
            Bool, '/motor/turntable_done',
            self._ref_turntable_done_cb, 10)

        # 자율 시퀀스(P1~P6)는 오케스트레이터 + robot_control_node 단일 경로가 담당한다.
        # HMI는 모터를 직접 제어하지 않는다 (이중 제어 제거: docs/act_sequence_fix_plan.md P0).

        self.get_logger().info(
            f'HMI_NODE 초기화 완료 | http://{self._host}:{self._port}')

    # ─── ROS 콜백 ───
    def _status_cb(self, msg: SystemStatus):
        with self._lock:
            if self._system_status.get('teleop_active', False):
                if msg.current_state == 'ERROR':
                    self._system_status['current_state'] = 'ERROR'
                else:
                    self._system_status['current_state'] = 'TELEOPING'
            else:
                self._system_status['current_state'] = msg.current_state
            self._system_status['total_objects'] = msg.total_objects
            self._system_status['processed_count'] = msg.processed_count
            self._system_status['pass_count'] = msg.pass_count
            self._system_status['fail_count'] = msg.fail_count
            self._system_status['grasp_ready'] = msg.grasp_ready
            self._system_status['inspect_ready'] = msg.inspect_ready
            self._system_status['motor_ready'] = msg.motor_ready
            self._system_status['error_message'] = msg.error_message

    def _joint_states_cb(self, msg: JointState):
        with self._lock:
            self._system_status['joint_positions'] = list(msg.position)



    def _inspection_cb(self, msg: InspectionResult):
        record = {
            'timestamp': datetime.now().isoformat(),
            'passed': msg.passed,
            'fail_reason': msg.fail_reason,
            'ssim_scores': list(msg.ssim_scores),
            'area_ratios': list(msg.area_ratios),
            'pixel_diff_ratios': list(msg.pixel_diff_ratios),
            'solidity': float(msg.solidity),
            'area_ratio': float(msg.area_ratio),
            'hole_count': int(msg.hole_count),
            'hole_area_ratio': float(msg.hole_area_ratio),
            'texture_variance': float(msg.texture_variance),
            'object_index': int(msg.object_index),
            'inspection_time_sec': float(msg.inspection_time_sec),
        }
        with self._lock:
            self._inspection_history.append(record)
            if len(self._inspection_history) > 100:
                self._inspection_history.pop(0)
            # pass/fail 카운트는 오케스트레이터의 /hmi/status 가 단일 source of truth.
            # 여기서 증가시키면 _status_cb 와 이중 집계되므로 history 누적만 한다.

    def _cam_cb(self, msg: CompressedImage, key: str):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            with self._lock:
                self._camera_frames[key] = frame
                self._jpeg_cache[key] = jpeg.tobytes()

    def _cam_raw_cb(self, msg: Image, key: str):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            with self._lock:
                self._camera_frames[key] = frame
                self._jpeg_cache[key] = jpeg.tobytes()

    # ─── 제어 명령 발행 ───
    def send_command(self, command: str):
        msg = String()
        msg.data = command
        self._cmd_pub.publish(msg)
        self.get_logger().info(f'HMI 명령: {command}')
        # 명령은 /hmi/command 로만 발행한다. 실제 모터 시퀀스는 오케스트레이터가
        # 단독으로 구동한다 (이중 제어 제거: docs/act_sequence_fix_plan.md P0).

    def send_rail_command(self, mm: float):
        """레일 목표 위치 발행 (Int32, steps 단위).

        [fix] 기존 Float32(mm) 발행을 Int32(steps) 발행으로 변경.
        ESP32 rail_subscription_callback은 std_msgs/Int32를 수신하므로
        타입 불일치로 콜백이 호출되지 않던 버그를 수정한다.
        mm → steps 변환: RAIL_STEPS_PER_MM(80.0) 곱셈 후 RAIL_MAX_LIMIT 클램핑.
        """
        steps = int(round(mm * RAIL_STEPS_PER_MM))
        steps = max(0, min(RAIL_MAX_STEPS, steps))
        msg = Int32()
        msg.data = steps
        self._rail_pub.publish(msg)
        with self._lock:
            self._system_status['rail_position'] = float(mm)  # UI 표시는 mm 유지
        self.get_logger().info(f'Rail 명령: {mm:.2f} mm → {steps} steps')

    def send_turntable_command(self, angle: int):
        """턴테이블 목표 각도 발행 (Int32, 0~360°).

        /motor/turntable_cmd 는 명령 전용 토픽이므로 구독하지 않고,
        여기서 turntable_angle 을 직접 갱신한다 (루프백 방지).
        """
        angle = max(0, min(360, int(angle)))
        msg = Int32()
        msg.data = angle
        self._turntable_pub.publish(msg)
        with self._lock:
            self._system_status['turntable_angle'] = angle
        self.get_logger().info(f'턴테이블 명령: {angle}°')

    def send_led_command(self, on: bool):
        """LED(턴테이블 링 조명) ON/OFF 발행."""
        msg = Bool()
        msg.data = bool(on)
        self._led_pub.publish(msg)
        with self._lock:
            self._system_status['led_state'] = bool(on)
        self.get_logger().info(f'LED 명령: {"ON" if on else "OFF"}')

    def _ref_turntable_done_cb(self, msg: Bool):
        if msg.data:
            self._ref_turntable_done_event.set()

    # ─── ACT 모델 선택 ───
    def _act_models_cb(self, msg: String):
        import json
        try:
            with self._lock:
                self._act_models = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'ACT 모델 목록 파싱 실패: {e}')

    def _act_current_cb(self, msg: String):
        import json
        try:
            with self._lock:
                self._act_current = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'ACT 현재상태 파싱 실패: {e}')

    def send_act_model_select(self, path: str):
        """선택한 ACT 모델 경로를 robot_control_node 로 발행."""
        msg = String()
        msg.data = str(path)
        self._act_model_select_pub.publish(msg)
        self.get_logger().info(f'ACT 모델 선택 발행: {path}')

    # ─── 장치 매핑 (카메라/로봇/ESP USB) ───
    def scan_devices(self) -> dict:
        """연결 가능한 장치 후보를 스캔한다 (안정적 by-id 우선)."""
        import glob

        def _uniq(seq):
            seen, out = set(), []
            for x in seq:
                if x and x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        serial = _uniq(
            sorted(glob.glob('/dev/serial/by-id/*'))
            + sorted(glob.glob('/dev/ttyUSB*'))
            + sorted(glob.glob('/dev/ttyACM*'))
            + [p for p in ('/dev/ttyFollower', '/dev/ttyLeader', '/dev/ttyESP32')
               if os.path.exists(p)])
        video = _uniq(
            sorted(glob.glob('/dev/v4l/by-id/*'))
            + sorted(glob.glob('/dev/video*'))
            + [p for p in ('/dev/sidecam', '/dev/fixed_cam') if os.path.exists(p)])
        return {'serial': serial, 'video': video}

    def load_device_config(self) -> dict:
        """저장된 장치 매핑(없으면 기본값)을 반환."""
        cfg = {}
        try:
            with open(DEVICE_CONFIG_PATH, encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        merged = dict(DEVICE_DEFAULTS)
        merged.update({k: v for k, v in cfg.items() if k in DEVICE_DEFAULTS})
        return merged

    def save_device_config(self, cfg: dict):
        """장치 매핑을 config 파일로 저장 (알려진 키만)."""
        clean = {k: str(cfg[k]) for k in DEVICE_DEFAULTS if cfg.get(k)}
        os.makedirs(os.path.dirname(DEVICE_CONFIG_PATH), exist_ok=True)
        with open(DEVICE_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        self.get_logger().info(f'장치 설정 저장: {clean}')

    def request_restart(self, delay: int = 5):
        """delay 초 후 시스템(ros2 launch)을 재시작한다.

        run.sh 의 감시 루프가 sentinel 을 보고 relaunch 한다. 여기서는
        sentinel 을 만든 뒤, 분리된 프로세스로 잠시 후 launch 를 종료시킨다.
        """
        import subprocess
        try:
            with open(RESTART_SENTINEL, 'w'):
                pass
        except Exception as e:
            self.get_logger().error(f'재시작 sentinel 생성 실패: {e}')
            return False
        self.get_logger().warn(f'{delay}초 후 시스템 재시작')
        # 분리 세션으로 실행해 launch 종료에도 살아남아 종료 신호를 보낸다.
        # 패턴은 "bin/ros2 ..." 로 실제 launch 프로세스만 매칭해야 한다 —
        # "ros2 launch quvi_bringup" 만으로는 스크립트 본문에 같은 문자열을 가진
        # run.sh 감시 루프 bash 까지 죽어 relaunch 가 불발된다.
        # 시그널은 SIGINT — ros2 launch 는 SIGTERM 수신 시 자식 노드를 정리하지
        # 않고 고아로 남긴다 (ros2/launch#666, Jazzy 미수정).
        subprocess.Popen(
            ['bash', '-c', f'sleep {int(delay)}; pkill -INT -f "bin/ros2 launch quvi_bringup"'],
            start_new_session=True)
        return True

    def send_capture_reference_command(self, start: bool):
        """기준 이미지 캡쳐 트리거 발행 (/inspection/capture_reference).

        start=True  : 캡쳐 모드 시작 — inspect_node가 턴테이블 done 신호마다 기준 이미지를 저장한다.
        start=False : 캡쳐 모드 중단.
        """
        msg = Bool()
        msg.data = bool(start)
        self._ref_capture_pub.publish(msg)
        self.get_logger().info(f'기준 이미지 캡쳐 명령: {"START" if start else "STOP"}')

    # 수동 트리거가 허용되는 FSM 상태 (자율 시퀀스 진행 중에는 거부).
    _MANUAL_TRIGGER_SAFE_STATES = frozenset({'IDLE', 'FINISHED', 'INIT'})

    def _manual_trigger_allowed(self) -> bool:
        """오케스트레이터 FSM 이 수동 트리거를 받아도 안전한 상태인지 확인."""
        with self._lock:
            state = self._system_status.get('current_state', 'IDLE')
        return state in self._MANUAL_TRIGGER_SAFE_STATES



    def trigger_inspection(self, enable: bool) -> bool:
        """수동 검사 트리거. FSM 이 자율 시퀀스 중이면 거부하고 False 반환."""
        if enable and not self._manual_trigger_allowed():
            self.get_logger().warn('수동 검사 트리거 거부: FSM 이 자율 시퀀스 진행 중')
            return False
        msg = Bool()
        msg.data = enable
        self._inspect_trigger_pub.publish(msg)
        return True

    # ─── 데이터 접근 ───
    def get_status(self) -> dict:
        with self._lock:
            return self._system_status.copy()



    def get_inspection_history(self) -> list:
        with self._lock:
            return self._inspection_history.copy()

    def compute_stats(self) -> dict:
        """검사 통계 (#6). pass/fail/total 은 오케스트레이터 카운트를 단일 출처로
        사용하고(= /hmi/status), 평균 검사시간만 history 에서 구한다.
        history 길이로 pass/fail 를 재집계하면 오케스트레이터 카운트와 어긋난다.
        """
        with self._lock:
            passed = int(self._system_status.get('pass_count', 0))
            failed = int(self._system_status.get('fail_count', 0))
            times = [h['inspection_time_sec'] for h in self._inspection_history]
        total = passed + failed
        avg_time = sum(times) / len(times) if times else 0.0
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0.0,
            'avg_inspection_time': round(avg_time, 2),
        }

    def get_camera_jpeg(self, key: str) -> bytes | None:
        with self._lock:
            return self._jpeg_cache.get(key)


# ═══════════════════════════════════════════════
# Flask 앱
# ═══════════════════════════════════════════════

def create_flask_app(hmi_node: HmiNode) -> tuple:
    """Flask 앱 + SocketIO 생성."""
    from ament_index_python.packages import get_package_share_directory
    
    # ROS 2 share 디렉토리에서 리소스 경로 탐색 (setup.py가 리소스를 배치하는 정확한 위치)
    try:
        pkg_share_dir = get_package_share_directory('quvi_hmi')
        template_dir = os.path.join(pkg_share_dir, 'templates')
        static_dir = os.path.join(pkg_share_dir, 'static')
    except Exception:
        # 폴백: 로컬 소스 코드 파일 기준 경로
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(pkg_dir, 'templates')
        static_dir = os.path.join(pkg_dir, 'static')

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)

    # SECRET_KEY: 환경변수 우선, 없으면 랜덤 생성(운영 시 QUVI_HMI_SECRET_KEY 지정 권장).
    secret_key = os.environ.get('QUVI_HMI_SECRET_KEY')
    if not secret_key:
        secret_key = os.urandom(24).hex()
        hmi_node.get_logger().warn(
            'QUVI_HMI_SECRET_KEY 미설정 — 임시 랜덤 키 사용. '
            '운영/세션 유지가 필요하면 환경변수를 지정하세요.')
    app.config['SECRET_KEY'] = secret_key

    # CORS 허용 오리진: 기본은 로컬 LAN 시연용으로 동일 출처('*' 아님).
    # 외부 접근이 필요하면 QUVI_HMI_CORS_ORIGINS(쉼표 구분)로 명시.
    cors_env = os.environ.get('QUVI_HMI_CORS_ORIGINS', '').strip()
    if cors_env == '*':
        cors_origins = '*'
    elif cors_env:
        cors_origins = [o.strip() for o in cors_env.split(',') if o.strip()]
    else:
        cors_origins = []  # 동일 출처만 허용 (가장 안전한 기본값)

    socketio = SocketIO(app, cors_allowed_origins=cors_origins,
                        async_mode='threading')

    # ─── 페이지 라우트 ───
    @app.route('/')
    def index():
        return render_template('dashboard.html')

    # ─── REST API ───
    @app.route('/api/status')
    def api_status():
        return jsonify(hmi_node.get_status())



    # ─── ACT 모델 선택 API ───
    @app.route('/api/act/models')
    def api_act_models():
        """사용 가능한 ACT 모델 목록 + 현재 모델 상태."""
        with hmi_node._lock:
            models = list(hmi_node._act_models)
            current = dict(hmi_node._act_current)
        return jsonify({'models': models, 'current': current})

    @app.route('/api/act/select', methods=['POST'])
    def api_act_select():
        """ACT 모델 선택 → robot_control_node 로 재로드 요청 발행."""
        data = request.get_json(silent=True) or {}
        path = (data.get('path') or '').strip()
        if not path:
            return jsonify({'error': 'path 필요'}), 400
        # 알려진 모델 경로인지 검증 (임의 경로 로드 방지)
        with hmi_node._lock:
            valid = any(m.get('path') == path for m in hmi_node._act_models)
        if not valid:
            return jsonify({'error': '알 수 없는 모델 경로'}), 400
        hmi_node.send_act_model_select(path)
        return jsonify({'ok': True, 'path': path})

    # ─── 장치 매핑 API ───
    @app.route('/api/devices')
    def api_devices():
        """장치 후보 목록 + 현재 매핑 + 역할 정의."""
        return jsonify({
            'candidates': hmi_node.scan_devices(),
            'current': hmi_node.load_device_config(),
            'roles': DEVICE_ROLES,
        })

    @app.route('/api/devices/apply', methods=['POST'])
    def api_devices_apply():
        """장치 매핑 저장 후 시스템 재시작 예약 (기본 5초)."""
        data = request.get_json(silent=True) or {}
        cfg = data.get('config') or {}
        known = {k: v for k, v in cfg.items() if k in DEVICE_DEFAULTS}
        if not known:
            return jsonify({'error': '유효한 장치 항목 없음'}), 400
        try:
            merged = hmi_node.load_device_config()
            merged.update(known)
            hmi_node.save_device_config(merged)
        except Exception as e:
            return jsonify({'error': f'저장 실패: {e}'}), 500
        delay = int(data.get('delay_sec', 5))
        ok = hmi_node.request_restart(delay=delay)
        return jsonify({'ok': bool(ok), 'restart_in': delay, 'saved': merged})

    @app.route('/api/inspection/history')
    def api_inspection_history():
        return jsonify(hmi_node.get_inspection_history())

    @app.route('/api/inspection/stats')
    def api_inspection_stats():
        return jsonify(hmi_node.compute_stats())

    # ─── 제어 API ───
    @app.route('/api/command/<cmd>', methods=['POST'])
    def api_command(cmd):
        valid = ['start', 'stop', 'estop', 'reset']
        if cmd not in valid:
            return jsonify({'error': f'Unknown command: {cmd}'}), 400
        
        if cmd == 'estop':
            msg = Bool()
            msg.data = True
            hmi_node._estop_pub.publish(msg)
            
        hmi_node.send_command(cmd.upper())
        return jsonify({'ok': True, 'command': cmd})

    @app.route('/api/teleop/<action>', methods=['POST'])
    def api_teleop(action):
        if action == 'on':
            msg = Bool()
            msg.data = True
            hmi_node._teleop_pub.publish(msg)
            with hmi_node._lock:
                hmi_node._system_status['teleop_active'] = True
            return jsonify({'ok': True, 'teleop': 'on'})
        elif action == 'off':
            msg = Bool()
            msg.data = False
            hmi_node._teleop_pub.publish(msg)
            with hmi_node._lock:
                hmi_node._system_status['teleop_active'] = False
            return jsonify({'ok': True, 'teleop': 'off'})
        else:
            return jsonify({'error': f'Invalid teleop action: {action}'}), 400



    @app.route('/api/trigger/inspection', methods=['POST'])
    def api_trigger_inspection():
        if not hmi_node.trigger_inspection(True):
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 검사 트리거를 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        return jsonify({'ok': True})

    # ─── Rail 이동 API ───
    @app.route('/api/rail/move', methods=['POST'])
    def api_rail_move():
        """JSON body: {"mm": <float>} — Rail 목표 위치 발행."""
        if not hmi_node._manual_trigger_allowed():
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 레일 이동을 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        data = request.get_json(silent=True) or {}
        try:
            mm = float(data['mm'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'body에 {"mm": <float>} 필요'}), 400
        if not (0.0 <= mm <= 420.0):
            return jsonify({'error': f'범위 초과: 0~420mm (요청: {mm}mm)'}), 400
        hmi_node.send_rail_command(mm)
        return jsonify({'ok': True, 'mm': mm})

    @app.route('/api/turntable/move', methods=['POST'])
    def api_turntable_move():
        """JSON body: {"angle": <int>} — 턴테이블 목표 각도 발행."""
        if not hmi_node._manual_trigger_allowed():
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 턴테이블 이동을 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        data = request.get_json(silent=True) or {}
        try:
            angle = int(data['angle'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'body에 {"angle": <int>} 필요'}), 400
        if not (0 <= angle <= 360):
            return jsonify({'error': f'범위 초과: 0~360° (요청: {angle}°)'}), 400
        hmi_node.send_turntable_command(angle)
        return jsonify({'ok': True, 'angle': angle})

    @app.route('/api/led/toggle', methods=['POST'])
    def api_led_toggle():
        """LED 현재 상태를 반전하여 발행."""
        if not hmi_node._manual_trigger_allowed():
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 LED 제어를 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        with hmi_node._lock:
            current = hmi_node._system_status.get('led_state', False)
        new_state = not current
        hmi_node.send_led_command(new_state)
        return jsonify({'ok': True, 'led': new_state})

    @app.route('/api/led/<action>', methods=['POST'])
    def api_led_action(action):
        """LED 명시적 ON/OFF."""
        if not hmi_node._manual_trigger_allowed():
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 LED 제어를 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        if action == 'on':
            hmi_node.send_led_command(True)
            return jsonify({'ok': True, 'led': True})
        elif action == 'off':
            hmi_node.send_led_command(False)
            return jsonify({'ok': True, 'led': False})
        return jsonify({'error': f'Invalid action: {action}'}), 400

    # ─── 기준 이미지 캡쳐 API ───
    @app.route('/api/capture/reference/start', methods=['POST'])
    def api_capture_reference_start():
        """기준 이미지 캡쳐를 시작한다.

        정상품을 턴테이블에 올려둠 상태에서 호출한다.
        1. /inspection/capture_reference = True 발행
        2. 턴테이블을 0° → 90° → 180° → 270° 순서로 순환
           (inspect_node가 /motor/turntable_done 수신 시 자동 캡쳐)
        """
        data = request.get_json(silent=True) or {}
        angles = data.get('angles', [0, 90, 180, 270])
        delay  = float(data.get('delay_sec', 1.5))  # 턴테이블 회전 안정화 대기

        def _run_capture_sequence():
            import time as _time
            hmi_node.send_capture_reference_command(True)
            _time.sleep(0.3)
            for angle in angles:
                hmi_node._ref_turntable_done_event.clear()
                hmi_node.send_turntable_command(angle)
                # ESP32의 실제 turntable_done 신호를 기다림 (시간 기반 추측 제거)
                done = hmi_node._ref_turntable_done_event.wait(timeout=delay + 5.0)
                if not done:
                    hmi_node.get_logger().warn(f'기준 캡처: {angle}° turntable_done 타임아웃')
            hmi_node.get_logger().info(f'기준 이미지 캡쳐 순환 완료: {angles}')

        t = threading.Thread(target=_run_capture_sequence, daemon=True)
        t.start()
        return jsonify({'ok': True, 'angles': angles, 'delay_sec': delay})

    @app.route('/api/capture/reference/stop', methods=['POST'])
    def api_capture_reference_stop():
        """기준 이미지 캡쳐를 중단한다."""
        hmi_node.send_capture_reference_command(False)
        return jsonify({'ok': True})

    # ─── MJPEG 스트리밍 ───
    # No Signal 프레임은 정적이므로 앱 초기화 시 한 번만 인코딩해 캐싱한다
    # (루프마다 np.zeros + imencode 재생성 시 CPU 낭비).
    _blank = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(_blank, 'No Signal', (80, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    _, _blank_buf = cv2.imencode('.jpg', _blank)
    _blank_jpeg = _blank_buf.tobytes()

    def _mjpeg_generator(cam_key: str):
        while True:
            jpeg = hmi_node.get_camera_jpeg(cam_key)
            if jpeg is None:
                jpeg = _blank_jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(1.0 / hmi_node._stream_fps)

    @app.route('/stream/<cam_key>')
    def video_stream(cam_key):
        valid_keys = ['sidecam', 'camera2', 'inspect_debug']
        if cam_key not in valid_keys:
            return 'Invalid camera key', 404
        return Response(
            _mjpeg_generator(cam_key),
            mimetype='multipart/x-mixed-replace; boundary=frame')

    # ─── WebSocket: 실시간 상태 업데이트 ───
    def _ws_broadcast():
        """주기적으로 상태를 WebSocket으로 브로드캐스트."""
        while rclpy.ok():
            try:
                status = hmi_node.get_status()
                history = hmi_node.get_inspection_history()
                # pass/fail 는 오케스트레이터 카운트 단일 출처 (#6)
                stats = hmi_node.compute_stats()

                socketio.emit('status_update', {
                    'status': status,
                    'stats': stats,
                    'latest_inspection': history[-1] if history else None,
                })
            except Exception:
                pass
            time.sleep(0.1)  # threading 모드에서는 time.sleep 사용

    ws_thread = threading.Thread(target=_ws_broadcast, daemon=True)
    ws_thread.start()

    return app, socketio


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)

    hmi_node = HmiNode()
    app, socketio = create_flask_app(hmi_node)

    executor = MultiThreadedExecutor()
    executor.add_node(hmi_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    hmi_node.get_logger().info(
        f'Web HMI 시작: http://{hmi_node._host}:{hmi_node._port}')

    # 시연용 Werkzeug 개발 서버 허용 여부. 운영 WSGI(eventlet/gevent) 사용 시
    # QUVI_HMI_ALLOW_DEV_SERVER=0 으로 끄고 적절한 서버로 구동할 수 있다.
    allow_dev_server = os.environ.get(
        'QUVI_HMI_ALLOW_DEV_SERVER', '1').lower() in ('1', 'true', 'yes')
    if allow_dev_server:
        hmi_node.get_logger().warn(
            'Werkzeug 개발 서버로 구동 중 (시연용). 운영 배포 시 '
            'eventlet/gevent 기반 WSGI 서버 사용을 권장합니다.')

    try:
        socketio.run(
            app,
            host=hmi_node._host,
            port=hmi_node._port,
            debug=hmi_node._debug,
            allow_unsafe_werkzeug=allow_dev_server,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        hmi_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
