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
from flask import Flask, Response, jsonify, render_template, send_from_directory
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

        # ─── ROS 2 Subscribers ───
        self.create_subscription(
            SystemStatus, '/hmi/status', self._status_cb, 10)
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



    @app.route('/api/inspection/history')
    def api_inspection_history():
        return jsonify(hmi_node.get_inspection_history())

    @app.route('/api/inspection/stats')
    def api_inspection_stats():
        history = hmi_node.get_inspection_history()
        total = len(history)
        passed = sum(1 for h in history if h['passed'])
        failed = total - passed
        avg_time = (sum(h['inspection_time_sec'] for h in history) / total
                    if total > 0 else 0.0)
        return jsonify({
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0.0,
            'avg_inspection_time': round(avg_time, 2),
        })

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
        from flask import request
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
        from flask import request
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

    # ─── MJPEG 스트리밍 ───
    def _mjpeg_generator(cam_key: str):
        while True:
            jpeg = hmi_node.get_camera_jpeg(cam_key)
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, 'No Signal', (80, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
                _, buf = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
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

                total = len(history)
                passed = sum(1 for h in history if h['passed'])

                socketio.emit('status_update', {
                    'status': status,
                    'stats': {
                        'total': total,
                        'passed': passed,
                        'failed': total - passed,
                        'pass_rate': round((passed / total * 100), 1) if total > 0 else 0.0,
                    },
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
