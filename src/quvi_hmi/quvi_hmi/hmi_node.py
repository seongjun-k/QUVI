"""
QUVI HMI_NODE
─────────────
Flask + WebSocket 기반 Web HMI 대시보드.

기능:
  - 실시간 시스템 상태 모니터링 (WebSocket)
  - 카메라 MJPEG 스트리밍 (핸드캠 / 검사챔버 / YOLO 디버그 / 검사 디버그)
  - 검사 결과 히스토리 + 통계 (PASS/FAIL 카운트, 그래프)
  - 시작/정지/비상정지 제어 버튼
  - 검사 로그 이미지 뷰어

토픽:
  구독: /hmi/status, /detection/objects, /inspection/result,
        /camera1/..., /camera2/..., /yolo/debug_image, /inspect/debug_image
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

from quvi_msgs.msg import InspectionResult, ObjectArray, SystemStatus

# Flask + SocketIO
from flask import Flask, Response, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO


# 레일 스테이션 맵 (index → {name, steps})
# 캘리브레이션 후 steps 값을 여기서만 수정하면 UI에 자동 반영된다.
RAIL_STATION_MAP = [
    {'name': 'BED (D)',      'steps': 0},
    {'name': 'INSPECT (A)', 'steps': 1000},
    {'name': 'PASS (B)',    'steps': 1700},
    {'name': 'FAIL (C)',    'steps': 2400},
]


class HmiNode(Node):
    """ROS 2 ↔ Flask 브리지 노드."""

    def __init__(self):
        super().__init__('hmi_node')

        # ─── 파라미터 ───
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 5000)
        self.declare_parameter('debug', False)
        self.declare_parameter('camera1_topic', '/camera1/image_raw/compressed')
        self.declare_parameter('camera2_topic', '/camera2/image_raw/compressed')
        self.declare_parameter('yolo_debug_topic', '/yolo/debug_image')
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
            'yolo_ready': False,
            'grasp_ready': False,
            'inspect_ready': False,
            'motor_ready': False,
            'teleop_active': False,
            'error_message': '',
            'joint_positions': [0.0] * 6,
            'rail_position': 0,
            'turntable_angle': 0,
            'rail_station_map': RAIL_STATION_MAP,
        }
        self._latest_detections = []
        self._inspection_history = []  # 최근 100건
        self._camera_frames = {
            'camera1': None,
            'camera2': None,
            'yolo_debug': None,
            'inspect_debug': None,
        }
        self._jpeg_cache = {
            'camera1': None,
            'camera2': None,
            'yolo_debug': None,
            'inspect_debug': None,
        }

        # ─── ROS 2 Subscribers ───
        self.create_subscription(
            SystemStatus, '/hmi/status', self._status_cb, 10)
        self.create_subscription(
            ObjectArray, '/detection/objects', self._detection_cb, 10)
        self.create_subscription(
            InspectionResult, '/inspection/result', self._inspection_cb, 10)
        self.create_subscription(
            JointState, '/robot/joint_states', self._joint_states_cb, 10)
        self.create_subscription(
            Int32, '/robot/rail_command', self._rail_command_cb, 10)
        self.create_subscription(
            Int32, '/motor/turntable_cmd', self._turntable_cb, 10)

        # 카메라 스트림
        cam1_topic = self.get_parameter('camera1_topic').value
        cam2_topic = self.get_parameter('camera2_topic').value
        yolo_topic = self.get_parameter('yolo_debug_topic').value
        inspect_topic = self.get_parameter('inspect_debug_topic').value

        self.create_subscription(
            CompressedImage, cam1_topic,
            lambda msg: self._cam_cb(msg, 'camera1'), 5)
        self.create_subscription(
            CompressedImage, cam2_topic,
            lambda msg: self._cam_cb(msg, 'camera2'), 5)
        self.create_subscription(
            Image, yolo_topic,
            lambda msg: self._cam_raw_cb(msg, 'yolo_debug'), 5)
        self.create_subscription(
            Image, inspect_topic,
            lambda msg: self._cam_raw_cb(msg, 'inspect_debug'), 5)

        # ─── ROS 2 Publishers ───
        self._cmd_pub = self.create_publisher(String, '/hmi/command', 10)
        self._trigger_pub = self.create_publisher(Bool, '/detection/trigger', 10)
        self._inspect_trigger_pub = self.create_publisher(Bool, '/inspection/trigger', 10)
        self._teleop_pub = self.create_publisher(Bool, '/robot/teleop_command', 10)
        self._estop_pub = self.create_publisher(Bool, '/system/estop', 10)

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
            self._system_status['yolo_ready'] = msg.yolo_ready
            self._system_status['grasp_ready'] = msg.grasp_ready
            self._system_status['inspect_ready'] = msg.inspect_ready
            self._system_status['motor_ready'] = msg.motor_ready
            self._system_status['error_message'] = msg.error_message

    def _joint_states_cb(self, msg: JointState):
        with self._lock:
            # ROS 2 JointState positions are floats
            self._system_status['joint_positions'] = list(msg.position)

    def _rail_command_cb(self, msg: Int32):
        with self._lock:
            self._system_status['rail_position'] = msg.data

    def _turntable_cb(self, msg: Int32):
        with self._lock:
            self._system_status['turntable_angle'] = msg.data

    def _detection_cb(self, msg: ObjectArray):
        with self._lock:
            self._latest_detections = [{
                'x': float(o.x), 'y': float(o.y),
                'width': float(o.width), 'height': float(o.height),
                'confidence': float(o.confidence),
                'class_name': o.class_name,
            } for o in msg.objects]

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

    # 수동 트리거가 허용되는 FSM 상태 (자율 시퀀스 진행 중에는 거부).
    _MANUAL_TRIGGER_SAFE_STATES = frozenset({'IDLE', 'FINISHED', 'INIT'})

    def _manual_trigger_allowed(self) -> bool:
        """오케스트레이터 FSM 이 수동 트리거를 받아도 안전한 상태인지 확인."""
        with self._lock:
            state = self._system_status.get('current_state', 'IDLE')
        return state in self._MANUAL_TRIGGER_SAFE_STATES

    def trigger_detection(self, enable: bool) -> bool:
        """수동 감지 트리거. FSM 이 자율 시퀀스 중이면 거부하고 False 반환."""
        if enable and not self._manual_trigger_allowed():
            self.get_logger().warn('수동 감지 트리거 거부: FSM 이 자율 시퀀스 진행 중')
            return False
        msg = Bool()
        msg.data = enable
        self._trigger_pub.publish(msg)
        return True

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

    def get_detections(self) -> list:
        with self._lock:
            return self._latest_detections.copy()

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

    @app.route('/api/detections')
    def api_detections():
        return jsonify(hmi_node.get_detections())

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

    @app.route('/api/trigger/detection', methods=['POST'])
    def api_trigger_detection():
        if not hmi_node.trigger_detection(True):
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 감지 트리거를 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        return jsonify({'ok': True})

    @app.route('/api/trigger/inspection', methods=['POST'])
    def api_trigger_inspection():
        if not hmi_node.trigger_inspection(True):
            return jsonify({
                'ok': False,
                'error': '자율 시퀀스 진행 중에는 수동 검사 트리거를 사용할 수 없습니다. '
                         'STOP 후 다시 시도하세요.',
            }), 409
        return jsonify({'ok': True})

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
        valid_keys = ['camera1', 'camera2', 'yolo_debug', 'inspect_debug']
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
                detections = hmi_node.get_detections()
                history = hmi_node.get_inspection_history()

                total = len(history)
                passed = sum(1 for h in history if h['passed'])

                socketio.emit('status_update', {
                    'status': status,
                    'detections': detections,
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
