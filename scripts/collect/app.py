"""VLA 에피소드 수집 웹 앱 (포트 5100) — HMI 와 별개로 수집할 때만 실행한다.

구독: camera1·camera3 압축 이미지, /robot/teleop_record_state (텔레옵 중에만 발행됨)
텔레옵 시작/정지는 이 앱이 하지 않는다(HMI 에서). 원본 폴더 형식: docs/_local/collect_raw_format.md

실행(컨테이너 quvi-dev):
  source /opt/ros/jazzy/setup.bash
  PYTHONPATH=/workspace/src/quvi_robot_control:$PYTHONPATH python3 /workspace/scripts/collect/app.py
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import rclpy
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_socketio import SocketIO
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_store as store  # noqa: E402
from quvi_robot_control import camera_profile, topics  # noqa: E402
from quvi_robot_control.act_common import (  # noqa: E402
    EPISODE_DURATION_SEC, JOINT_NAMES, RECORD_START_POSE, rad_to_raw)

# ─── 상수 ───
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(APP_DIR, '..', '..'))
DATA_DIR = os.path.join(REPO, 'data')  # 로봇 launch 의 data_dir(/workspace/data)과 같은 곳 — 카메라 프로필 공유
DEVICE_CONFIG_PATH = os.path.join(DATA_DIR, 'device_config.json')
# hmi_node.DEVICE_DEFAULTS 의 사이드캠·탑뷰 값과 같아야 한다(HMI 는 import 하지 않는다)
DEVICE_DEFAULTS = {'sidecam_device': '/dev/sidecam', 'topcam_device': '/dev/topcam'}
CAMERA_SPEC = {  # camera -> (토픽, device_config 키)
    'camera1': ('/camera1/image_raw/compressed', 'sidecam_device'),
    'camera3': ('/camera3/image_raw/compressed', 'topcam_device'),
}
V4L2_CTRLS = ('exposure_time_absolute', 'gain', 'brightness', 'contrast', 'saturation',
              'white_balance_temperature', 'auto_exposure', 'white_balance_automatic',
              'exposure_dynamic_framerate')
CONVERT_PY = os.path.join(REPO, 'data', 'collect_tools', 'venv', 'bin', 'python')
CONVERT_SCRIPT = os.path.join(REPO, 'scripts', 'build_lerobot_dataset.py')

MOVE_THRESH_RAW = 8      # 준비 시점 평균 대비 어느 관절이든 이만큼 움직이면 "움직임 감지"
BUFFER_SEC = 1.5         # 시작 전 pre-roll 확보용 링버퍼 길이
STATE_STALE_SEC = 1.0    # 이보다 오래 상태가 안 오면 "텔레옵 미실행"
FINALIZE_MARGIN_SEC = 0.1  # 종료 시각 이후 지연 도착 샘플 대기
HOME_RAW = [RECORD_START_POSE[j] for j in JOINT_NAMES]
_ID_RE = re.compile(r'^[\w\-]+$')


def _load_devices() -> dict:
    """data/device_config.json 을 기본값에 병합(hmi_node.load_device_config 와 같은 방식)."""
    merged = dict(DEVICE_DEFAULTS)
    try:
        with open(DEVICE_CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
        merged.update({k: v for k, v in cfg.items() if k in DEVICE_DEFAULTS})
    except (OSError, ValueError):
        pass
    return merged


def read_v4l2_ctrls(dev: str) -> dict:
    """v4l2-ctl 값 읽기. 실패해도 예외 없이 error 로 남긴다(녹화를 막지 않음)."""
    ctrls, err = {}, None
    for name in V4L2_CTRLS:
        try:
            r = subprocess.run(['v4l2-ctl', '-d', dev, f'--get-ctrl={name}'],
                               capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired) as e:
            err = str(e)
            break
        if r.returncode == 0 and ':' in r.stdout:
            try:  # 메뉴형은 "auto_exposure: 1 (Manual Mode)" 처럼 설명이 붙는다
                ctrls[name] = int(r.stdout.split(':', 1)[1].split()[0])
                continue
            except ValueError:
                pass
        err = (r.stderr or r.stdout).strip() or f'{name} 읽기 실패'
    out = {'ctrls': ctrls}
    if not ctrls and err:
        out['error'] = err
    return out


def list_cam_ctrls(dev: str) -> dict:
    """설정 패널용: 프로필 대상 컨트롤의 현재값·범위. 실패하면 ValueError."""
    try:
        r = subprocess.run(['v4l2-ctl', '-d', dev, '--list-ctrls'],
                           capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f'{dev} 읽기 실패: {e}') from e
    ctrls = camera_profile.parse_list_ctrls(r.stdout)
    if not ctrls:
        raise ValueError(f'{dev} 읽기 실패: {(r.stderr or r.stdout).strip() or "컨트롤 없음"}')
    return ctrls


def apply_cam_ctrls(dev: str, ctrls: dict) -> str:
    """v4l2 값 쓰기. 장치 오류 문구를 돌려준다(빈 문자열 = 성공). 결과 확인은 다시 읽어서 한다."""
    arg = camera_profile.set_ctrl_arg(ctrls)
    if not arg:
        return ''
    try:
        r = subprocess.run(['v4l2-ctl', '-d', dev, '--set-ctrl', arg],
                           capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)
    return r.stderr.strip() if r.returncode else ''


# ─── ROS 노드 + 녹화 상태기계 ───
class CollectNode(Node):
    """phase: idle(세션 없음) → align → ready → recording → result(saving) → align ..."""

    def __init__(self, data_root: str):
        super().__init__('collect_app')
        self.data_root = data_root
        self.emit = lambda *a, **k: None  # create 후 socketio.emit 으로 교체
        self.lock = threading.RLock()
        self._cams = {c: [] for c in store.CAMERAS}      # (t, jpeg bytes), 시간 오름차순
        self._latest = {c: None for c in store.CAMERAS}  # 미리보기용 최신 (t, bytes)
        self._states = []                                # (t, present_rad[6], goal_rad[6])
        self._last_state_t = None
        self._dev = None
        self.phase = 'idle'
        self.session_id = None
        self.error = None
        self._align_since = None
        self._ref_raw = None
        self._t0 = None
        self._t0_wall = None
        self._pending = None
        self.saved_count = 0

        for cam, (topic, _) in CAMERA_SPEC.items():
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, c=cam: self._cam_cb(c, msg), qos_profile_sensor_data)
        self.create_subscription(
            Float32MultiArray, topics.TOPIC_ROBOT_TELEOP_RECORD_STATE,
            self._state_cb, qos_profile_sensor_data)

    # ─── 수신 (모든 샘플에 앱 수신 시각) ───
    def _cam_cb(self, cam, msg):
        t = time.monotonic()
        data = bytes(msg.data)
        with self.lock:
            self._cams[cam].append((t, data))
            self._latest[cam] = (t, data)

    def _state_cb(self, msg):
        t = time.monotonic()
        d = list(msg.data)
        if len(d) != 12:
            return
        with self.lock:
            self._states.append((t, d[:6], d[6:]))
            self._last_state_t = t
            self._dev = [rad_to_raw(r) - h for r, h in zip(d[:6], HOME_RAW)]
            try:
                self._on_state(t, d[:6])
            except Exception as e:  # 녹화기 예외는 녹화만 중단
                self._fail(e)

    def _on_state(self, t, present):
        if self.phase == 'align':
            if store.start_aligned(self._dev):
                if self._align_since is None:
                    self._align_since = t
                elif t - self._align_since >= store.START_HOLD_SEC:
                    win = [s[1] for s in self._states if s[0] >= t - store.START_HOLD_SEC]
                    self._ref_raw = [sum(rad_to_raw(p[i]) for p in win) / len(win) for i in range(6)]
                    self.phase = 'ready'
            else:
                self._align_since = None
        elif self.phase == 'ready':
            raw = [rad_to_raw(r) for r in present]
            if max(abs(a - b) for a, b in zip(raw, self._ref_raw)) > MOVE_THRESH_RAW:
                self._t0 = t - store.PRE_ROLL_SEC
                self._t0_wall = (datetime.now().astimezone()
                                 - timedelta(seconds=time.monotonic() - self._t0)
                                 ).isoformat(timespec='milliseconds')
                self.phase = 'recording'

    def _fail(self, e):
        self.get_logger().error(f'녹화기 예외: {e!r}')
        self.error = f'녹화 중단: {e}'
        self._reset_to_align()

    def _reset_to_align(self):
        self.phase = 'align' if self.session_id else 'idle'
        self._align_since = None
        self._t0 = None
        self._pending = None

    # ─── 주기 처리 (링버퍼 정리, 녹화 종료, 상태 방송) ───
    def tick_loop(self):
        last_emit = 0.0
        while True:
            time.sleep(0.02)
            now = time.monotonic()
            try:
                with self.lock:
                    self._tick(now)
                if now - last_emit >= 0.1:
                    last_emit = now
                    self.emit('status', self.get_status())
            except Exception as e:
                with self.lock:
                    self._fail(e)

    def _tick(self, now):
        cutoff = now - BUFFER_SEC
        if self.phase == 'recording':
            cutoff = min(cutoff, self._t0 - store.PRE_ROLL_SEC - 0.3)
            if now >= self._t0 + EPISODE_DURATION_SEC + FINALIZE_MARGIN_SEC:
                self._finalize()
        elif self.phase in ('align', 'ready') and not self.teleop_alive(now):
            self.phase = 'align'
            self._align_since = None
        self._states = [s for s in self._states if s[0] >= cutoff]
        for c in self._cams:
            self._cams[c] = [f for f in self._cams[c] if f[0] >= cutoff]

    def _finalize(self):
        t0, lo, hi = self._t0, self._t0 - store.PRE_ROLL_SEC, self._t0 + EPISODE_DURATION_SEC
        state = [(t - t0, p, g) for t, p, g in self._states if lo <= t <= hi]
        cams = {c: [(t - t0, d) for t, d in self._cams[c] if lo <= t <= hi] for c in self._cams}
        meta = store.compute_meta(self._t0_wall, state, cams)
        self._pending = {'state': state, 'cams': cams, 'meta': meta}
        self.phase = 'result'

    def teleop_alive(self, now=None):
        now = now or time.monotonic()
        return self._last_state_t is not None and now - self._last_state_t < STATE_STALE_SEC

    # ─── 외부 조회·제어 ───
    def get_status(self) -> dict:
        now = time.monotonic()
        with self.lock:
            alive = self.teleop_alive(now)

            def age(t):
                return None if t is None else round(now - t, 2)

            if self.phase in ('ready', 'recording', 'result', 'saving'):
                hold = 1.0
            elif self.phase == 'align' and self._align_since is not None:
                hold = min(1.0, (self._last_state_t - self._align_since) / store.START_HOLD_SEC)
            else:
                hold = 0.0
            return {
                'phase': self.phase, 'teleop': alive, 'joints': JOINT_NAMES,
                'dev': [round(d) for d in self._dev] if alive and self._dev else None,
                'tol': store.START_TOL_RAW, 'tol_gripper': store.START_TOL_GRIPPER_RAW,
                'hold': hold,
                'elapsed': round(now - self._t0, 2) if self.phase == 'recording' else None,
                'duration': EPISODE_DURATION_SEC, 'target': store.TARGET_RETURN_SEC,
                'ages': {'state': age(self._last_state_t),
                         **{c: age(self._latest[c][0] if self._latest[c] else None) for c in self._cams}},
                'max_gap': store.MAX_GAP_SEC,
                'session': self.session_id, 'saved_count': self.saved_count,
                'result': self._pending['meta'] if self._pending else None,
                'error': self.error,
            }

    def latest_jpeg(self, cam):
        with self.lock:
            return self._latest[cam]

    def start_session(self, product: str, task: str) -> tuple:
        name = store.sanitize_name(product)
        if not name:
            raise ValueError('품종 이름이 비어 있음')
        with self.lock:
            if self.phase not in ('idle', 'align', 'ready'):
                raise ValueError('녹화/결과 처리 중에는 세션을 바꿀 수 없음')
        devices = _load_devices()
        profile = camera_profile.load(DATA_DIR)
        cameras, warnings = {}, []
        for cam, (topic, key) in CAMERA_SPEC.items():  # v4l2 읽기는 느릴 수 있어 락 밖에서
            dev = devices[key]
            # 카메라를 다시 꽂거나 재부팅하면 값이 기본으로 돌아가므로 세션마다 프로필을 다시 건다
            want = profile.get(cam, {})
            err = apply_cam_ctrls(dev, want)
            cameras[cam] = {'topic': topic, 'device': dev, **read_v4l2_ctrls(dev)}
            if want:
                cameras[cam]['profile'] = want
                bad = camera_profile.mismatches(want, cameras[cam].get('ctrls', {}))
                if err or bad:
                    cameras[cam]['profile_mismatch'] = ([err] if err else []) + bad
                    warnings.append(f'{cam} 프로필 불일치: ' + '; '.join(cameras[cam]['profile_mismatch']))
            else:
                warnings.append(f'{cam} 카메라 프로필 없음 — 카메라 설정에서 맞추고 저장하세요')
        sid = f'{datetime.now():%Y%m%d_%H%M%S}_{name}'
        sdir = os.path.join(self.data_root, sid)
        os.makedirs(sdir)
        with open(os.path.join(sdir, 'session.json'), 'w', encoding='utf-8') as f:
            json.dump(store.new_session_json(product.strip(), task, cameras), f,
                      ensure_ascii=False, indent=2)
        with self.lock:
            self.session_id = sid
            self.saved_count = 0
            self.error = None
            self._reset_to_align()
        return sid, warnings

    def end_session(self):
        with self.lock:
            if self.phase in ('recording', 'result', 'saving'):
                raise ValueError('녹화/결과 처리 중에는 세션을 끝낼 수 없음')
            self.session_id = None
            self._reset_to_align()

    def save_pending(self) -> int:
        with self.lock:
            if self.phase != 'result' or not self._pending:
                raise ValueError('저장할 결과가 없음')
            pending, self.phase = self._pending, 'saving'
        try:
            idx = store.write_episode(os.path.join(self.data_root, self.session_id),
                                      pending['state'], pending['cams'], pending['meta'])
        except Exception:
            with self.lock:
                self.phase = 'result'
            raise
        with self.lock:
            self.saved_count += 1
            self._reset_to_align()
        return idx

    def discard_pending(self):
        with self.lock:
            if self.phase != 'result':
                raise ValueError('버릴 결과가 없음')
            self._reset_to_align()


# ─── Flask ───
def create_flask_app(node: CollectNode, data_root: str):
    app = Flask(__name__, template_folder=os.path.join(APP_DIR, 'templates'),
                static_folder=os.path.join(APP_DIR, 'static'))
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    socketio = SocketIO(app, cors_allowed_origins=[], async_mode='threading')

    convert = {'running': False, 'lines': [], 'rc': None, 'out': None}
    convert_lock = threading.Lock()

    def _sdir(sid) -> str:
        if not isinstance(sid, str) or not _ID_RE.match(sid) or \
                not os.path.isdir(os.path.join(data_root, sid)):
            raise ValueError(f'세션을 찾을 수 없음: {sid}')
        return os.path.join(data_root, sid)

    def _guard_busy(sids):
        if node.session_id in sids and node.phase in ('recording', 'result', 'saving'):
            raise ValueError('녹화/결과 처리 중인 세션은 편집할 수 없음')

    for exc in (ValueError, KeyError, OSError):
        app.register_error_handler(
            exc, lambda e: (jsonify(ok=False, error=str(e)), 400))

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/status')
    def api_status():
        return jsonify(node.get_status())

    @app.route('/stream/<cam>')
    def stream(cam):
        if cam not in store.CAMERAS:
            return 'Invalid camera', 404

        def gen():
            last = None
            while True:
                item = node.latest_jpeg(cam)
                if item and item[0] != last:
                    last = item[0]
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + item[1] + b'\r\n')
                time.sleep(0.03)
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # ─── 녹화 ───
    @app.route('/api/session', methods=['POST'])
    def api_session_start():
        d = request.get_json(force=True)
        sid, warnings = node.start_session(d.get('product', ''), d.get('task') or store.DEFAULT_TASK)
        return jsonify(ok=True, session=sid, warnings=warnings)

    # ─── 카메라 설정 (VLA 입력 카메라 노출·화이트밸런스) ───
    def _cam_dev(cam) -> str:
        if cam not in CAMERA_SPEC:
            raise ValueError(f'알 수 없는 카메라: {cam}')
        return _load_devices()[CAMERA_SPEC[cam][1]]

    def _guard_cam_change():
        # 에피소드 도중 영상 조건이 바뀌면 그 에피소드는 못 쓴다
        if node.phase in ('recording', 'result', 'saving'):
            raise ValueError('녹화/결과 처리 중에는 카메라 설정을 바꿀 수 없음')

    def _cam_view():
        profile = camera_profile.load(DATA_DIR)
        out = {}
        for cam in CAMERA_SPEC:
            dev = _cam_dev(cam)
            try:
                out[cam] = {'device': dev, 'ctrls': list_cam_ctrls(dev)}
            except ValueError as e:
                out[cam] = {'device': dev, 'error': str(e)}
            out[cam]['profile'] = profile.get(cam)
        return out

    @app.route('/api/camera')
    def api_camera():
        return jsonify(ok=True, cameras=_cam_view())

    @app.route('/api/camera/set', methods=['POST'])
    def api_camera_set():
        d = request.get_json(force=True)
        _guard_cam_change()
        ctrls = {k: int(v) for k, v in (d.get('ctrls') or {}).items() if k in camera_profile.CTRL_ORDER}
        err = apply_cam_ctrls(_cam_dev(d.get('camera')), ctrls)
        return jsonify(ok=not err, error=err or None, cameras=_cam_view())

    @app.route('/api/camera/profile/save', methods=['POST'])
    def api_camera_profile_save():
        # 지금 장치 값을 그대로 저장한다(입력칸 값이 아니라 실제로 적용된 값)
        profile = {}
        for cam in CAMERA_SPEC:
            profile[cam] = camera_profile.from_ctrls(list_cam_ctrls(_cam_dev(cam)))
        camera_profile.save(DATA_DIR, profile)
        return jsonify(ok=True, cameras=_cam_view())

    @app.route('/api/camera/profile/apply', methods=['POST'])
    def api_camera_profile_apply():
        _guard_cam_change()
        profile = camera_profile.load(DATA_DIR)
        if not profile:
            raise ValueError('저장된 프로필 없음')
        errs = [f'{cam}: {e}' for cam in CAMERA_SPEC
                if (e := apply_cam_ctrls(_cam_dev(cam), profile.get(cam, {})))]
        return jsonify(ok=not errs, error='; '.join(errs) or None, cameras=_cam_view())

    @app.route('/api/session/end', methods=['POST'])
    def api_session_end():
        node.end_session()
        return jsonify(ok=True)

    @app.route('/api/episode/save', methods=['POST'])
    def api_episode_save():
        return jsonify(ok=True, index=node.save_pending())

    @app.route('/api/episode/discard', methods=['POST'])
    def api_episode_discard():
        node.discard_pending()
        return jsonify(ok=True)

    # ─── 데이터 관리 ───
    @app.route('/api/sessions')
    def api_sessions():
        return jsonify(sessions=store.list_sessions(data_root), current=node.session_id,
                       dataset_root=os.path.join(REPO, 'data', 'datasets'),
                       convert_ready=os.path.isfile(CONVERT_PY) and os.path.isfile(CONVERT_SCRIPT))

    @app.route('/api/frame/<sid>/<int:ep>/<cam>/<pos>')
    def api_frame(sid, ep, cam, pos):
        if cam not in store.CAMERAS or not (pos in ('first', 'mid', 'last') or pos.isdigit()):
            return 'bad request', 400
        return send_file(store.frame_path(_sdir(sid), ep, cam, pos), mimetype='image/jpeg',
                         max_age=3600)  # 재생 시 프리로드한 프레임을 다시 받지 않게

    @app.route('/api/episode/<sid>/<int:ep>/stamps')
    def api_episode_stamps(sid, ep):
        return jsonify(ok=True, stamps=store.episode_stamps(_sdir(sid), ep),
                       duration=EPISODE_DURATION_SEC, target=store.TARGET_RETURN_SEC)

    @app.route('/api/review', methods=['POST'])
    def api_review():
        d = request.get_json(force=True)
        store.set_review(_sdir(d['session']), int(d['episode']), d.get('review'))
        return jsonify(ok=True)

    @app.route('/api/delete', methods=['POST'])
    def api_delete():
        d = request.get_json(force=True)
        _guard_busy([d['session']])
        n = store.delete_episodes(_sdir(d['session']), [int(i) for i in d['episodes']],
                                  bool(d.get('renumber')))
        return jsonify(ok=True, deleted=n)

    @app.route('/api/merge', methods=['POST'])
    def api_merge():
        d = request.get_json(force=True)
        sids = d['sessions']
        if len(set(sids)) != len(sids):
            raise ValueError('같은 세션이 중복 선택됨')
        _guard_busy(sids)
        out_id = store.sanitize_name(d.get('out_name') or '') or f'{datetime.now():%Y%m%d_%H%M%S}_merged'
        if out_id in sids:
            raise ValueError('대상 세션이 소스와 같음')
        n = store.merge_sessions([_sdir(s) for s in sids], os.path.join(data_root, out_id),
                                 bool(d.get('move')))
        return jsonify(ok=True, session=out_id, episodes=n)

    # ─── 변환 (도구 venv 의 build_lerobot_dataset.py 를 subprocess 로) ───
    def _run_convert(cmd):
        # ROS PYTHONPATH 가 섞이면 도구 venv 에서 임포트 에러가 난다(변환기가 act_common 경로는 스스로 넣음)
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        env['PYTHONUNBUFFERED'] = '1'
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 cwd=REPO, env=env)
            for line in p.stdout:
                with convert_lock:
                    convert['lines'] = (convert['lines'] + [line.rstrip('\n')])[-5000:]
            rc = p.wait()
        except OSError as e:
            rc = -1
            with convert_lock:
                convert['lines'].append(f'실행 실패: {e}')
        with convert_lock:
            convert.update(running=False, rc=rc)

    @app.route('/api/convert', methods=['GET', 'POST'])
    def api_convert():
        if request.method == 'GET':
            with convert_lock:
                return jsonify(ok=True, **convert)
        if not (os.path.isfile(CONVERT_PY) and os.path.isfile(CONVERT_SCRIPT)):
            return jsonify(ok=False, missing=True, error='변환 도구 미설치'), 400
        d = request.get_json(force=True)
        out = d['out']
        if not os.path.isabs(out) or (os.path.isdir(out) and os.listdir(out)):
            raise ValueError('출력 경로는 절대경로이고 비어 있거나 없는 폴더여야 함')
        sdirs = [_sdir(s) for s in d['sessions']]
        if not sdirs:
            raise ValueError('변환할 세션을 선택')
        _guard_busy(d['sessions'])
        with convert_lock:
            if convert['running']:
                raise ValueError('이미 변환 중')
            convert.update(running=True, lines=[], rc=None, out=out)
        threading.Thread(target=_run_convert, daemon=True,
                         args=([CONVERT_PY, CONVERT_SCRIPT, *sdirs, '--out', out],)).start()
        return jsonify(ok=True)

    return app, socketio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.join(REPO, 'data', 'episodes_raw'))
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5100)
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    os.makedirs(data_root, exist_ok=True)

    rclpy.init()
    node = CollectNode(data_root)
    app, socketio = create_flask_app(node, data_root)
    node.emit = socketio.emit
    threading.Thread(target=node.tick_loop, daemon=True).start()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.get_logger().info(f'수집 앱 시작: http://{args.host}:{args.port} (데이터 {data_root})')
    try:
        socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True,
                     use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
