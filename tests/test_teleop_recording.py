"""텔레옵 자체 녹화: 에피소드 점검(변환 스크립트)과 TeleopRecorder 저장 검증."""
import importlib.util
import os
from pathlib import Path

import numpy as np

from quvi_robot_control.teleop_recorder import TeleopRecorder

HOME_RAW = [2037, 1114, 2771, 3257, 2016, 2500]

_spec = importlib.util.spec_from_file_location(
    'teleop_episodes_to_lerobot',
    Path(__file__).resolve().parents[1] / 'scripts' / 'teleop_episodes_to_lerobot.py')
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def _episode(seconds=5.0, start_offset_raw=0, close=True):
    t = np.arange(0, seconds, 1 / 15)
    home_rad = (np.array(HOME_RAW) - 2048) / conv.RAW_PER_RAD
    state = np.tile(home_rad, (len(t), 1))
    state[:, :5] += start_offset_raw / conv.RAW_PER_RAD
    action = state.copy()
    if close:
        action[len(t) // 2:, 5] -= 0.5
    return t, state, action


def test_good_episode_passes():
    assert conv.check_episode(*_episode(), HOME_RAW) == []


def test_too_short():
    reasons = conv.check_episode(*_episode(seconds=1.0), HOME_RAW)
    assert any('길이' in r for r in reasons)


def test_start_off_home():
    reasons = conv.check_episode(*_episode(start_offset_raw=50), HOME_RAW)
    assert any('홈' in r for r in reasons)


def test_gripper_not_closed():
    reasons = conv.check_episode(*_episode(close=False), HOME_RAW)
    assert any('그리퍼' in r for r in reasons)


def test_frame_gap():
    t, state, action = _episode()
    t[len(t) // 2:] += 1.0
    reasons = conv.check_episode(t, state, action, HOME_RAW)
    assert any('간격' in r for r in reasons)


def test_stop_without_frames_returns_none(tmp_path):
    rec = TeleopRecorder(str(tmp_path), HOME_RAW, on_error=lambda e: None)
    rec.start(0.0)
    assert rec.stop() is None


def test_rate_is_15hz_under_50hz_loop(tmp_path):
    rec = TeleopRecorder(str(tmp_path), HOME_RAW, on_error=lambda e: None)
    rec.start(0.0)
    for k in range(100):  # 50Hz 로 2초
        now = k / 50
        if rec.due(now):
            rec.add(now, [0.0] * 6, [0.0] * 6, b'a', b'b')
    assert 29 <= len(rec._rows) <= 31


def test_save_roundtrip(tmp_path):
    rec = TeleopRecorder(str(tmp_path), HOME_RAW, on_error=lambda e: None)
    rec.start(100.0)
    for k in range(10):
        rec.add(100.0 + k / 15, [0.1] * 6, [0.2] * 6, b'side%d' % k, b'top%d' % k)
    rec.stop().join()
    names = os.listdir(tmp_path)
    assert len(names) == 1 and names[0].startswith('ep_') and not names[0].endswith('.tmp')
    ep = conv.load_episode(tmp_path / names[0])
    assert len(ep['t']) == 10 and ep['t'][0] == 0
    assert (tmp_path / names[0] / '00000_top.jpg').read_bytes() == b'top0'


def test_double_stop_saves_once(tmp_path):
    rec = TeleopRecorder(str(tmp_path), HOME_RAW, on_error=lambda e: None)
    rec.start(0.0)
    rec.add(0.0, [0.0] * 6, [0.0] * 6, b'a', b'b')
    rec.stop().join()
    assert rec.stop() is None
    assert len(os.listdir(tmp_path)) == 1


def test_add_after_stop_ignored(tmp_path):
    rec = TeleopRecorder(str(tmp_path), HOME_RAW, on_error=lambda e: None)
    rec.start(0.0)
    rec.stop()
    rec.add(1.0, [0.0] * 6, [0.0] * 6, b'a', b'b')
    assert rec._rows == []


def test_save_failure_calls_on_error(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('x')  # 폴더 대신 파일이라 makedirs 가 실패한다
    errors = []
    rec = TeleopRecorder(str(blocked), HOME_RAW, on_error=errors.append)
    rec.start(0.0)
    rec.add(0.0, [0.0] * 6, [0.0] * 6, b'a', b'b')
    rec.stop().join()
    assert len(errors) == 1


def test_nan_rejected():
    t, state, action = _episode()
    state[3, 0] = np.nan
    assert any('NaN' in r for r in conv.check_episode(t, state, action, HOME_RAW))


def test_bad_shape_rejected():
    t, state, action = _episode()
    reasons = conv.check_episode(t, state[:, :5], action, HOME_RAW)
    assert any('shape' in r for r in reasons)
