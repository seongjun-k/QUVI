"""정책 실행 중 그리퍼를 천천히 닫고, 닫는 동안 팔을 세워 두는지 확인한다(하드웨어 불필요)."""
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from quvi_robot_control import robot_control_node as rc  # noqa: E402
from quvi_robot_control.act_common import JOINT_NAMES, raw_to_rad  # noqa: E402


def _robot():
    writes = []
    present = {'shoulder_pan': 2000, 'shoulder_lift': 1500, 'elbow_flex': 2500,
               'wrist_flex': 2900, 'wrist_roll': 2040, 'gripper': 2460}
    bus = SimpleNamespace(sync_read=lambda *_a, **_k: dict(present),
                          sync_write=lambda _reg, goal, **_k: writes.append(dict(goal)))
    r = SimpleNamespace(_use_real_hardware=True, _dxl_ready=True, _dxl_io_lock=threading.Lock(),
                        _follower=SimpleNamespace(bus=bus), _act_max_rel_target=200.0,
                        _clip_safe_targets=lambda goal, is_raw: goal, _act_trace=None,
                        _act_grip_cmd=None, _act_held_arm=None, writes=writes)
    r._limit_gripper_closing = lambda g, p: rc.RobotControlNode._limit_gripper_closing(r, g, p)
    return r


def _action(lift, grip):
    raw = {'shoulder_pan': 2000, 'shoulder_lift': lift, 'elbow_flex': 2500,
           'wrist_flex': 2900, 'wrist_roll': 2040, 'gripper': grip}
    return [raw_to_rad(raw[n]) for n in JOINT_NAMES]


def _send(r, action):
    return rc.RobotControlNode._act_send_action_step(r, action)


def test_closing_holds_arm_and_steps_gripper():
    r = _robot()
    assert _send(r, _action(1500, 2460)) is True          # 열린 채 이동 — 그대로 반영

    steps = 0
    while not _send(r, _action(1300, 2060)):              # 닫으면서 팔을 드는 액션
        steps += 1
        assert steps < 20
    closing = r.writes[1:-1]
    assert all(w['shoulder_lift'] == 1500 for w in closing)   # 닫는 동안 팔은 그대로
    assert [w['gripper'] for w in closing[:2]] == [2430, 2400]
    assert r.writes[-1]['shoulder_lift'] == 1300              # 다 닫은 뒤에야 팔이 움직임
    assert r.writes[-1]['gripper'] == 2060


def test_opening_and_small_changes_are_not_delayed():
    r = _robot()
    assert _send(r, _action(1500, 2100)) is False             # 첫 닫힘은 천천히
    r2 = _robot()
    r2._act_grip_cmd = 2100
    assert _send(r2, _action(1400, 2480)) is True             # 여는 동작은 바로
    assert r2.writes[-1]['gripper'] == 2480
    r3 = _robot()
    r3._act_grip_cmd = 2440
    assert _send(r3, _action(1400, 2425)) is True             # 작은 떨림으로 팔이 멈추지 않음
