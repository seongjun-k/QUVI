"""턴테이블 재파지 후 분류함까지 높은 자세로 이동하는지 확인한다(하드웨어 불필요)."""
import math
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from quvi_robot_control import robot_control_node as rc  # noqa: E402


def _tool_height_mm(pose):
    """URDF 평면 FK 로 joint2 기준 손끝 높이(mm)를 구한다."""
    k = 2 * math.pi / 4096
    a = (pose['shoulder_lift'] - 2048) * k
    z = -0.0415 * math.sin(a) + 0.11315 * math.cos(a)
    a += (pose['elbow_flex'] - 2048) * k
    z -= 0.162 * math.sin(a)
    a += (pose['wrist_flex'] - 2048) * k
    z -= 0.12063 * math.sin(a)
    return z * 1000


def test_transit_is_about_50mm_above_sort_up():
    rise = _tool_height_mm(rc.POSE_SORT_TRANSIT) - _tool_height_mm(rc.POSE_SORT_UP)
    assert 45 < rise < 55


def test_pick_from_chamber_ends_in_transit_pose():
    moves = []
    robot = SimpleNamespace(
        _cmd_gen=1,
        _set_state_if_current=lambda *_a: None,
        _publish_status=lambda _m: None,
        get_logger=lambda: SimpleNamespace(info=lambda _m: None, error=lambda _m: None),
        _write_raw_position=lambda *_a, **_k: True,
        _wait_gripper=lambda *_a: None,
        _move_arm=lambda pose: moves.append(pose) or True,
        _check_grasp_success=lambda: True,
        _should_abort=lambda: False,
        _pick_chamber_done_pub=SimpleNamespace(publish=lambda _m: None),
    )
    assert rc.RobotControlNode._execute_pick_from_chamber(robot) is True
    assert moves == [rc.POSE_P4, rc.POSE_P3, rc.POSE_SORT_UP, rc.POSE_SORT_TRANSIT]



def test_turntable_offset_applies_to_both_entry_and_place():
    assert rc.POSE_P3['shoulder_pan'] == -1 + rc.TURNTABLE_PAN_OFFSET
    assert rc.POSE_P4['shoulder_pan'] == 16 + rc.TURNTABLE_PAN_OFFSET
