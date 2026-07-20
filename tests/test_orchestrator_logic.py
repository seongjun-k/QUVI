"""
HMI 수동 트리거 가드 순수 로직 테스트.

hmi_node 는 모듈 임포트 시 rclpy 가 필요하므로,
ROS 없이 검증 가능한 순수 판정 규칙만 독립적으로 확인한다.
"""


# ─── HMI 수동 트리거 안전 상태 규칙 (hmi_node._MANUAL_TRIGGER_SAFE_STATES) ───
SAFE_STATES = frozenset({'IDLE', 'FINISHED', 'INIT'})


def manual_trigger_allowed(current_state):
    return current_state in SAFE_STATES


def test_manual_trigger_allowed_in_idle():
    assert manual_trigger_allowed('IDLE')
    assert manual_trigger_allowed('FINISHED')


def test_manual_trigger_blocked_mid_sequence():
    for s in ('GRASPING_WAIT', 'INSPECTING_STEP_90', 'SORTING_TRIGGER'):
        assert not manual_trigger_allowed(s)
