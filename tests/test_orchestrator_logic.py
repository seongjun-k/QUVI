"""
오케스트레이터 픽셀→로봇좌표 변환 및 HMI 트리거 가드 순수 로직 테스트.

main_orchestrator_node / hmi_node 는 모듈 임포트 시 rclpy 가 필요하므로,
ROS 없이 검증 가능한 순수 계산/판정 규칙만 독립적으로 확인한다.
변환식이 노드 구현과 동일하게 유지되는지에 대한 회귀 가드 역할.
"""


def px_to_robot(px, py, px_to_mm_x, px_to_mm_y, offset_x, offset_y):
    """main_orchestrator_node.GRASPING_TRIGGER 의 좌표 변환과 동일한 식.

    goal.target_x = obj.x * px_to_mm_x + offset_x
    goal.target_y = obj.y * px_to_mm_y + offset_y
    """
    return (px * px_to_mm_x + offset_x, py * px_to_mm_y + offset_y)


def test_px_to_robot_identity_offset():
    x, y = px_to_robot(0, 0, 1.0, 1.0, 0.0, 0.0)
    assert (x, y) == (0.0, 0.0)


def test_px_to_robot_affine():
    # 기본 데모 파라미터 (px_to_mm=0.5, offset=100)
    x, y = px_to_robot(200, 40, 0.5, 0.5, 100.0, 100.0)
    assert x == 200.0  # 200*0.5 + 100
    assert y == 120.0  # 40*0.5 + 100


def test_px_to_robot_monotonic():
    a = px_to_robot(10, 10, 0.5, 0.5, 0.0, 0.0)
    b = px_to_robot(20, 20, 0.5, 0.5, 0.0, 0.0)
    assert b[0] > a[0] and b[1] > a[1]


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
