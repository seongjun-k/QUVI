"""
robot_control_node 의 상태머신 선점 로직(_try_acquire_state / _set_state_if_current) 회귀 테스트.

robot_control_node.py 는 모듈 임포트 시 rclpy/lerobot 이 필요하므로,
기존 tests/test_orchestrator_logic.py 패턴을 따라 ROS 없이 로직만
동일하게 재현한 스텁으로 검증한다 (2026-07-11 리뷰 #2 대응).
"""
import threading
from enum import IntEnum


class RobotState(IntEnum):
    """robot_control_node.RobotState 와 동일 (IDLE/ERROR 관계만 검증에 필요)."""
    IDLE          = 0
    HOMING        = 1
    MOVING_RAIL   = 2
    ROTATING_BASE = 3
    ACT_GRASPING  = 4
    ERROR         = 99


class _StubNode:
    """RobotControlNode._try_acquire_state / _set_state_if_current 와 동일 로직."""

    def __init__(self, initial_state=RobotState.IDLE):
        self._state = initial_state
        self._state_lock = threading.Lock()
        self._cmd_gen = 0
        self._abort_event = threading.Event()

    def _publish_status(self, _name):
        pass  # 실제 노드는 여기서 토픽 퍼블리시 — 테스트에서는 불필요

    def get_logger(self):
        class _NullLogger:
            def warn(self, _msg):
                pass
        return _NullLogger()

    def _try_acquire_state(self, target_state, allow_error=False):
        with self._state_lock:
            allowed = (RobotState.IDLE, RobotState.ERROR) if allow_error else (RobotState.IDLE,)
            if self._state not in allowed:
                self.get_logger().warn(f'명령 무시: 현재 {self._state.name} 동작 중')
                return False
            self._state = target_state
            self._cmd_gen += 1
            self._publish_status(self._state.name)
        self._abort_event.clear()
        return True

    def _set_state(self, state):
        with self._state_lock:
            self._state = state

    def _set_state_if_current(self, state, gen):
        if gen != self._cmd_gen:
            self.get_logger().warn(
                f'잔존 스레드의 상태 변경 무시: {state.name} (gen {gen} != {self._cmd_gen})')
            return
        self._set_state(state)


def test_acquire_from_idle_succeeds_and_transitions():
    node = _StubNode(RobotState.IDLE)
    ok = node._try_acquire_state(RobotState.MOVING_RAIL)
    assert ok is True
    assert node._state == RobotState.MOVING_RAIL
    assert node._cmd_gen == 1


def test_acquire_blocked_while_busy():
    node = _StubNode(RobotState.MOVING_RAIL)
    ok = node._try_acquire_state(RobotState.ACT_GRASPING)
    assert ok is False
    assert node._state == RobotState.MOVING_RAIL  # 상태 불변


def test_acquire_from_error_blocked_by_default_allowed_with_flag():
    node = _StubNode(RobotState.ERROR)
    assert node._try_acquire_state(RobotState.HOMING) is False
    assert node._state == RobotState.ERROR

    node2 = _StubNode(RobotState.ERROR)
    ok = node2._try_acquire_state(RobotState.HOMING, allow_error=True)
    assert ok is True
    assert node2._state == RobotState.HOMING


def test_set_state_if_current_ignores_stale_gen():
    node = _StubNode(RobotState.IDLE)
    node._try_acquire_state(RobotState.HOMING)  # cmd_gen -> 1
    stale_gen = 0
    node._set_state_if_current(RobotState.IDLE, stale_gen)
    assert node._state == RobotState.HOMING  # 잔존 스레드 무시, 상태 불변


def test_set_state_if_current_applies_matching_gen():
    node = _StubNode(RobotState.IDLE)
    node._try_acquire_state(RobotState.HOMING)  # cmd_gen -> 1
    node._set_state_if_current(RobotState.IDLE, node._cmd_gen)
    assert node._state == RobotState.IDLE
